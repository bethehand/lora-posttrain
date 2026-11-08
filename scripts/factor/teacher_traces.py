#!/usr/bin/env python3
"""Have a teacher model work through ideas with test_factor, following the research manual.

The teacher sees the manual; the saved trajectory holds only what the student will see: the
task prompt, the teacher's replies and the real tool results. Each trajectory is then checked by
the gates, on rows up to 2024 only, so SFT never reads the reward year or the sealed test.

    python scripts/factor/teacher_traces.py --out results/factor/teacher/trial.jsonl --ideas volume_surge_down --rollouts 3

Three ways to widen the data:
    --paraphrases FILE --variants 6   also use other wordings of each idea
    --strategies                      one trajectory per research strategy, for SFT2 and offline RL
    --recover-from FILE[,FILE]        start from a student's broken formula, marked as not to be
                                      trained on, so the teacher shows how to get out of an error
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs", "factor_idea"))

from openai import AsyncOpenAI  # noqa: E402

from factor_idea import workers  # noqa: E402
from factor_idea.prompt import TOOL_SYSTEM_PROMPT, extract_factor  # noqa: E402

MAX_TURNS = 6
_TAGGED = re.compile(r"<factor>.*?</factor>", re.DOTALL)
# The student never sees the manual, so a trajectory that talks about it or its sources teaches noise.
_LEAK = re.compile(r"manual|handbook|guideline|手册|Strategy10|S10\b|notebook|ai_0[12]|\bneg_[a-z]|instructions? say", re.IGNORECASE)
# Keep in step with FactorToolset.test_factor: the student's tool definition must be the same.
TOOL = {
    "type": "function",
    "function": {
        "name": "test_factor",
        "description": "Compute a candidate factor on July 2022 to December 2024. Returns an error if the expression is invalid; otherwise its rank IC and incremental IC, overall and by period.",
        "parameters": {
            "properties": {"expression": {"title": "Expression", "type": "string"}},
            "required": ["expression"],
            "title": "test_factorArguments",
            "type": "object",
        },
    },
}
# The eight ways of working for SFT2: each idea is worked once per strategy and the best stays.
STRATEGIES = {
    "start_simple": "Begin with the plainest reading of the idea, and add something only when a result gives you a reason to.",
    "who_to_compare": "Think about what the coin should be compared with: its own past, other coins on the same day, or the market. Try more than one.",
    "compare_windows": "Compare short, medium and long windows for the same quantity before settling on one.",
    "compare_forms": "Compare different forms of the same quantity: its level, a ratio to its own past, its rank in its own history.",
    "denoise": "Look for a less noisy version of the quantity, by smoothing or by using a median, and check whether that helps.",
    "extremes": "Think about what extreme days do to the formula. Try compressing them, or ranking before averaging, and check whether that helps.",
    "stability": "Prefer a formula whose incremental IC has the same sign and a similar size in all three periods, even if its overall number is a little lower.",
    "less_style": "Compare the rank IC with the incremental IC. If most of the signal is a known style, change the form so that less of it is.",
}

TEACHER_RULES = """You are writing worked examples that a smaller model will learn research from, so your visible text matters as much as your formulas. Follow the research manual below.

How to write:
- Write in plain English, for someone learning the craft.
- Before every test_factor call, write one or two sentences in the reply itself: what you are checking and why this form. A call with no sentence before it is a failure.
- After a result, say in one sentence what you read from it before changing anything.
- Use the manual's conventions: min_periods about 80% of the window, rounded up; an EMA's min_periods about 80% of its span, and no window argument.
- Stop as soon as two tests in a row fail to beat your best incremental IC by at least 0.002. Five tests is a limit, not a target.
- The reader has only the task and the tool results. Never mention the manual, these instructions, where a rule came from, or the name of any existing factor. Give the reason itself.
- If the conversation already holds a call that failed, it counts toward the five tests. Read the error, say in one sentence what was wrong, and fix it. Do not comment on who wrote it.
- The factor you hand in must be one you tested. Finish with two or three sentences on what you found, then the factor between <factor> and </factor>. Be honest if the idea looks wrong on this history, and keep its direction.

Research manual (Chinese):
"""


def teacher_system(manual_path: str, strategy: str | None = None) -> str:
    with open(manual_path) as f:
        text = TEACHER_RULES + f.read()
    return f"{text}\n\nIn this example, work this way: {STRATEGIES[strategy]}" if strategy else text


async def episode(client, args, system: str, idea: dict, rollout: int, text: str, variant: int, strategy: str | None, prefix: list[dict]) -> dict:
    student_prompt = f"{TOOL_SYSTEM_PROMPT}\n\nIdea: {text}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": student_prompt}]
    saved = [{"role": "user", "content": student_prompt}]
    tests, stop = 0, "max_turns"
    for i, turn in enumerate(prefix):
        ids = [f"student_{i}_{j}" for j in range(len(turn["calls"]))]
        messages.append({
            "role": "assistant",
            "content": turn["content"] or None,
            # DeepSeek's thinking mode refuses a past tool call without this field.
            "reasoning_content": "",
            "tool_calls": [
                {"id": c, "type": "function", "function": {"name": "test_factor", "arguments": json.dumps({"expression": e})}}
                for c, e in zip(ids, turn["calls"])
            ],
        })
        saved.append({"role": "assistant", "content": turn["content"], "calls": list(turn["calls"]), "train": False})
        for c, e in zip(ids, turn["calls"]):
            result = await workers.report(e)
            tests += 1
            messages.append({"role": "tool", "tool_call_id": c, "content": result})
            saved.append({"role": "tool", "content": result})
    for _ in range(MAX_TURNS - len(prefix)):
        reply = await client.chat.completions.create(model=args.model, messages=messages, tools=[TOOL], max_tokens=args.max_tokens)
        message = reply.choices[0].message
        calls = message.tool_calls or []
        content = (message.content or "").strip()
        messages.append(message.model_dump(exclude_none=True))
        content = _TAGGED.sub(lambda m: m.group(0).replace('"', "'"), content)
        entry = {"role": "assistant", "content": content, "calls": []}
        saved.append(entry)
        if not calls:
            stop = "answered"
            break
        for call in calls:
            try:
                expression = json.loads(call.function.arguments or "{}").get("expression", "")
            except ValueError:
                expression = ""
            result = await workers.report(expression)
            tests += 1
            # Single quotes need no escaping inside OLMo's call format, which it gets wrong easily.
            entry["calls"].append(expression.replace('"', "'"))
            saved.append({"role": "tool", "content": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    final = extract_factor(saved[-1].get("content", "")) if saved[-1]["role"] == "assistant" else None
    tested = [e for m in saved if m["role"] == "assistant" for e in m["calls"]]
    score = await workers.score(final or "", idea["key"], idea["direction"], tuple(idea.get("also", ())))
    return dict(
        idea=idea["id"],
        family=idea["family"],
        rollout=rollout,
        variant=variant,
        strategy=strategy,
        recovered=bool(prefix),
        stop=stop,
        tests=tests,
        factor=final,
        final_was_tested=final is not None and final.strip() in [e.strip() for e in tested],
        parsed=score.parsed,
        computes=score.computes,
        matches_idea=score.matches_idea,
        reason=score.reason,
        reasons_before_calls=sum(1 for m in saved if m["role"] == "assistant" and m["calls"] and m["content"]),
        leaks_the_manual=any(_LEAK.search(m["content"]) for m in saved if m["role"] == "assistant"),
        messages=saved,
    )


def broken_starts(paths: list[str], wanted: set[str], limit_per_idea: int) -> dict[str, list[list[dict]]]:
    """From evals of the student: the turns up to its first test that came back as an error, each broken formula once."""
    starts: dict[str, list[list[dict]]] = {}
    seen = set()
    for path in paths:
        with open(path) as f:
            for row in map(json.loads, f):
                if row.get("idea") not in wanted:
                    continue
                steps, turns = row.get("steps") or [], []
                for j, step in enumerate(steps):
                    if step["role"] != "assistant" or not step.get("calls"):
                        continue
                    result = steps[j + 1]["content"] if j + 1 < len(steps) and steps[j + 1]["role"] == "tool" else ""
                    calls = [c if isinstance(c, str) else json.dumps(c) for c in step["calls"]]
                    expressions = []
                    for c in calls:
                        try:
                            expressions.append(json.loads(c).get("expression", ""))
                        except (ValueError, AttributeError):
                            expressions.append("")
                    if len(expressions) == 1 and expressions[0] and result.startswith(("error:", "does not compute")):
                        turns = [{"content": step.get("content") or "", "calls": expressions}]
                        break
                    if result.startswith("computes"):
                        break
                key = (row["idea"], turns[0]["calls"][0]) if turns else None
                if turns and key not in seen and len(starts.setdefault(row["idea"], [])) < limit_per_idea:
                    seen.add(key)
                    starts[row["idea"]].append(turns)
    return starts


async def main_async(args) -> list[dict]:
    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env], timeout=300, max_retries=5)
    with open(args.ideas_file) as f:
        ideas = [json.loads(line) for line in f]
    if args.ideas:
        wanted = set(args.ideas.split(","))
        ideas = [i for i in ideas if i["id"] in wanted]
    if args.split != "all":
        ideas = [i for i in ideas if i["split"] == args.split]
    wordings = {}
    if args.paraphrases:
        with open(args.paraphrases) as f:
            wordings = {r["id"]: r["paraphrases"] for r in map(json.loads, f)}
    strategies = list(STRATEGIES) if args.strategies else [None]
    systems = {s: teacher_system(args.manual, s) for s in strategies}
    starts = broken_starts(args.recover_from.split(","), {i["id"] for i in ideas}, args.recover_per_idea) if args.recover_from else {}
    gate = asyncio.Semaphore(args.concurrency)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    output = open(args.out, "w")

    async def run(idea, rollout, text, variant, strategy, prefix):
        async with gate:
            try:
                row = await episode(client, args, systems[strategy], idea, rollout, text, variant, strategy, prefix)
            except Exception as exc:  # one lost episode shouldn't cost the rest of a long run
                print(f"lost {idea['id']} {variant} {strategy} {rollout}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                return None
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        return row

    jobs = []
    for idea in ideas:
        texts = [idea["en"], *wordings.get(idea["id"], [])][: args.variants]
        if args.recover_from:
            for rollout, prefix in enumerate(starts.get(idea["id"], []) * args.recover_repeats):
                jobs.append(run(idea, rollout, texts[rollout % len(texts)], rollout % len(texts), None, prefix))
            continue
        for variant, text in enumerate(texts):
            for strategy in strategies:
                for rollout in range(args.rollouts):
                    jobs.append(run(idea, rollout, text, variant, strategy, []))
    print(f"{len(jobs)} episodes", flush=True)
    return [r for r in await asyncio.gather(*jobs) if r is not None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ideas", default="")
    ap.add_argument("--ideas-file", default="task/ideas_draft.jsonl")
    ap.add_argument("--manual", default="task/manual.zh.md")
    ap.add_argument("--split", default="all")
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--paraphrases", default="")
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--strategies", action="store_true")
    ap.add_argument("--recover-from", default="")
    ap.add_argument("--recover-per-idea", type=int, default=4)
    ap.add_argument("--recover-repeats", type=int, default=1)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    rows = asyncio.run(main_async(args))
    print(f"{len(rows)} episodes written to {args.out}; passed {sum(bool(r['matches_idea']) for r in rows)}, leaks {sum(bool(r['leaks_the_manual']) for r in rows)}")
    for r in rows[:: max(1, len(rows) // 12)]:
        print(r["idea"], r["rollout"], "stop", r["stop"], "tests", r["tests"], "tested_final", r["final_was_tested"], "passed", r["matches_idea"], "reasons", r["reasons_before_calls"], repr(r["reason"])[:50], (r["factor"] or "")[:110])


if __name__ == "__main__":
    main()
