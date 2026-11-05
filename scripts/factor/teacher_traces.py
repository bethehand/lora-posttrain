#!/usr/bin/env python3
"""Have a teacher model work through ideas with test_factor, following the research manual.

The teacher sees the manual; the saved trajectory holds only what the student will see: the
task prompt, the teacher's replies and the real tool results. Each trajectory is then checked by
the sandbox, on rows up to 2024 only, so SFT never touches the reward year or the sealed test.

    python scripts/factor/teacher_traces.py --out results/factor/teacher/trial.jsonl --ideas volume_surge_down --rollouts 3
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

from factor_idea import dsl, sandbox as sb, workers  # noqa: E402
from factor_idea.prompt import TOOL_SYSTEM_PROMPT, extract_factor  # noqa: E402

MAX_TURNS = 6
_TAGGED = re.compile(r"<factor>.*?</factor>", re.DOTALL)
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
TEACHER_RULES = """You are writing worked examples that a smaller model will learn research from, so your visible text matters as much as your formulas. Follow the research manual below.

How to write:
- Write in plain English, for someone learning the craft.
- Before every test_factor call, write one or two sentences in the reply itself: what you are checking and why this form. A call with no sentence before it is a failure.
- After a result, say in one sentence what you read from it before changing anything.
- Change one thing at a time.
- Use the manual's conventions: min_periods about 80% of the window, rounded up; an EMA's window ten times its span and its min_periods about 80% of its span.
- Stop as soon as two tests in a row bring no clear improvement. Five tests is a limit, not a target.
- The factor you hand in must be one you tested. Finish with two or three sentences on what you found, then the factor between <factor> and </factor>. Be honest if the idea looks wrong on this history, and keep its direction.

Research manual (Chinese):
"""


def teacher_system(manual_path: str) -> str:
    with open(manual_path) as f:
        # The row of a formula whose date is unconfirmed stays away from the teacher.
        lines = [line for line in f if "待确认：这个公式 2026" not in line]
    return TEACHER_RULES + "".join(lines)


async def episode(client, args, system: str, idea: dict, rollout: int) -> dict:
    student_prompt = f"{TOOL_SYSTEM_PROMPT}\n\nIdea: {idea['en']}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": student_prompt}]
    saved = [{"role": "user", "content": student_prompt}]
    tests, stop = 0, "max_turns"
    for _ in range(MAX_TURNS):
        reply = await client.chat.completions.create(model=args.model, messages=messages, tools=[TOOL], max_tokens=args.max_tokens)
        message = reply.choices[0].message
        calls = message.tool_calls or []
        content = (message.content or "").strip()
        messages.append(message.model_dump(exclude_none=True))
        content = _TAGGED.sub(lambda m: m.group(0).replace('"', "'"), content)
        saved.append({"role": "assistant", "content": content, "calls": []})
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
            saved[-1]["calls"].append(expression.replace('"', "'"))
            saved.append({"role": "tool", "content": result})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    final = extract_factor(saved[-1].get("content", "")) if saved[-1]["role"] == "assistant" else None
    tested = [e for m in saved if m["role"] == "assistant" for e in m["calls"]]
    score = await workers.score(final or "", idea["key"], idea["direction"], tuple(idea.get("also", ())))
    return dict(
        idea=idea["id"],
        family=idea["family"],
        rollout=rollout,
        stop=stop,
        tests=tests,
        factor=final,
        final_was_tested=final is not None and final.strip() in [e.strip() for e in tested],
        parsed=score.parsed,
        computes=score.computes,
        matches_idea=score.matches_idea,
        reason=score.reason,
        reasons_before_calls=sum(1 for m in saved if m["role"] == "assistant" and m["calls"] and m["content"]),
        messages=saved,
    )


async def main_async(args) -> list[dict]:
    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env], timeout=300)
    with open(args.ideas_file) as f:
        ideas = [json.loads(line) for line in f]
    if args.ideas:
        wanted = set(args.ideas.split(","))
        ideas = [i for i in ideas if i["id"] in wanted]
    system = teacher_system(args.manual)
    gate = asyncio.Semaphore(args.concurrency)

    async def run(idea, rollout):
        async with gate:
            return await episode(client, args, system, idea, rollout)

    return await asyncio.gather(*(run(i, r) for i in ideas for r in range(args.rollouts)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ideas", default="")
    ap.add_argument("--ideas-file", default="task/ideas_draft.jsonl")
    ap.add_argument("--manual", default="task/manual.zh.md")
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    rows = asyncio.run(main_async(args))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in rows:
        print(r["idea"], r["rollout"], "stop", r["stop"], "tests", r["tests"], "tested_final", r["final_was_tested"], "passed", r["matches_idea"], "reasons", r["reasons_before_calls"], repr(r["reason"])[:50], (r["factor"] or "")[:110])


if __name__ == "__main__":
    main()
