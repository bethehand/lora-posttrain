#!/usr/bin/env python3
"""Ask a model to turn each idea into a factor, and run every answer through the sandbox.

Works with any OpenAI-compatible endpoint: a local vLLM server or a hosted API. Scores on
the RL reward period only (2025); the sealed test is never read here.

    python scripts/factor/baseline.py --name olmo-base --model /path/to/model \
        --base-url http://127.0.0.1:18000/v1 --samples 8
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs", "factor_idea"))

from openai import AsyncOpenAI  # noqa: E402

from factor_idea import sandbox as sb  # noqa: E402
from factor_idea.panel import load_dev  # noqa: E402
from factor_idea.prompt import SYSTEM_PROMPT, extract_factor  # noqa: E402


async def ask(client, model, idea, sample, args, gate) -> dict:
    async with gate:
        for attempt in range(4):
            try:
                reply = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": idea["en"]}],
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                choice = reply.choices[0]
                return dict(idea=idea["id"], sample=sample, reply=choice.message.content or "", finish=choice.finish_reason)
            except Exception as exc:  # the API can rate-limit or drop a request; a late answer is still an answer
                if attempt == 3:
                    return dict(idea=idea["id"], sample=sample, reply="", finish=f"error: {type(exc).__name__}")
                await asyncio.sleep(2**attempt * 3)


async def collect(args, ideas) -> list[dict]:
    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ.get(args.api_key_env, "EMPTY"), timeout=300)
    gate = asyncio.Semaphore(args.concurrency)
    jobs = [ask(client, args.model, idea, s, args, gate) for idea in ideas for s in range(args.samples)]
    return await asyncio.gather(*jobs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key-env", default="NO_KEY")
    ap.add_argument("--ideas", default="task/ideas_draft.jsonl")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default="results/factor/baselines")
    ap.add_argument("--rescore", action="store_true", help="score the saved replies again instead of asking the model")
    args = ap.parse_args()

    with open(args.ideas) as f:
        ideas = {i["id"]: i for i in map(json.loads, f)}
    started = time.monotonic()
    if args.rescore:
        with open(f"{args.out}/{args.name}.jsonl") as f:
            answers = [{k: r[k] for k in ("idea", "sample", "reply", "finish")} for r in map(json.loads, f)]
    else:
        answers = asyncio.run(collect(args, list(ideas.values())))
    print(f"{len(answers)} answers in {time.monotonic() - started:.0f}s")

    box = sb.Sandbox(load_dev())
    rows = []
    for a in answers:
        idea = ideas[a["idea"]]
        text = extract_factor(a["reply"])
        result = box.score(text or "", idea["key"], idea["direction"], also=tuple(idea.get("also", ())))
        ic = result.ic
        rows.append(dict(
            **a,
            factor=text,
            family=idea["family"],
            key=idea["key"],
            direction=idea["direction"],
            parsed=result.parsed,
            computes=result.computes,
            matches_idea=result.matches_idea,
            reason=result.reason if text else "no factor tag",
            skeleton=result.skeleton,
            ic=ic.incremental if ic else None,
            raw_ic=ic.raw if ic else None,
            reward_gates=result.reward("gates"),
            reward_ic=result.reward("ic"),
            seconds=round(result.seconds, 3),
        ))
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{args.name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    tagged = sum(r["factor"] is not None for r in rows)
    parsed = sum(r["parsed"] for r in rows)
    computes = sum(r["computes"] for r in rows)
    passed = [r for r in rows if r["matches_idea"]]
    ics = [r["ic"] for r in passed if r["ic"] is not None and np.isfinite(r["ic"])]
    summary = dict(
        name=args.name,
        model=args.model,
        samples=n,
        tagged=tagged / n,
        parsed=parsed / n,
        computes=computes / n,
        passed=len(passed) / n,
        distinct_skeletons_among_passed=len({r["skeleton"] for r in passed}),
        ic_mean_among_passed=float(np.mean(ics)) if ics else None,
        ic_median_among_passed=float(np.median(ics)) if ics else None,
        reward_gates_mean=float(np.mean([r["reward_gates"] for r in rows])),
        reward_ic_mean=float(np.mean([r["reward_ic"] for r in rows])),
        seconds_median=float(np.median([r["seconds"] for r in rows])),
        seconds_max=float(np.max([r["seconds"] for r in rows])),
        reasons=collections.Counter(
            (r["reason"].split(":")[0] if r["reason"] else "") for r in rows if not r["matches_idea"]
        ).most_common(),
        passed_by_key={k: float(np.mean([r["matches_idea"] for r in rows if r["key"] == k])) for k in sorted({r["key"] for r in rows})},
        passed_by_idea={k: float(np.mean([r["matches_idea"] for r in rows if r["idea"] == k])) for k in ideas},
    )
    with open(f"{args.out}/{args.name}.summary.json", "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "passed_by_idea"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
