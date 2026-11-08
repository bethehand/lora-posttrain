#!/usr/bin/env python3
"""Ask the LLM judge about every passing formula of an eval: the same question for every stage and arm.

Adds `faithful` to each passing row in place, so compare_stages.py can read it. A formula that gets no
verdict has `faithful` null, and comparisons leave it out. A verdict already there is kept, since a choice
may have been made on it, and only a null is asked again.

    python scripts/factor/judge_eval.py results/factor/agent/s1-heldout.jsonl [...]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs", "factor_idea"))

from factor_idea.judge import judge  # noqa: E402


async def main_async(args) -> None:
    with open(args.ideas) as f:
        ideas = {r["id"]: r for r in map(json.loads, f)}
    gate = asyncio.Semaphore(args.concurrency)

    async def ask(row):
        async with gate:
            try:
                row["faithful"] = await judge(ideas[row["idea"]]["en"], row["factor"])
            except Exception as exc:  # one lost verdict shouldn't cost the rest
                row["faithful"] = None
                print(f"no verdict on {row['idea']}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)

    for path in args.files:
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        passed = [r for r in rows if r.get("matches_idea") and r.get("factor")]
        await asyncio.gather(*(ask(r) for r in passed if r.get("faithful") is None))
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        verdicts = [r["faithful"] for r in passed if r["faithful"] is not None]
        share = f"{mean(verdicts):.1%}" if verdicts else "n/a"
        print(f"{path}: {len(passed)} passing of {len(rows)}, judged faithful {share}, no verdict {len(passed) - len(verdicts)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ideas", default="task/ideas_v2.jsonl")
    ap.add_argument("--concurrency", type=int, default=16)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
