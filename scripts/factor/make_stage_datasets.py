#!/usr/bin/env python3
"""The strategy trajectories, turned into the datasets of SFT2 and offline RL.

Each idea and wording was worked once per strategy, which makes a group.

- SFT2 keeps each group's best: the highest incremental IC on the research window, read from the
  result the teacher itself saw, among trajectories that pass the gates and hand in what they tested.
- Offline RL keeps them all, each weighted by its reward on 2025 minus its group's mean, so good
  answers are pushed up and bad ones pushed down. A trajectory that fails the gates earns 0. There
  is one dataset per reward arm, written to <dir>/s3-<arm>. In arm C, a trajectory the judge gives
  no verdict on is left out of its group.
- S3 trains on the rows with a positive weight, written to <dir>/s3p-<arm>, since the full set broke
  the model's syntax on validation ideas (see results/factor/week2.md).

Gates and rewards come from the verifier as it stands now, not from when a trajectory was written.

    python scripts/factor/make_stage_datasets.py results/factor/teacher/strategies.jsonl --sft2 <dir> --offline <dir> --arms ic,gates,judged
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import re
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs", "factor_idea"))

from factor_idea import workers  # noqa: E402
from factor_idea.judge import judge  # noqa: E402
from make_sft_dataset import keep, off_script, wire  # noqa: E402
from teacher_traces import TOOL  # noqa: E402

MAX_TESTS = 5

_IC = re.compile(r"incremental IC after removing known styles: ([+-][0-9.]+)")


def research_ic(row: dict) -> float | None:
    """The incremental IC test_factor reported for the formula this trajectory handed in."""
    history = row["messages"]
    for i, m in enumerate(history):
        if m["role"] == "assistant" and row["factor"] and row["factor"].strip() in [c.strip() for c in m["calls"]]:
            found = _IC.search(history[i + 1]["content"]) if i + 1 < len(history) else None
            if found:
                return float(found.group(1))
    return None


def write(path: str, rows: list[dict]) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "train.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--ideas", default="task/ideas_v2.jsonl")
    ap.add_argument("--arms", default="ic,gates,judged", help="the rewards offline RL weights by")
    ap.add_argument("--sft2", default="")
    ap.add_argument("--offline", default="")
    args = ap.parse_args()
    with open(args.ideas) as f:
        ideas = {r["id"]: r for r in map(json.loads, f)}
    split_of = {k: v["split"] for k, v in ideas.items()}
    groups = collections.defaultdict(list)
    for path in args.inputs:
        with open(path) as f:
            for row in map(json.loads, f):
                if split_of.get(row["idea"]) == "train" and not off_script(row):
                    groups[(row["idea"], row["variant"])].append(row)
    tools = json.dumps([TOOL])

    def to_row(r: dict, weight: float = 1.0) -> dict:
        return {"messages": wire(r["messages"]), "tools": tools, "weight": weight, "idea": r["idea"], "family": r["family"], "strategy": r["strategy"]}

    rows = [r for group in groups.values() for r in group]

    async def verdicts(scores: list) -> list[bool | None]:
        gate = asyncio.Semaphore(16)

        async def ask(row, score):
            if not score.matches_idea:
                return False
            async with gate:
                try:
                    return await judge(ideas[row["idea"]]["en"], row["factor"])
                except Exception as exc:  # one lost verdict shouldn't cost the rest
                    print(f"no verdict on {row['idea']}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                    return None

        return await asyncio.gather(*(ask(r, s) for r, s in zip(rows, scores)))

    async def scored() -> list:
        return await asyncio.gather(*(
            workers.score(r["factor"] or "", ideas[r["idea"]]["key"], ideas[r["idea"]]["direction"], tuple(ideas[r["idea"]].get("also", ())))
            for r in rows
        ))

    scores = asyncio.run(scored())
    was = sum(bool(r["matches_idea"]) for r in rows)
    for row, score in zip(rows, scores):
        row["matches_idea"] = score.matches_idea
    print(f"{len(rows)} trajectories in {len(groups)} groups; passing {was} when written, {sum(s.matches_idea for s in scores)} now")

    if args.sft2:
        best, chosen = [], collections.Counter()
        for group in groups.values():
            usable = [(research_ic(r), r) for r in group if not keep(r, split_of, "train")]
            usable = [(ic, r) for ic, r in usable if ic is not None]
            if usable:
                ic, r = max(usable, key=lambda x: x[0])
                best.append(to_row(r))
                chosen[r["strategy"]] += 1
        write(args.sft2, best)
        print(f"SFT2: {len(best)} of {len(groups)} groups; strategy chosen: {dict(chosen)}")

    if args.offline:
        arms = args.arms.split(",")
        faithful = asyncio.run(verdicts(scores)) if "judged" in arms else [False] * len(rows)
        if "judged" in arms:
            print(f"judged faithful: {sum(v is True for v in faithful)} of {sum(s.matches_idea for s in scores)} passing, no verdict {sum(v is None for v in faithful)}")
        for arm in arms:
            earned = collections.defaultdict(list)
            for row, score, verdict in zip(rows, scores, faithful):
                if arm == "judged" and verdict is None:
                    continue
                reward = score.reward("ic" if arm == "judged" else arm)
                if row["tests"] > MAX_TESTS or (arm == "judged" and not verdict):
                    reward = 0.0
                earned[(row["idea"], row["variant"])].append((row, reward))
            weighted = []
            for group in earned.values():
                base = mean(reward for _, reward in group)
                weighted += [to_row(r, reward - base) for r, reward in group if reward != base]
            write(os.path.join(args.offline, f"s3-{arm}"), weighted)
            write(os.path.join(args.offline, f"s3p-{arm}"), [r for r in weighted if r["weight"] > 0])
            print(f"offline {arm}: {len(weighted)} trajectories with a non-zero weight on {len({r['idea'] for r in weighted})} ideas; mean |weight| {mean(abs(r['weight']) for r in weighted):.3f}")


if __name__ == "__main__":
    main()
