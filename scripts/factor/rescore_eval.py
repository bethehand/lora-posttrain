#!/usr/bin/env python3
"""Score handed-in formulas again without running the model: after a change to the verifier, or on another period.

    python scripts/factor/rescore_eval.py results/factor/agent/v2-olmo-tools.jsonl                 # in place, current verifier
    python scripts/factor/rescore_eval.py <file> --period reward --out <new file>                  # must reproduce the file's IC
    python scripts/factor/rescore_eval.py <file> --period sealed --confirm <phrase> --out <new file>   # the final exam, logged

A model's formula can't depend on the scored period: test_factor stops at 2024 whichever is used.
Rehearse with --period reward first, since the sealed period opens only once.
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs", "factor_idea"))

from factor_idea import sandbox as sb  # noqa: E402
from factor_idea.panel import load_dev, load_sealed  # noqa: E402

MAX_TESTS = 5
_box = _panel = _range = None


def _score(job: tuple) -> sb.Score:
    text, key, direction, also = job
    return _box.score(text, key, direction, also=also, ic_range=_range, ic_panel=_panel)


def apply(row: dict, score: sb.Score) -> None:
    over = row.get("over_limit", row.get("tests", 0) > MAX_TESTS)
    row.update(
        parsed=score.parsed,
        computes=score.computes,
        matches_idea=score.matches_idea and not over,
        reason=score.reason if row.get("factor") else "no factor tag",
        ic=score.ic.incremental if score.ic else None,
        raw_ic=score.ic.raw if score.ic else None,
        reward=0.0 if over else score.reward("ic"),
        reward_ic=0.0 if over else score.reward("ic"),
    )


def main() -> None:
    global _box, _panel, _range
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--period", choices=["reward", "sealed"], default="reward")
    ap.add_argument("--confirm", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--ideas", default="task/ideas_v2.jsonl")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    with open(args.ideas) as f:
        ideas = {r["id"]: r for r in map(json.loads, f)}
    with open(args.file) as f:
        rows = [json.loads(line) for line in f]
    _box = sb.Sandbox(load_dev())
    _range = sb.RL_IC
    if args.period == "sealed":
        _panel, _range = load_sealed(args.confirm), sb.SEALED_IC
    usable = [r for r in rows if r.get("idea") in ideas]
    jobs = [(r.get("factor") or "", ideas[r["idea"]]["key"], ideas[r["idea"]]["direction"], tuple(ideas[r["idea"]].get("also", ()))) for r in usable]
    with mp.get_context("fork").Pool(args.jobs) as pool:
        scores = pool.map(_score, jobs, chunksize=4)
    before = sum(bool(r.get("matches_idea")) for r in usable)
    old_ic = [r.get("ic") for r in usable]
    changed = collections.Counter()
    for row, score in zip(usable, scores):
        was = bool(row.get("matches_idea"))
        apply(row, score)
        if was and not row["matches_idea"]:
            changed[row["reason"]] += 1
    out = args.out or args.file
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    passed = [r for r in usable if r["matches_idea"]]
    ics = [r["ic"] for r in passed if r["ic"] is not None]
    print(f"{args.file}: {len(rows)} rows, {len(rows) - len(usable)} without an idea; passed {before} -> {len(passed)}; newly failing {dict(changed)}; "
          f"mean IC among passes {mean(ics) if ics else float('nan'):+.5f}")
    if args.period == "reward":
        gaps = [abs(a - r["ic"]) for a, r in zip(old_ic, usable) if a is not None and r["ic"] is not None]
        if gaps:
            print(f"  largest change in a formula's IC against the file: {max(gaps):.2e}")


if __name__ == "__main__":
    main()
