#!/usr/bin/env python3
"""The three rules of prereg/03b-pilot.md, for each run against S4 arm B, and what is reported beside them.

Reads training rollouts under $ROOT/outputs and val answers under results/factor/agent. It reads no held-out idea
and no sealed file, and scores formulas again on 2025 only.

    python scripts/factor/pilot_compare.py factor-p1-s2-ic factor-p2-s2-rank
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explore_after_exam import AGENT, IDEAS, answer_research_ic, by_idea, ci, job_for, read_jsonl, run_rollouts, score_all, trials  # noqa: E402

from compare_stages import interval  # noqa: E402

COMPARED_WITH = ["factor-s4-ic-seed1", "factor-s4-ic-seed2"]
MOST_PASS_LOST = 0.05
HANDS_IN_BEST = {"factor-p2-s2-rank"}


# Kept pure so it can be tested.
def passes_rules(training_low: float, val_gap: float, pass_gap: float) -> bool:
    return training_low > 0 and val_gap > 0 and pass_gap >= -MOST_PASS_LOST - 1e-12


def mean_ic_per_idea(rows: list[dict]) -> dict[str, float]:
    """IC among passes that handed in an answer, one number per idea."""
    out = collections.defaultdict(list)
    for row in rows:
        if row["answered"] and row["passed"] and row["ic"] is not None:
            out[row["idea"]].append(row["ic"])
    return {idea: float(np.mean(values)) for idea, values in out.items()}


def gap(later: dict[str, float], earlier: dict[str, float]) -> tuple[float, float, float, int]:
    shared = sorted(set(later) & set(earlier))
    return (*interval([later[i] - earlier[i] for i in shared]), len(shared))


def val_rows(name: str) -> list[dict]:
    out = []
    for row in read_jsonl(f"{AGENT}/{name}.jsonl"):
        out.append({**row, "answered": bool(row.get("factor")), "passed": bool(row.get("matches_idea") and row.get("factor")), "tested": trials(row.get("steps"))})
    return out


def with_best_test(rows: list[dict]) -> list[dict]:
    """An episode with no answer scored on the formula it tested best, as this run's reward did in training."""
    slots = [i for i, r in enumerate(rows) if not r["answered"] and r["tested"]]
    scored = score_all([job_for(rows[i]["idea"], max(rows[i]["tested"], key=lambda t: t[1])[0]) for i in slots]) if slots else []
    out = [dict(r) for r in rows]
    for i, (passed, ic, _) in zip(slots, scored):
        out[i].update(passed=passed, ic=ic)
    return out


def pass_rate_per_idea(rows: list[dict]) -> dict[str, float]:
    return {idea: float(np.mean([r["passed"] for r in members])) for idea, members in by_idea(rows).items()}


def describe(rows: list[dict]) -> str:
    reasons = collections.Counter("wrong direction" if "against" in (r.get("reason") or "") else "other" for r in rows if r["answered"] and not r["passed"])
    judged = [r["faithful"] for r in rows if r["passed"] and r.get("faithful") is not None]
    research = [answer_research_ic(r.get("factor"), r["tested"]) for r in rows if r["passed"]]
    research = [x for x in research if x is not None]
    return (
        f"{len(rows)} episodes, answered {np.mean([r['answered'] for r in rows]):.1%}, passed {np.mean([r['passed'] for r in rows]):.1%}, wrong direction {reasons['wrong direction'] / len(rows):.1%}, "
        f"tests {np.mean([r.get('tests') or 0 for r in rows]):.2f}, faithful {f'{np.mean(judged):.1%} of {len(judged)}' if judged else 'not judged'}, research IC of answers {np.mean(research) if research else float('nan'):+.4f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    base_train = mean_ic_per_idea([r for name in COMPARED_WITH for r in run_rollouts(name, 26, 35)])
    base_val_rows = [r for name in COMPARED_WITH for r in val_rows(name)]
    base_val = mean_ic_per_idea(base_val_rows)
    base_pass = pass_rate_per_idea(base_val_rows)
    print(f"S4 arm B, val: {describe(base_val_rows)}")
    late = {}
    for name in args.runs:
        early = mean_ic_per_idea(run_rollouts(name, 1, 10))
        late[name] = mean_ic_per_idea(run_rollouts(name, 26, 35))
        training = gap(late[name], base_train)
        change = gap(late[name], early)
        rows = val_rows(name)
        own_gap = gap(mean_ic_per_idea(rows), base_val)
        scored = with_best_test(rows) if name in HANDS_IN_BEST else rows
        rate = pass_rate_per_idea(scored)
        pass_gap = float(np.mean([rate[i] - base_pass[i] for i in rate]))
        print(name)
        print(f"  val: {describe(rows)}")
        if name in HANDS_IN_BEST:
            print(f"  val, best test when no answer: passed {np.mean([r['passed'] for r in scored]):.1%}")
        print(f"  rule 1, training IC, steps 26-35, minus S4 arm B: {training[0]:+.4f} ({training[1]:+.4f} to {training[2]:+.4f}) over {training[3]} ideas")
        print(f"  rule 2, val IC minus S4 arm B: {own_gap[0]:+.4f} ({own_gap[1]:+.4f} to {own_gap[2]:+.4f}) over {own_gap[3]} ideas")
        print(f"  rule 3, val pass rate minus S4 arm B: {pass_gap:+.1%} per idea, {ci(rate[i] - base_pass[i] for i in rate)}")
        print(f"  beside them, training IC from steps 1-10 to 26-35: {change[0]:+.4f} ({change[1]:+.4f} to {change[2]:+.4f}) over {change[3]} ideas")
        print(f"  {'PASSES' if passes_rules(training[1], own_gap[0], pass_gap) else 'DOES NOT PASS'}")
    if len(args.runs) == 2:
        between = gap(late[args.runs[1]], late[args.runs[0]])
        print(f"{args.runs[1]} minus {args.runs[0]}, training IC, steps 26-35: {between[0]:+.4f} ({between[1]:+.4f} to {between[2]:+.4f}) over {between[3]} ideas")


if __name__ == "__main__":
    main()
