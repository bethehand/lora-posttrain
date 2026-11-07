#!/usr/bin/env python3
"""The audit of prereg 03: passing held-out formulas of S2 and of S4 arms A, B and C, read once and blind.

    python scripts/factor/audit.py draw    # the sample to read and, apart from it, the key
    python scripts/factor/audit.py score   # once verdicts.jsonl is written: H7, H9 and how far the judge agrees

Each model gives up to 60 formulas, evenly across ideas. A pair of idea and formula that several models handed
in is read once, and its verdict counts for each of them. The reader sees only the idea and the formula, as the
judge does, and writes one {"n", "faithful", "why"} per item before the key is opened. The sealed period is never
read here.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import sys
from fractions import Fraction

import numpy as np

AGENT = "results/factor/agent"
OUT = "results/factor/audit"
MODELS = {
    "S2": ["s2-heldout"],
    "S4 arm A": ["factor-s4-gates-seed1-heldout", "factor-s4-gates-seed2-heldout"],
    "S4 arm B": ["factor-s4-ic-seed1-heldout", "factor-s4-ic-seed2-heldout"],
    "S4 arm C": ["factor-s4-judged-seed1-heldout", "factor-s4-judged-seed2-heldout"],
}
PER_MODEL = 60
TOO_FEW = 30
SEED = 3
BOOTSTRAP = 10_000


# Kept pure so it can be tested.
def sample_evenly(rows: list[dict], limit: int, rng: random.Random) -> list[dict]:
    """Up to limit rows, one from each idea in turn."""
    left = collections.defaultdict(list)
    for row in rows:
        left[row["idea"]].append(row)
    for group in left.values():
        rng.shuffle(group)
    order = sorted(left)
    rng.shuffle(order)
    taken = []
    while len(taken) < limit and any(left.values()):
        for idea in order:
            if left[idea] and len(taken) < limit:
                taken.append(left[idea].pop())
    return taken


def read_once(drawn: dict[str, list[dict]], rng: random.Random) -> list[dict]:
    """The items to read, shuffled, with a repeated pair of idea and formula merged. `from` says who handed it in."""
    merged = {}
    for model, rows in drawn.items():
        for row in rows:
            pair = (row["idea"], re.sub(r"\s+", "", row["factor"]))
            item = merged.setdefault(pair, {"idea": row["idea"], "factor": row["factor"], "from": []})
            item["from"].append({"model": model, "file": row["file"], "judge": row.get("faithful")})
    items = list(merged.values())
    rng.shuffle(items)
    return items


def difference(a: list[bool], b: list[bool], seed: int = 0) -> tuple[float, float, float]:
    """Share in a minus share in b, with a 95% CI from a bootstrap over formulas."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diffs = x[rng.integers(0, x.size, size=(BOOTSTRAP, x.size))].mean(axis=1) - y[rng.integers(0, y.size, size=(BOOTSTRAP, y.size))].mean(axis=1)
    return float(x.mean() - y.mean()), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def same_ideas(a: list[dict], b: list[dict], seed: int = 0) -> tuple[float, float, float, int]:
    """The same difference taken per idea over the ideas both have, with whole ideas resampled."""
    per = [collections.defaultdict(list), collections.defaultdict(list)]
    for side, rows in zip(per, (a, b)):
        for row in rows:
            side[row["idea"]].append(row["faithful"])
    shared = sorted(set(per[0]) & set(per[1]))
    values = np.asarray([np.mean(per[0][i]) - np.mean(per[1][i]) for i in shared], dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(BOOTSTRAP, values.size))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), len(shared)


def by_model(key: list[dict], verdicts: dict[int, bool]) -> dict[str, list[dict]]:
    out = collections.defaultdict(list)
    for item in key:
        for source in item["from"]:
            out[source["model"]].append({"idea": item["idea"], "faithful": verdicts[item["n"]], "judge": source["judge"], "n": item["n"]})
    return out


def within_ten_points(later: list[bool], earlier: list[bool]) -> bool:
    """In fractions: 42/60 minus 44/55 is exactly -1/10, and floats put it just under."""
    return Fraction(sum(later), len(later)) - Fraction(sum(earlier), len(earlier)) >= Fraction(-1, 10)


def settle(models: dict[str, list[dict]]) -> None:
    """H7 and H9 by the rules of prereg 03."""
    faithful = {m: [r["faithful"] for r in models[m]] for m in MODELS}
    if min(len(faithful["S2"]), len(faithful["S4 arm B"])) < TOO_FEW:
        print("H7: undecided, a model has fewer than 30")
    else:
        diff, low, high = difference(faithful["S4 arm B"], faithful["S2"])
        verdict = "passes" if within_ten_points(faithful["S4 arm B"], faithful["S2"]) else "fails"
        print(f"H7: S4 arm B minus S2 {diff:+.1%} (95% CI {low:+.1%} to {high:+.1%}, beside the rule); {verdict}, the rule is no more than 10 points below")
    if min(len(faithful["S4 arm C"]), len(faithful["S4 arm B"])) < TOO_FEW:
        print("H9: undecided, a model has fewer than 30")
    else:
        diff, low, high = difference(faithful["S4 arm C"], faithful["S4 arm B"])
        print(f"H9: S4 arm C minus arm B {diff:+.1%} (95% CI {low:+.1%} to {high:+.1%}); {'passes' if low > 0 else 'does not pass'}, the rule is a lower bound above 0")


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def draw(ideas: dict[str, dict]) -> None:
    if os.path.exists(f"{OUT}/sample.jsonl"):
        sys.exit("the sample is already drawn, and it is read once")
    rng = random.Random(SEED)
    drawn = {}
    for model, names in MODELS.items():
        rows = [{**row, "file": name} for name in names for row in read_jsonl(f"{AGENT}/{name}.jsonl") if row.get("matches_idea") and row.get("factor")]
        drawn[model] = sample_evenly(rows, PER_MODEL, rng)
        print(f"{model}: {len(rows)} passing, {len(drawn[model])} drawn, {len({r['idea'] for r in drawn[model]})} ideas")
    items = read_once(drawn, rng)
    os.makedirs(OUT, exist_ok=True)
    write_jsonl(f"{OUT}/sample.jsonl", [{"n": n, "idea": ideas[i["idea"]]["en"], "zh": ideas[i["idea"]]["zh"], "formula": i["factor"]} for n, i in enumerate(items, start=1)])
    write_jsonl(f"{OUT}/key.jsonl", [{"n": n, "idea": i["idea"], "from": i["from"]} for n, i in enumerate(items, start=1)])
    print(f"{len(items)} items to read, from {sum(map(len, drawn.values()))} formulas")


def score(ideas: dict[str, dict]) -> None:
    sample, key, answers = (read_jsonl(f"{OUT}/{name}.jsonl") for name in ("sample", "key", "verdicts"))
    verdicts = {r["n"]: bool(r["faithful"]) for r in answers}
    borderline = {r["n"] for r in answers if r.get("borderline")}
    missing = [r["n"] for r in sample if r["n"] not in verdicts]
    if missing:
        sys.exit(f"no verdict on items {missing[:20]}")
    models = by_model(key, verdicts)
    print(f"{len(sample)} items read, {sum(map(len, models.values()))} formulas")
    for model in MODELS:
        rows = models[model]
        judged = [r for r in rows if r["judge"] is not None]
        agree = sum(r["judge"] == r["faithful"] for r in judged)
        print(
            f"  {model}: {sum(r['faithful'] for r in rows)}/{len(rows)} faithful = {np.mean([r['faithful'] for r in rows]):.1%} over {len({r['idea'] for r in rows})} ideas; "
            f"the judge says {sum(bool(r['judge']) for r in judged)}/{len(judged)} and agrees on {agree}/{len(judged)}"
        )
    settle(models)
    print(f"The same with the {len(borderline)} borderline items all ruled not faithful:")
    settle(by_model(key, {n: v and n not in borderline for n, v in verdicts.items()}))
    print("The same with them all ruled faithful:")
    settle(by_model(key, {n: v or n in borderline for n, v in verdicts.items()}))
    print("Exploratory, per idea over the ideas both models have:")
    for later, earlier in (("S4 arm B", "S2"), ("S4 arm C", "S4 arm B"), ("S4 arm A", "S2"), ("S4 arm B", "S4 arm A")):
        diff, low, high, count = same_ideas(models[later], models[earlier])
        print(f"  {later} minus {earlier}: {diff:+.1%} (95% CI {low:+.1%} to {high:+.1%}) over {count} ideas")
    print("Faithful share by family:")
    for family in sorted({ideas[r["idea"]]["family"] for rows in models.values() for r in rows}):
        cells = []
        for model in MODELS:
            rows = [r for r in models[model] if ideas[r["idea"]]["family"] == family]
            cells.append(f"{model} {sum(r['faithful'] for r in rows)}/{len(rows)}")
        print(f"  {family:18s} " + ", ".join(cells))
    print("Items where the reader and the judge differ, reader's verdict first:")
    differ = {(r["n"], r["faithful"], r["judge"]) for rows in models.values() for r in rows if r["judge"] is not None and r["judge"] != r["faithful"]}
    print("  " + ", ".join(f"{n}:{'faithful' if mine else 'not'}" for n, mine, _ in sorted(differ)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["draw", "score"])
    ap.add_argument("--ideas", default="task/ideas_v2.jsonl")
    args = ap.parse_args()
    ideas = {r["id"]: r for r in read_jsonl(args.ideas)}
    (draw if args.step == "draw" else score)(ideas)


if __name__ == "__main__":
    main()
