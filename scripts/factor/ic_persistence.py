#!/usr/bin/env python3
"""Does a factor's incremental IC persist from one year to the next?

If it doesn't, an RL reward built on one year's IC only chases noise, however clean the
sandbox is. The question that matters for RL is the within-idea one: among formulas for the
same idea, does the one that scores higher in one year also score higher in another? Reads
dev.npz only, so 2026 stays sealed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs", "factor_idea"))

from factor_idea import dsl, sandbox as sb  # noqa: E402
from factor_idea.evaluate import evaluate  # noqa: E402
from factor_idea.panel import load_dev  # noqa: E402

YEARS = {"2023": ("2023-01-01", "2023-12-31"), "2024": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-29")}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def within_group(values, groups):
    values = np.asarray(values, float)
    out = values.copy()
    for g in set(groups):
        idx = np.array([x == g for x in groups])
        out[idx] -= np.nanmean(values[idx])
    return out


def bootstrap_by_group(a, b, groups, n=2000, seed=0):
    """A 95% interval for the correlation, resampling whole ideas because formulas within an idea are related."""
    rng = np.random.default_rng(seed)
    names = sorted(set(groups))
    index = {g: np.flatnonzero(np.array(groups) == g) for g in names}
    stats = []
    for _ in range(n):
        picked = rng.choice(len(names), len(names))
        rows = np.concatenate([index[names[i]] for i in picked])
        stats.append(spearman(np.asarray(a)[rows], np.asarray(b)[rows]))
    return [float(np.nanquantile(stats, 0.025)), float(np.nanquantile(stats, 0.975))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", nargs="+", default=["results/factor/baselines/deepseek-flash.jsonl"])
    ap.add_argument("--out", default="results/factor/ic_persistence.json")
    args = ap.parse_args()

    box = sb.Sandbox(load_dev())
    panel = box.dev
    formulas: dict[tuple[str, dsl.Node], str] = {}
    for path in args.answers:
        with open(path) as f:
            for r in map(json.loads, f):
                if r["matches_idea"]:
                    formulas.setdefault((r["idea"], dsl.parse(r["factor"])), r["factor"])
    rows = []
    for (idea, node), text in formulas.items():
        with np.errstate(all="ignore"):
            values = evaluate(node, panel.fields)
        row = dict(idea=idea, factor=text)
        for year, (first, last) in YEARS.items():
            ic = box.incremental_ic(values, panel, first, last)
            row[year] = ic.incremental
            row[f"{year}_raw"] = ic.raw
        rows.append(row)
    groups = [r["idea"] for r in rows]
    print(f"{len(rows)} distinct formulas across {len(set(groups))} ideas")

    report = dict(formulas=len(rows), ideas=len(set(groups)), pairs={})
    for a, b in (("2023", "2024"), ("2024", "2025"), ("2023", "2025")):
        xa, xb = [r[a] for r in rows], [r[b] for r in rows]
        ra, rb = [r[f"{a}_raw"] for r in rows], [r[f"{b}_raw"] for r in rows]
        report["pairs"][f"{a}->{b}"] = dict(
            all_formulas=spearman(xa, xb),
            all_formulas_ci=bootstrap_by_group(xa, xb, groups),
            within_idea=spearman(within_group(xa, groups), within_group(xb, groups)),
            within_idea_ci=bootstrap_by_group(within_group(xa, groups), within_group(xb, groups), groups),
            raw_all=spearman(ra, rb),
            raw_within_idea=spearman(within_group(ra, groups), within_group(rb, groups)),
        )
        print(f"{a}->{b}: {json.dumps(report['pairs'][f'{a}->{b}'])}")

    print("\nper idea: mean incremental IC by year, and the spread between its formulas in 2025")
    per_idea = {}
    for idea in sorted(set(groups)):
        items = [r for r in rows if r["idea"] == idea]
        per_idea[idea] = {y: float(np.mean([r[y] for r in items])) for y in YEARS}
        per_idea[idea]["n"] = len(items)
        per_idea[idea]["spread_2025"] = float(np.ptp([r["2025"] for r in items]))
        print(f"  {idea:20s} n={len(items):2d} 2023={per_idea[idea]['2023']:+.4f} 2024={per_idea[idea]['2024']:+.4f} 2025={per_idea[idea]['2025']:+.4f} spread_2025={per_idea[idea]['spread_2025']:.4f}")
    ic25 = np.array([r["2025"] for r in rows])
    print("\n2025 incremental IC across formulas: quantiles 5/25/50/75/95", np.round(np.quantile(ic25, [.05, .25, .5, .75, .95]), 4), "share above 0:", round(float((ic25 > 0).mean()), 3))
    report["per_idea"] = per_idea
    report["rows"] = rows
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
