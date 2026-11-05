#!/usr/bin/env python3
"""Does testing candidates on 2022-2024 help on 2025, which the model never sees?

Two comparisons, both resampling whole ideas because answers to one idea are related:
1. with the tool against one-shot, per idea, on the same model;
2. inside each episode with the tool: the first formula it tested against the one it handed in.

Reads dev.npz only; the sealed test stays closed.

    python scripts/factor/compare_arms.py --tools results/factor/agent/deepseek-tools.jsonl \
        --oneshot results/factor/agent/deepseek-oneshot.jsonl --out results/factor/agent/deepseek-compare.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from statistics import mean

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs", "factor_idea"))

from factor_idea import dsl, sandbox as sb  # noqa: E402
from factor_idea.evaluate import evaluate  # noqa: E402
from factor_idea.panel import load_dev  # noqa: E402

BOOTSTRAP = 10_000
_INC = re.compile(r"incremental IC after removing known styles: ([+-][0-9.]+)")


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def per_idea(rows, field):
    out = {}
    for r in rows:
        value = field(r)
        if value is not None:
            out.setdefault(r["idea"], []).append(value)
    return {k: mean(v) for k, v in out.items()}


def bootstrap_mean(values, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    picks = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    means = values[picks].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired(a, b):
    ideas = sorted(set(a) & set(b))
    return bootstrap_mean([a[i] - b[i] for i in ideas]) + (len(ideas),)


def tested_expressions(row):
    """Every expression the episode sent to the tool, with what the tool said about it."""
    out = []
    steps = row["steps"]
    for i, step in enumerate(steps):
        if step["role"] != "assistant" or not step.get("calls"):
            continue
        results = [s["content"] for s in steps[i + 1 : i + 1 + len(step["calls"])] if s["role"] == "tool"]
        for call, result in zip(step["calls"], results):
            try:
                expression = json.loads(call)["expression"]
            except (TypeError, ValueError, KeyError):
                continue
            match = _INC.search(result)
            out.append(dict(expression=expression, in_sample=float(match.group(1)) if match else None))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", required=True)
    ap.add_argument("--oneshot", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tools, oneshot = load(args.tools), load(args.oneshot)
    report = {}

    report["tools_minus_oneshot"] = {
        "reward": dict(zip(("mean", "low", "high", "ideas"), paired(per_idea(tools, lambda r: r["reward_ic"] or 0.0), per_idea(oneshot, lambda r: r["reward_ic"] or 0.0)))),
        "passed": dict(zip(("mean", "low", "high", "ideas"), paired(per_idea(tools, lambda r: float(r["matches_idea"])), per_idea(oneshot, lambda r: float(r["matches_idea"]))))),
        "ic_among_passed": dict(zip(("mean", "low", "high", "ideas"), paired(per_idea(tools, lambda r: r["ic"] if r["matches_idea"] else None), per_idea(oneshot, lambda r: r["ic"] if r["matches_idea"] else None)))),
    }

    box = sb.Sandbox(load_dev())
    panel = box.dev
    cache: dict[str, float] = {}

    def ic_2025(expression):
        if expression not in cache:
            with np.errstate(all="ignore"):
                values = evaluate(dsl.parse(expression), panel.fields)
            cache[expression] = box.incremental_ic(values, panel, *sb.RL_IC).incremental
        return cache[expression]

    rows = []
    for r in tools:
        if not r["matches_idea"]:
            continue
        tests = [t for t in tested_expressions(r) if t["in_sample"] is not None]
        if not tests:
            continue
        first = tests[0]
        final = next((t for t in reversed(tests) if t["expression"].strip() == r["factor"].strip()), None)
        rows.append(dict(
            idea=r["idea"],
            tests=len(tests),
            first_in_sample=first["in_sample"],
            final_in_sample=final["in_sample"] if final else None,
            best_in_sample=max(t["in_sample"] for t in tests),
            first_2025=ic_2025(first["expression"]),
            final_2025=r["ic"],
            final_was_tested=final is not None,
        ))
    report["episodes_compared"] = len(rows)
    report["final_was_tested_share"] = mean(x["final_was_tested"] for x in rows)
    report["final_minus_first_2025"] = dict(zip(("mean", "low", "high", "ideas"), paired(per_idea(rows, lambda x: x["final_2025"]), per_idea(rows, lambda x: x["first_2025"]))))
    tested = [x for x in rows if x["final_was_tested"]]
    report["final_minus_first_in_sample"] = dict(zip(("mean", "low", "high", "ideas"), paired(per_idea(tested, lambda x: x["final_in_sample"]), per_idea(tested, lambda x: x["first_in_sample"]))))
    report["same_episodes_2025"] = dict(zip(("mean", "low", "high", "ideas"), paired(per_idea(tested, lambda x: x["final_2025"]), per_idea(tested, lambda x: x["first_2025"]))))
    report["means"] = dict(
        first_in_sample=mean(x["first_in_sample"] for x in tested),
        final_in_sample=mean(x["final_in_sample"] for x in tested),
        first_2025=mean(x["first_2025"] for x in tested),
        final_2025=mean(x["final_2025"] for x in tested),
    )
    report["final_better_than_first_2025_share"] = mean(x["final_2025"] > x["first_2025"] for x in rows)
    report["rows"] = rows
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
