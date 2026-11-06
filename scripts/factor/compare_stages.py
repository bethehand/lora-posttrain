#!/usr/bin/env python3
"""One stage against another, per idea, on a set of ideas: the numbers prereg 03 judges by.

Answers to one idea are related, so whole ideas are resampled. An idea with no pass has no IC among
passes, so that comparison uses the ideas where both stages have one, and says how many that is.

    python scripts/factor/compare_stages.py results/factor/agent/s1-heldout.jsonl results/factor/agent/s0-heldout.jsonl --split heldout

Seeds of a stage are pooled by giving their files with commas: rollouts of one idea count together.
"""
from __future__ import annotations

import argparse
import collections
import json
from statistics import mean

import numpy as np

BOOTSTRAP = 10_000


def load(paths: str, split_of: dict[str, str], split: str, allow_missing: bool) -> dict[str, list[dict]]:
    """Rows of a set of ideas. A failed episode has no idea on it, and dropping it would shrink the denominator."""
    out, lost = collections.defaultdict(list), 0
    for path in paths.split(","):
        with open(path) as f:
            for row in map(json.loads, f):
                if row.get("idea") is None:
                    lost += 1
                elif split == "all" or split_of.get(row["idea"]) == split:
                    out[row["idea"]].append(row)
    sizes = collections.Counter(len(v) for v in out.values())
    if (lost or len(sizes) > 1) and not allow_missing:
        raise SystemExit(f"{paths}: {lost} failed episodes, rollouts per idea {dict(sizes)}. Rerun them, or pass --allow-missing and report it")
    return out


def measure(rows: list[dict]) -> dict[str, float | None]:
    passed = [r for r in rows if r.get("matches_idea")]
    ics = [r["ic"] for r in passed if r.get("ic") is not None]
    verdicts = [r["faithful"] for r in passed if r.get("faithful") is not None]
    return {
        "pass": len(passed) / len(rows),
        "reward_ic": mean(r.get("reward_ic") or 0.0 for r in rows),
        "ic_among_passes": mean(ics) if ics else None,
        "faithful_among_passes": mean(verdicts) if verdicts else None,
    }


def interval(values: list[float], seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(values, dtype=float)
    means = a[rng.integers(0, a.size, size=(BOOTSTRAP, a.size))].mean(axis=1)
    return float(a.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("later")
    ap.add_argument("earlier")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--ideas", default="task/ideas_v2.jsonl")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    with open(args.ideas) as f:
        ideas = {r["id"]: r for r in map(json.loads, f)}
    split_of = {k: v["split"] for k, v in ideas.items()}
    later, earlier = (load(path, split_of, args.split, args.allow_missing) for path in (args.later, args.earlier))
    shared = sorted(set(later) & set(earlier))
    print(f"{args.split}: {len(shared)} ideas; later {sum(map(len, later.values()))} episodes, earlier {sum(map(len, earlier.values()))}")
    a, b = {i: measure(later[i]) for i in shared}, {i: measure(earlier[i]) for i in shared}
    for name in ("pass", "reward_ic", "ic_among_passes", "faithful_among_passes"):
        both = [i for i in shared if a[i][name] is not None and b[i][name] is not None]
        if not both:
            print(f"  {name}: no idea has a value in both")
            continue
        diff, low, high = interval([a[i][name] - b[i][name] for i in both])
        print(f"  {name}: later {mean(a[i][name] for i in both):+.4f}, earlier {mean(b[i][name] for i in both):+.4f}, difference {diff:+.4f} (95% CI {low:+.4f} to {high:+.4f}) over {len(both)} ideas")
    print("  by family, pass rate later / earlier:")
    families = collections.defaultdict(list)
    for i in shared:
        families[ideas[i]["family"]].append(i)
    for family, members in sorted(families.items()):
        print(f"    {family:18s} {mean(a[i]['pass'] for i in members):.2f} / {mean(b[i]['pass'] for i in members):.2f} ({len(members)} ideas)")


if __name__ == "__main__":
    main()
