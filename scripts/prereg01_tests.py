"""The pre-registered tests of prereg/01-wiki-search.md, from the summarized eval results.

Reads results/evals/<base>/<set>/questions.jsonl and the same for the trained model (written by
summarize_eval.py, navigation.jsonl by navigation_metrics.py, <set>-flash by rejudge.py), prints
a table and writes results/prereg01_tests.md. Differences are trained minus base over the
questions both runs scored, with 95% bootstrap CIs over questions (10,000 resamples).

    python3 scripts/prereg01_tests.py [base name] [trained name]   # prereg01-base prereg01-rl35
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

BOOTSTRAP = 10_000
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/evals"


def load(name: str, file: str = "questions.jsonl", key: str = "mean_reward") -> dict[int, float] | None:
    path = RESULTS / name / file
    return {r["idx"]: r[key] for r in map(json.loads, path.open())} if path.exists() else None


def resample_mean(values: list[float], rng: random.Random) -> float:
    return sum(rng.choice(values) for _ in range(len(values))) / len(values)


def ci(boot: list[float]) -> tuple[float, float]:
    boot.sort()
    return boot[int(0.025 * BOOTSTRAP)], boot[int(0.975 * BOOTSTRAP)]


def diffs(a: dict[int, float], b: dict[int, float]) -> list[float]:
    return [b[i] - a[i] for i in sorted(a.keys() & b.keys())]


def paired(a: dict[int, float], b: dict[int, float]) -> tuple[float, float, float]:
    d = diffs(a, b)
    rng = random.Random(0)
    return (sum(d) / len(d), *ci([resample_mean(d, rng) for _ in range(BOOTSTRAP)]))


def main() -> None:
    base, trained = (sys.argv[1:3] + ["prereg01-base", "prereg01-rl35"][len(sys.argv[1:3]):])[:2]
    rows = []  # (id, quantity, estimate, ci, criterion, verdict)

    def accuracy_row(label: str, run: str, subset=None) -> None:
        scores = load(f"{run}/heldout")
        if subset:
            scores = {i: v for i, v in scores.items() if subset(i)}
        rows.append(("", label, sum(scores.values()) / len(scores), None, "", ""))

    ho_a, ho_b = load(f"{base}/heldout"), load(f"{trained}/heldout")
    gain, lo, hi = paired(ho_a, ho_b)
    rows.append(("H1", "heldout gain", gain, (lo, hi), "CI lower bound > 0", lo > 0))
    rows.append(("H1b", "heldout gain", gain, (lo, hi), "gain >= 0.20", gain >= 0.20))

    slide = [v for i, v in ho_a.items() if i < 100]
    slide_acc = sum(slide) / len(slide)
    rows.append(("H2", "base on slide100, this session", slide_acc, None, "0.45-0.65; passed before the freeze at 0.568",
                 0.45 <= slide_acc <= 0.65))

    ts_a, ts_b = load(f"{base}/train_sample"), load(f"{trained}/train_sample")
    d_ts, d_ho = diffs(ts_a, ts_b), diffs(ho_a, ho_b)
    rng = random.Random(0)
    gap_boot = [resample_mean(d_ts, rng) - resample_mean(d_ho, rng) for _ in range(BOOTSTRAP)]
    gap = sum(d_ts) / len(d_ts) - sum(d_ho) / len(d_ho)
    rows.append(("H3", "train_sample gain minus heldout gain", gap, ci(gap_boot), "<= 0.10", gap <= 0.10))

    nav_a, nav_b = (load(f"{run}/heldout", "navigation.jsonl", "section_reads") for run in (base, trained))
    wrong_a, wrong_b = (load(f"{run}/heldout", "navigation.jsonl", "wrong_section") for run in (base, trained))
    if nav_a and nav_b:
        reads_b = sum(nav_b[i] for i in nav_a.keys() & nav_b.keys()) / len(nav_a.keys() & nav_b.keys())
        r_gain, r_lo, r_hi = paired(nav_a, nav_b)
        w_gain, w_lo, w_hi = paired(wrong_a, wrong_b)
        rows.append(("H4", "trained section reads per episode", reads_b, None, "> 1.5", reads_b > 1.5))
        rows.append(("H4", "section reads gain", r_gain, (r_lo, r_hi), "CI lower bound > 0", r_lo > 0))
        rows.append(("H4", "wrong-section failure share change", w_gain, (w_lo, w_hi), "CI upper bound < 0", w_hi < 0))

    for label, set_name in (("train_mixed gain (trained questions)", "train_mixed"),
                            ("heldout-notools gain", "heldout-notools"),
                            ("heldout gain judged by deepseek-flash", "heldout-flash")):
        a, b = load(f"{base}/{set_name}"), load(f"{trained}/{set_name}")
        if a and b:
            g, g_lo, g_hi = paired(a, b)
            rows.append(("expl.", label, g, (g_lo, g_hi), "", ""))
    accuracy_row("base heldout accuracy", base)
    accuracy_row("trained heldout accuracy", trained)

    lines = ["| ID | Quantity | Estimate | 95% CI | Criterion | Verdict |", "|---|---|---|---|---|---|"]
    for hid, label, est, interval, criterion, verdict in rows:
        interval_text = f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else ""
        verdict_text = "" if verdict == "" else ("pass" if verdict else "fail")
        est_text = f"{est:+.3f}" if interval else f"{est:.3f}"  # differences carry a sign and a CI
        lines.append(f"| {hid} | {label} | {est_text} | {interval_text} | {criterion} | {verdict_text} |")
    table = "\n".join(lines)
    print(table)
    header = f"# Prereg 01 tests\n\nBase `{base}`, trained `{trained}`; `python3 scripts/prereg01_tests.py`.\n\n"
    (ROOT / "results/prereg01_tests.md").write_text(header + table + "\n")


if __name__ == "__main__":
    main()
