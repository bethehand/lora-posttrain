"""The pre-registered tests of prereg/02-olmo-wiki-search.md, from the summarized eval results.

Reads results/evals/<base>/<set>/questions.jsonl and the same for the trained model (written by
summarize_eval.py, call_failures.jsonl by call_failures.py, <set>-flash by rejudge.py), prints a
table and writes results/prereg02_tests.md. Sets are evaluated in the section environment unless
named -sectionid. Differences are trained minus base over the questions both runs scored, with
95% bootstrap CIs over questions (10,000 resamples).

    python3 scripts/prereg02_tests.py [base name] [trained name]   # prereg02-base prereg02-rl35
"""

from __future__ import annotations

import random
import sys

from prereg01_tests import BOOTSTRAP, ROOT, ci, diffs, load, paired, resample_mean

QWEN_RATIO = 0.066 / 0.371  # prereg 01: heldout gain over train_mixed gain


def main() -> None:
    base, trained = (sys.argv[1:3] + ["prereg02-base", "prereg02-rl35"][len(sys.argv[1:3]):])[:2]
    rows = []  # (id, quantity, estimate, ci, criterion, verdict)

    ho_a, ho_b = load(f"{base}/heldout"), load(f"{trained}/heldout")
    gain, lo, hi = paired(ho_a, ho_b)
    rows.append(("H1", "heldout gain", gain, (lo, hi), "CI lower bound > 0", lo > 0))

    base_acc = sum(ho_a.values()) / len(ho_a)
    rows.append(("H2", "base heldout accuracy, this session", base_acc, None,
                 "0.06-0.13; 0.094 before the freeze", 0.06 <= base_acc <= 0.13))

    d_ho = diffs(ho_a, ho_b)
    ts_a, ts_b = load(f"{base}/train_sample"), load(f"{trained}/train_sample")
    if ts_a and ts_b:
        d_ts = diffs(ts_a, ts_b)
        rng = random.Random(0)
        gap_boot = [resample_mean(d_ts, rng) - resample_mean(d_ho, rng) for _ in range(BOOTSTRAP)]
        gap = sum(d_ts) / len(d_ts) - sum(d_ho) / len(d_ho)
        rows.append(("H3", "train_sample gain minus heldout gain", gap, ci(gap_boot), "<= 0.10", gap <= 0.10))

    cf_a, cf_b = (load(f"{run}/heldout", "call_failures.jsonl", "call_failure") for run in (base, trained))
    if cf_a and cf_b:
        c_gain, c_lo, c_hi = paired(cf_a, cf_b)
        rows.append(("H4", "call-failure share change", c_gain, (c_lo, c_hi), "CI upper bound < 0", c_hi < 0))

    tm_a, tm_b = load(f"{base}/train_mixed_olmo"), load(f"{trained}/train_mixed_olmo")
    tm_lo = 0.0
    if tm_a and tm_b:
        tm_gain, tm_lo, tm_hi = paired(tm_a, tm_b)
        rows.append(("H5", "train_mixed_olmo gain (trained questions)", tm_gain, (tm_lo, tm_hi),
                     "CI lower bound > 0, else H5 is not tested", tm_lo > 0))
    if tm_lo > 0:
        d_tm = diffs(tm_a, tm_b)
        rng = random.Random(0)
        ratio_boot = [resample_mean(d_ho, rng) / resample_mean(d_tm, rng) for _ in range(BOOTSTRAP)]
        ratio = (sum(d_ho) / len(d_ho)) / (sum(d_tm) / len(d_tm))
        rows.append(("H5", f"heldout gain / trained-question gain (Qwen {QWEN_RATIO:.2f})", ratio,
                     ci(ratio_boot), ">= 0.5", ratio >= 0.5))

    si_a, si_b = load(f"{base}/heldout-sectionid"), load(f"{trained}/heldout-sectionid")
    if si_a and si_b:
        s_gain, s_lo, s_hi = paired(si_a, si_b)
        rows.append(("H6", "heldout gain, section_id environment", s_gain, (s_lo, s_hi), "CI lower bound > 0", s_lo > 0))

    for label, set_name in (("heldout-notools gain", "heldout-notools"),
                            ("heldout gain judged by deepseek-flash", "heldout-flash")):
        a, b = load(f"{base}/{set_name}"), load(f"{trained}/{set_name}")
        if a and b:
            g, g_lo, g_hi = paired(a, b)
            rows.append(("expl.", label, g, (g_lo, g_hi), "", ""))
    for label, run in (("base heldout accuracy", base), ("trained heldout accuracy", trained)):
        scores = load(f"{run}/heldout")
        rows.append(("", label, sum(scores.values()) / len(scores), None, "", ""))

    lines = ["| ID | Quantity | Estimate | 95% CI | Criterion | Verdict |", "|---|---|---|---|---|---|"]
    for hid, label, est, interval, criterion, verdict in rows:
        interval_text = f"[{interval[0]:+.3f}, {interval[1]:+.3f}]" if interval else ""
        verdict_text = "" if verdict == "" else ("pass" if verdict else "fail")
        est_text = f"{est:+.3f}" if interval else f"{est:.3f}"  # differences carry a sign and a CI
        lines.append(f"| {hid} | {label} | {est_text} | {interval_text} | {criterion} | {verdict_text} |")
    table = "\n".join(lines)
    print(table)
    header = f"# Prereg 02 tests\n\nBase `{base}`, trained `{trained}`; `python3 scripts/prereg02_tests.py`.\n\n"
    (ROOT / "results/prereg02_tests.md").write_text(header + table + "\n")


if __name__ == "__main__":
    main()
