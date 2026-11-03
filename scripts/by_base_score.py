"""Accuracy grouped by the base model's score on each held-out question (exploratory, REPORT.md
Section 4): which questions a gain comes from. On slide100, the published 100, it adds the
published base and adapter rollouts re-graded by deepseek-v4-pro (results/judge_agreement), one
rollout per question, so the published gain can be split the same way.

A group's contribution is its questions' summed change over all questions in the table, so the
contributions add up to the gain.

    python3 scripts/by_base_score.py [base name] [trained name]   # prereg01-base prereg01-rl35
"""

from __future__ import annotations

import json
import sys

from prereg01_tests import ROOT, load

PUBLISHED = ROOT / "results/judge_agreement/deepseek-v4-pro"
GROUPS = [("0 of 4", lambda s: s == 0), ("1 to 3 of 4", lambda s: 0 < s < 1), ("4 of 4", lambda s: s == 1)]


def published(model: str) -> dict[int, float]:
    return {r["example_id"]: r["ours"] for r in map(json.loads, (PUBLISHED / f"{model}.jsonl").open())}


def breakdown(title: str, base: dict[int, float], runs: dict[str, dict[int, float]], gain: tuple[str, str]) -> list[str]:
    """Rows of the questions in `base` grouped by their base score: each run's mean, and each
    group's contribution to runs[gain[1]] minus runs[gain[0]]."""
    names = list(runs)
    before, after = runs[gain[0]], runs[gain[1]]
    lines = [f"## {title}", "",
             "| Base on the question | Questions | " + " | ".join(names) + f" | Contribution to {gain[1]} minus {gain[0]} |",
             "|---" * (len(names) + 3) + "|"]
    for label, member in GROUPS + [("all", lambda s: True)]:
        ids = [i for i in base if member(base[i])]
        means = " | ".join(f"{sum(runs[n][i] for i in ids) / len(ids):.3f}" for n in names)
        contribution = sum(after[i] - before[i] for i in ids) / len(base)
        lines.append(f"| {label} | {len(ids)} | {means} | {contribution:+.3f} |")
    return lines + [""]


def main() -> None:
    base_name, trained_name = (sys.argv[1:3] + ["prereg01-base", "prereg01-rl35"][len(sys.argv[1:3]):])[:2]
    base, trained = load(f"{base_name}/heldout"), load(f"{trained_name}/heldout")
    lines = ["# Held-out accuracy by the base model's score on each question", "",
             f"Base `{base_name}` and adapter `{trained_name}`, judge deepseek-v4-pro, 4 rollouts per question. "
             "Written by `scripts/by_base_score.py`. Exploratory.", ""]
    lines += breakdown("heldout", base, {"base": base, "adapter": trained}, ("base", "adapter"))
    lines += breakdown("slide100 with the published rollouts", {i: base[i] for i in range(100)},
                       {"base": base, "adapter": trained,
                        "published base": published("qwen--qwen3-4b-instruct-2507"),
                        "published adapter": published("willcb--qwen3-4b-wiki-search")},
                       ("published base", "published adapter"))
    out = ROOT / "results/by_base_score.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
