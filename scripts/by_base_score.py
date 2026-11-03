"""Accuracy grouped by the base model's score on each held-out question (exploratory, REPORT.md
Section 4): which questions a gain comes from. On slide100, the published 100, it adds the
published base and adapter rollouts re-graded by deepseek-v4-pro (results/judge_agreement), one
rollout per question, so the published gain can be split the same way.

A group's contribution is its questions' summed change over all questions in the table, so the
contributions add up to the gain.

Grouping and measuring with the same base rollouts invites regression to the mean, so a last
table groups the held-out questions by an earlier, independent eval of the base model and sets
each group's gain against the gain on the trained questions, with a 95% bootstrap CI of the
difference. Its "1 to 3 of 4" group is the untrained counterpart of the trained questions, which
were picked for mixed outcomes too.

    python3 scripts/by_base_score.py [base] [trained] [earlier base eval] [trained set]
    # defaults: prereg01-base prereg01-rl35 qwen3-4b-base train_mixed
"""

from __future__ import annotations

import json
import random
import sys

from prereg01_tests import BOOTSTRAP, ROOT, diffs, load, resample_mean

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


def against_trained(early_name: str, early: dict[int, float], base: dict[int, float], trained: dict[int, float],
                    set_name: str, set_base: dict[int, float], set_trained: dict[int, float]) -> list[str]:
    """Held-out questions grouped by an earlier base eval, each group's gain set against the gain
    on the trained questions."""
    d_set = diffs(set_base, set_trained)
    set_gain = sum(d_set) / len(d_set)
    lines = [f"## heldout grouped by the earlier base eval `{early_name}`", "",
             f"| Group | Questions | Base | Adapter | Gain | `{set_name}` gain minus this, 95% CI |",
             "|---|---|---|---|---|---|",
             f"| `{set_name}`, trained | {len(d_set)} | {sum(set_base.values()) / len(set_base):.3f} | "
             f"{sum(set_trained.values()) / len(set_trained):.3f} | {set_gain:+.3f} | |"]
    for label, member in GROUPS:
        ids = [i for i in base if i in early and member(early[i])]
        d = [trained[i] - base[i] for i in ids]
        gain = sum(d) / len(d)
        rng = random.Random(0)
        boot = sorted(resample_mean(d_set, rng) - resample_mean(d, rng) for _ in range(BOOTSTRAP))
        lo, hi = boot[int(0.025 * BOOTSTRAP)], boot[int(0.975 * BOOTSTRAP)]
        lines.append(f"| {label} | {len(ids)} | {sum(base[i] for i in ids) / len(ids):.3f} | "
                     f"{sum(trained[i] for i in ids) / len(ids):.3f} | {gain:+.3f} | {set_gain - gain:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    return lines + [""]


def main() -> None:
    defaults = ["prereg01-base", "prereg01-rl35", "qwen3-4b-base", "train_mixed"]
    base_name, trained_name, early_name, set_name = (sys.argv[1:5] + defaults[len(sys.argv[1:5]):])[:4]
    base, trained = load(f"{base_name}/heldout"), load(f"{trained_name}/heldout")
    lines = ["# Held-out accuracy by the base model's score on each question", "",
             f"Base `{base_name}` and adapter `{trained_name}`, judge deepseek-v4-pro, 4 rollouts per question. "
             "Written by `scripts/by_base_score.py`. Exploratory.", ""]
    lines += breakdown("heldout", base, {"base": base, "adapter": trained}, ("base", "adapter"))
    if base_name.startswith("prereg01"):  # the published rollouts are Qwen3-4B's
        lines += breakdown("slide100 with the published rollouts", {i: base[i] for i in range(100)},
                           {"base": base, "adapter": trained,
                            "published base": published("qwen--qwen3-4b-instruct-2507"),
                            "published adapter": published("willcb--qwen3-4b-wiki-search")},
                           ("published base", "published adapter"))
    lines += against_trained(early_name, load(f"{early_name}/heldout"), base, trained, set_name,
                             load(f"{base_name}/{set_name}"), load(f"{trained_name}/{set_name}"))
    out = ROOT / ("results/by_base_score.md" if base_name == "prereg01-base" else f"results/by_base_score-{trained_name}.md")
    out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
