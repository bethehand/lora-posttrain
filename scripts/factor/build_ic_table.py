#!/usr/bin/env python3
"""The percentile table of prereg/03b-pilot.md: every passing rollout on training ideas in the six S4 runs, on 2025.

    python scripts/factor/build_ic_table.py
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explore_after_exam import ARMS, IDEAS, REPO, run_rollouts  # noqa: E402

OUT = f"{REPO}/envs/factor_idea_rank/factor_idea_rank/ic_table.json"


def main() -> None:
    table = collections.defaultdict(list)
    for arm in ARMS.values():
        for seed in (1, 2):
            for row in run_rollouts(f"factor-s4-{arm}-seed{seed}", 1, 35):
                if row["passed"] and row["ic"] is not None:
                    table[row["idea"]].append(round(row["ic"], 6))
    train = {i for i, idea in IDEAS.items() if idea["split"] == "train"}
    if set(table) != train:
        sys.exit(f"the table has to cover the training ideas and nothing else: {sorted(set(table) ^ train)[:5]}")
    with open(OUT, "w") as f:
        json.dump({idea: sorted(values) for idea, values in sorted(table.items())}, f)
    with open(OUT, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    sizes = [len(v) for v in table.values()]
    print(f"{len(table)} ideas, passes per idea min {min(sizes)}, median {int(np.median(sizes))}, max {max(sizes)}, total {sum(sizes)}; sha256 {digest}")


if __name__ == "__main__":
    main()
