"""Paired comparison of two eval results on the same questions.

Reads results/evals/<name>/questions.jsonl for both runs and reports each accuracy over the
questions both scored, the difference b - a, and a 95% bootstrap CI that resamples questions
(the pre-registered analysis).

    python3 scripts/compare_evals.py qwen3-4b-base/heldout qwen3-4b-base-renamed/heldout
"""

import json
import random
import sys
from pathlib import Path

BOOTSTRAP = 10_000
RESULTS = Path(__file__).resolve().parents[1] / "results/evals"


def load(name: str) -> dict[int, float]:
    return {r["idx"]: r["mean_reward"] for r in map(json.loads, (RESULTS / name / "questions.jsonl").open())}


def main() -> None:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    shared = sorted(a.keys() & b.keys())
    diffs = [b[i] - a[i] for i in shared]
    n = len(diffs)
    rng = random.Random(0)
    boot = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(BOOTSTRAP))
    lo, hi = boot[int(0.025 * BOOTSTRAP)], boot[int(0.975 * BOOTSTRAP)]
    print(f"{n} shared questions")
    print(f"a  {sys.argv[1]:40s} {sum(a[i] for i in shared) / n:.3f}")
    print(f"b  {sys.argv[2]:40s} {sum(b[i] for i in shared) / n:.3f}")
    print(f"b - a = {sum(diffs) / n:+.3f}   95% paired CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"questions where b is higher: {sum(d > 0 for d in diffs)}, lower: {sum(d < 0 for d in diffs)}, tied: {sum(d == 0 for d in diffs)}")


if __name__ == "__main__":
    main()
