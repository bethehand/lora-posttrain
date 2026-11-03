"""Keep the training questions that give RL a learning signal for the base model.

Reads results/evals/<name>/questions.jsonl from an 8-rollout eval of the base model on the train
split (configs/wiki-search/filter.toml) and adds splits.json[<split>]: training questions the base
model got right in some but not all scored rollouts. A group that is all right or all wrong has
zero advantage and is skipped by the trainer, so generating it only costs time. Other splits are
left as they are.

    python3 scripts/make_train_filter.py qwen3-4b-base-filter/train
    python3 scripts/make_train_filter.py olmo3-7b-base-filter/train train_mixed_olmo \
        "OLMo-3-7B-Instruct, read_section argument renamed to section,"
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "envs/wiki_search_heldout/wiki_search_heldout/splits.json"
MIN_SCORED = 6  # rollouts lost to judge timeouts are dropped; need this many to classify a question


def main() -> None:
    name = sys.argv[1]
    split = sys.argv[2] if len(sys.argv) > 2 else "train_mixed"
    model = sys.argv[3] if len(sys.argv) > 3 else "Qwen3-4B-Instruct-2507"
    rows = [json.loads(line) for line in (REPO / "results/evals" / name / "questions.jsonl").open()]
    splits = json.loads(SPLITS.read_text())
    train = set(splits["train"])
    assert {r["idx"] for r in rows} <= train, "filter eval must be on the train split"

    scored = [r for r in rows if r["rollouts"] >= MIN_SCORED]
    mixed = sorted(r["idx"] for r in scored if 0 < r["mean_reward"] < 1)
    always = sum(r["mean_reward"] == 1 for r in scored)
    never = sum(r["mean_reward"] == 0 for r in scored)

    splits[split] = mixed
    splits[f"{split}_rule"] = (
        f"training questions {model} got right in some but not all of 8 rollouts "
        f"(judge deepseek-flash, eval {name}); questions with fewer than {MIN_SCORED} scored rollouts dropped"
    )
    SPLITS.write_text(json.dumps(splits, indent=1) + "\n")
    print(f"{len(rows)} questions evaluated, {len(scored)} with >= {MIN_SCORED} scored rollouts")
    print(f"always right {always}, never right {never}, mixed {len(mixed)} -> {split}")


if __name__ == "__main__":
    main()
