#!/usr/bin/env python3
"""The draws behind the split column of task/ideas_v2.jsonl, fixed before any model answered these ideas.

Held out: from families with at least five ideas, one taker family, two flow families and two price
families. Validation: one idea from each family left. The rest train.

    python scripts/factor/draw_splits.py       # checks the file against the draws
"""
from __future__ import annotations

import collections
import json
import random
import sys

HELD_OUT_SEED = 20261105
VAL_SEED = 20261106
# Which kind of data a key is about. Keep in step with KEY_GROUPS in sandbox.py.
GROUP = {
    "volume": "flow",
    "trades": "flow",
    "trade_size": "flow",
    "volume_on_up_days": "flow",
    "return": "price",
    "volatility": "price",
    "close_near_high": "price",
    "solid_body": "price",
    "taker_buy": "taker",
}


def draw(rows: list[dict]) -> dict[str, str]:
    families = collections.defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    group = {f: collections.Counter(GROUP[r["key"]] for r in members).most_common(1)[0][0] for f, members in families.items()}
    rng = random.Random(HELD_OUT_SEED)
    held = []
    for kind, count in (("taker", 1), ("flow", 2), ("price", 2)):
        held += rng.sample(sorted(f for f, g in group.items() if g == kind and len(families[f]) >= 5), count)
    rng = random.Random(VAL_SEED)
    val = {rng.choice(sorted(r["id"] for r in families[f])) for f in sorted(families) if f not in held}
    return {r["id"]: "heldout" if r["family"] in held else "val" if r["id"] in val else "train" for r in rows}


def main() -> None:
    with open(sys.argv[1] if len(sys.argv) > 1 else "task/ideas_v2.jsonl") as f:
        rows = [json.loads(line) for line in f]
    drawn = draw(rows)
    wrong = [r["id"] for r in rows if r["split"] != drawn[r["id"]]]
    print(dict(collections.Counter(drawn.values())), "differs from the file on:", wrong)
    if wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
