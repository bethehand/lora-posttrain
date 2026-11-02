"""Split the wiki-search question bank into train and held-out sets by source page.

Every question in willcb/wiki-trivia-questions was written from one Wikipedia page (its
`filename`). The held-out set takes whole pages, so no held-out question shares a page with
any training question:

1. the pages behind rows 0-99, which are the 100 questions in Prime Intellect's published
   eval, kept as the `slide100` subset for comparison with their numbers;
2. more pages, in sha256 order of their names, until the held-out set has at least
   MIN_HELDOUT questions.

`train_sample` is a fixed random 300 of the training questions, for measuring the
train/held-out gap after RL.

Run from the repo root:
    uv run --no-project --with datasets==5.1.0 python scripts/make_splits.py
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

DATASET = "willcb/wiki-trivia-questions"
REVISION = "bd2138213717fa69d5df34514169d6bec0f5b6b3"
MIN_HELDOUT = 300
TRAIN_SAMPLE = 300
OUT = Path(__file__).resolve().parents[1] / "envs/wiki_search_heldout/wiki_search_heldout/splits.json"


def content_sha256(rows) -> str:
    text = "\n".join(f"{q}\t{a}" for q, a in zip(rows["question"], rows["answer"]))
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    rows = load_dataset(DATASET, split="train", revision=REVISION)
    pages = rows["filename"]
    questions_on = defaultdict(list)
    for i, page in enumerate(pages):
        questions_on[page].append(i)

    slide100 = list(range(100))
    held = {pages[i] for i in slide100}
    n_held = sum(len(questions_on[p]) for p in held)
    for page in sorted(set(questions_on) - held, key=lambda p: hashlib.sha256(p.encode()).hexdigest()):
        if n_held >= MIN_HELDOUT:
            break
        held.add(page)
        n_held += len(questions_on[page])

    heldout = [i for i, p in enumerate(pages) if p in held]
    train = [i for i, p in enumerate(pages) if p not in held]
    assert not {pages[i] for i in train} & held
    assert set(slide100) <= set(heldout)
    train_sample = sorted(random.Random(0).sample(train, TRAIN_SAMPLE))

    OUT.write_text(json.dumps({
        "dataset": DATASET,
        "revision": REVISION,
        "content_sha256": content_sha256(rows),
        "rule": (
            "held-out = every question on the pages behind rows 0-99, then further pages in "
            f"sha256(name) order until >= {MIN_HELDOUT} questions; train = all other questions"
        ),
        "heldout_pages": sorted(held),
        "slide100": slide100,
        "heldout": heldout,
        "train_sample": train_sample,
        "train": train,
    }, indent=1) + "\n")
    n_pages = len(questions_on)
    print(f"train   {len(train):5d} questions on {n_pages - len(held):4d} pages")
    print(f"heldout {len(heldout):5d} questions on {len(held):4d} pages (includes slide100)")
    print(f"train_sample {len(train_sample)} of the training questions")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
