"""wiki-search with a held-out split by source page.

Upstream wiki-search (verifiers b2e4e81) trains and evaluates on the same 20 questions.
This taskset loads the full question bank and serves one split from splits.json
(made by scripts/make_splits.py): `train`, `heldout`, or `slide100`, the 100 questions
behind Prime Intellect's published eval. Tools, system prompt and judge are upstream's.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from wiki_search.servers.wiki import WikiSearchToolset
from wiki_search.taskset import SYSTEM, TriviaTask, TriviaTaskData, WikiSearchConfig

SPLITS = json.loads((Path(__file__).parent / "splits.json").read_text())


class HeldoutConfig(WikiSearchConfig):
    split: Literal["train", "heldout", "slide100"] = "train"


class WikiSearchHeldoutTaskset(vf.Taskset[TriviaTask, HeldoutConfig]):
    @classmethod
    def toolsets(cls, config: HeldoutConfig) -> list[vf.Toolset]:
        return [WikiSearchToolset(config.tools)]

    def load(self) -> list[TriviaTask]:
        from datasets import load_dataset

        rows = load_dataset(SPLITS["dataset"], split="train", revision=SPLITS["revision"])
        text = "\n".join(f"{q}\t{a}" for q, a in zip(rows["question"], rows["answer"]))
        if hashlib.sha256(text.encode()).hexdigest() != SPLITS["content_sha256"]:
            raise ValueError("question bank differs from the one splits.json was built from")
        return [
            TriviaTask(
                # idx is the row in the full question bank, so it identifies a question across splits
                TriviaTaskData(
                    idx=i,
                    question=rows[i]["question"],
                    answer=str(rows[i]["answer"]),
                    prompt=f"{SYSTEM}\n\nQuestion: {rows[i]['question']}",
                ),
                self.config.task,
            )
            for i in SPLITS[self.config.split]
        ]
