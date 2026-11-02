"""wiki-search with a held-out split by source page.

Upstream wiki-search (verifiers b2e4e81) trains and evaluates on the same 20 questions.
This taskset loads the full question bank and serves one split from splits.json
(made by scripts/make_splits.py): `train`, `heldout`, `slide100` (the 100 questions behind
Prime Intellect's published eval), `train_sample`, or `train_mixed` (training questions with a
learning signal for the base model, added by scripts/make_train_filter.py). Tools, system prompt and judge are
upstream's. `use_tools = false` removes the tools, to measure how much a model answers from
memory: the judge only sees the final reply, so a remembered answer scores like a searched one.
`read_arg = "section"` renames read_section's argument from `section_id` (see servers.py).
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from wiki_search.servers.wiki import WikiSearchToolset
from wiki_search.taskset import SYSTEM, TriviaTask, TriviaTaskData, WikiSearchConfig
from wiki_search_heldout.servers import SectionArgWikiToolset

SPLITS = json.loads((Path(__file__).parent / "splits.json").read_text())
NO_TOOLS_SYSTEM = "Answer the question from your own knowledge. Reply with a concise final answer."


class HeldoutConfig(WikiSearchConfig):
    split: Literal["train", "heldout", "slide100", "train_sample", "train_mixed"] = "train"
    use_tools: bool = True
    read_arg: Literal["section_id", "section"] = "section_id"
    # Serve every num_shards-th question starting at shard, to split one eval across servers
    num_shards: int = 1
    shard: int = 0


class WikiSearchHeldoutTaskset(vf.Taskset[TriviaTask, HeldoutConfig]):
    @classmethod
    def toolsets(cls, config: HeldoutConfig) -> list[vf.Toolset]:
        if not config.use_tools:
            return []
        toolset = SectionArgWikiToolset if config.read_arg == "section" else WikiSearchToolset
        return [toolset(config.tools)]

    def load(self) -> list[TriviaTask]:
        from datasets import load_dataset

        rows = load_dataset(SPLITS["dataset"], split="train", revision=SPLITS["revision"])
        text = "\n".join(f"{q}\t{a}" for q, a in zip(rows["question"], rows["answer"]))
        if hashlib.sha256(text.encode()).hexdigest() != SPLITS["content_sha256"]:
            raise ValueError("question bank differs from the one splits.json was built from")
        system = SYSTEM if self.config.use_tools else NO_TOOLS_SYSTEM
        return [
            TriviaTask(
                # idx is the row in the full question bank, so it identifies a question across splits
                TriviaTaskData(
                    idx=i,
                    question=rows[i]["question"],
                    answer=str(rows[i]["answer"]),
                    prompt=f"{system}\n\nQuestion: {rows[i]['question']}",
                ),
                self.config.task,
            )
            for i in SPLITS[self.config.split][self.config.shard :: self.config.num_shards]
        ]
