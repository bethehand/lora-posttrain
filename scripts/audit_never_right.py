"""Sample training questions the base model never got right and record why, for a manual audit.

For each sampled question (0 of 8 in the filter pass) and its 8 rollouts: is the gold answer in
the source page at all; did any search return the source page; did the model view or read it;
did the gold answer appear in any text the tools returned; and the 8 final replies. Writes
results/audit/never_right_sample.jsonl and prints one block per question.

On the server, from $PRIME_RL_DIR:
    uv run python $REPO/scripts/audit_never_right.py [n] [seed]
"""

import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    repo, root = Path(os.environ["REPO"]), Path(os.environ["ROOT"])
    questions = [json.loads(line) for line in (repo / "results/evals/qwen3-4b-base-filter/train/questions.jsonl").open()]
    never = sorted(q["idx"] for q in questions if q["mean_reward"] == 0 and q["rollouts"] >= 6)
    sample = sorted(random.Random(seed).sample(never, n))

    bank = load_dataset("willcb/wiki-trivia-questions", split="train", revision="bd2138213717fa69d5df34514169d6bec0f5b6b3")
    pages = {row["title"]: row for row in load_dataset("willcb/rare-wiki-pages", split="train")}

    rollouts = defaultdict(list)
    for record in map(json.loads, (root / "outputs/evals/qwen3-4b-base-filter/train/merged/traces.jsonl").open()):
        if record["task"]["data"]["idx"] in sample:
            rollouts[record["task"]["data"]["idx"]].append(record)

    out_path = repo / "results/audit/never_right_sample.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        for idx in sample:
            row = bank[idx]
            title = row["filename"].removesuffix(".md")
            page = pages.get(title)
            page_id = page["id"] if page else None
            gold = str(row["answer"])
            facts = {"searched": 0, "viewed": 0, "read": 0, "gold_in_context": 0}
            finals = []
            for record in rollouts[idx]:
                trace = (record.get("traces") or [{}])[0] or {}
                nodes = [nd.get("message") or {} for nd in trace.get("nodes") or []]
                tool_text = " ".join(str(m.get("content")) for m in nodes if m.get("role") == "tool")
                calls = [c for m in nodes for c in (m.get("tool_calls") or [])]
                args = " ".join(c.get("arguments", "") for c in calls)
                facts["searched"] += bool(page_id and f'"page_id": "{page_id}"' in tool_text)
                facts["viewed"] += bool(page_id and f'"page_id": "{page_id}"' in args)
                facts["read"] += bool(page_id and f'"section_id": "{page_id}:' in args)
                facts["gold_in_context"] += norm(gold) in norm(tool_text)
                replies = [m.get("content") or "" for m in nodes if m.get("role") == "assistant" and m.get("content")]
                finals.append(replies[-1][:160] if replies else "")
            entry = {
                "idx": idx, "question": row["question"], "gold": gold, "page": title,
                "page_in_corpus": page is not None,
                "gold_in_page": bool(page) and norm(gold) in norm(page["content"]),
                "rollouts": len(rollouts[idx]), **facts, "finals": finals,
            }
            out.write(json.dumps(entry) + "\n")
            print(f"=== {idx} | {row['question']}\n    gold: {gold} | page: {title} | in corpus {entry['page_in_corpus']} | "
                  f"gold in page {entry['gold_in_page']}\n    of {entry['rollouts']} rollouts: page in search results "
                  f"{facts['searched']}, viewed {facts['viewed']}, read {facts['read']}, gold in tool text {facts['gold_in_context']}")
            for final in finals[:3]:
                print(f"    > {final!r}")


if __name__ == "__main__":
    main()
