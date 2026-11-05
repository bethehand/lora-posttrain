#!/usr/bin/env python3
"""Show one teacher trajectory as the text OLMo's chat template produces, and mark what is trained on.

    python scripts/factor/render_trace.py results/factor/teacher/trial.jsonl 0 > trajectory.txt
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs", "factor_idea"))

from transformers import AutoTokenizer  # noqa: E402

from teacher_traces import TOOL  # noqa: E402

MODEL = "/mnt/nvme1/bruce/models/Olmo-3-7B-Instruct"


def main() -> None:
    path, index = sys.argv[1], int(sys.argv[2])
    with open(path) as f:
        row = [json.loads(line) for line in f][index]
    tok = AutoTokenizer.from_pretrained(MODEL)
    messages = []
    for m in row["messages"]:
        if m["role"] == "user":
            messages.append({"role": "user", "content": m["content"]})
        elif m["role"] == "tool":
            messages.append({"role": "environment", "content": m["content"]})
        else:
            one = {"role": "assistant", "content": m["content"]}
            if m["calls"]:
                one["function_calls"] = "\n".join(f"test_factor(expression={json.dumps(e)})" for e in m["calls"])
            messages.append(one)
    text = tok.apply_chat_template(messages, tools=[TOOL], tokenize=False)
    # Assistant text is what the student generates, so it is what the loss covers.
    pieces, cursor, trained = [], 0, 0
    marker = "<|im_start|>assistant\n"
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            pieces.append(("context", text[cursor:]))
            break
        body = start + len(marker)
        pieces.append(("context", text[cursor:body]))
        ends = [e for e in (text.find("<|im_end|>", body), text.find(tok.eos_token, body)) if e >= 0]
        end = min(ends)
        closing = "<|im_end|>" if text.startswith("<|im_end|>", end) else tok.eos_token
        pieces.append(("trained", text[body : end + len(closing)]))
        cursor = end + len(closing)
    for kind, piece in pieces:
        if not piece:
            continue
        if kind == "trained":
            trained += len(tok(piece, add_special_tokens=False)["input_ids"])
            print("\n>>>>>>>> TRAINED (loss on these tokens)")
            print(piece)
            print("<<<<<<<<")
        else:
            print("\n-------- CONTEXT (read only, no loss)")
            print(piece)
    total = len(tok(text, add_special_tokens=False)["input_ids"])
    print(f"\n[tokens: {total} in all, {trained} trained on, {total - trained} context only]")
    print(json.dumps({k: row[k] for k in ("idea", "family", "stop", "tests", "factor", "final_was_tested", "parsed", "computes", "matches_idea", "reason")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
