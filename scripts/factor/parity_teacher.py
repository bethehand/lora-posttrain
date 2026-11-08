#!/usr/bin/env python3
"""Does SFT render a teacher trajectory exactly as a live run would see it?

For every assistant turn of every trajectory in a teacher file, the prompt before that turn is
rendered three ways: render_trace's view of the file (what SFT trains on), prime-rl's renderer
(what RL trains on), and vLLM's chat endpoint (what the model is served). All three must give the
same token ids, or the model would practise on one context and run on another. Needs a server with
the olmo3_keep_text parser.

    python scripts/factor/parity_teacher.py results/factor/teacher/trial.jsonl http://localhost:18100/v1
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs", "factor_idea"))

import httpx  # noqa: E402
from renderers.configs import DefaultRendererConfig  # noqa: E402
from renderers.default import DefaultRenderer  # noqa: E402
from teacher_traces import TOOL  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL = "/mnt/nvme1/bruce/models/Olmo-3-7B-Instruct"


def wire(history: list[dict]) -> list[dict]:
    """The file's messages in the form a live run sends to the server."""
    out = []
    pending: list[str] = []
    for m in history:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": pending.pop(0), "content": m["content"]})
        else:
            one = {"role": "assistant", "content": m["content"] or None}
            if m["calls"]:
                ids = [f"call_{len(out)}_{i}" for i in range(len(m["calls"]))]
                one["tool_calls"] = [
                    {"id": i, "type": "function", "function": {"name": "test_factor", "arguments": json.dumps({"expression": e})}}
                    for i, e in zip(ids, m["calls"])
                ]
                pending.extend(ids)
            out.append(one)
    return out


def sft_view(tokenizer, history: list[dict]) -> list[int]:
    messages = []
    for m in history:
        if m["role"] == "user":
            messages.append({"role": "user", "content": m["content"]})
        elif m["role"] == "tool":
            messages.append({"role": "environment", "content": m["content"]})
        else:
            one = {"role": "assistant", "content": m["content"]}
            if m["calls"]:
                one["function_calls"] = "\n".join(f"test_factor(expression={json.dumps(e)})" for e in m["calls"])
            messages.append(one)
    text = tokenizer.apply_chat_template(messages, tools=[TOOL], add_generation_prompt=True, tokenize=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


async def main() -> None:
    path, base_url = sys.argv[1], sys.argv[2].rstrip("/")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    renderer = DefaultRenderer(tokenizer, DefaultRendererConfig(tool_parser="olmo3_keep_text"))
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    tally = Counter()
    async with httpx.AsyncClient(base_url=base_url, timeout=300) as http:
        for row in rows:
            for turn, message in enumerate(row["messages"]):
                if message["role"] != "assistant":
                    continue
                history = row["messages"][:turn]
                messages = wire(history)
                response = await http.post("/chat/completions", json={
                    "model": MODEL, "messages": messages, "tools": [TOOL], "max_tokens": 1, "return_token_ids": True})
                response.raise_for_status()
                served = response.json()["prompt_token_ids"]
                rl = renderer.render(messages, tools=[TOOL], add_generation_prompt=True).token_ids
                sft = sft_view(tokenizer, history)
                for name, ids in (("sft", sft), ("rl", rl)):
                    if ids == served:
                        tally[f"{name} matches served"] += 1
                    else:
                        tally[f"{name} differs from served"] += 1
                        first = next((i for i, (a, b) in enumerate(zip(ids, served)) if a != b), min(len(ids), len(served)))
                        print(f"{name} DIFFERS at token {first}: {tokenizer.decode(ids[first:first + 25])!r} vs served {tokenizer.decode(served[first:first + 25])!r}")
    for key, count in sorted(tally.items()):
        print(f"  {count:4d}  {key}")
    if any("differs" in k for k in tally):
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
