#!/usr/bin/env python3
"""Other wordings of each training idea, so the same sentence doesn't fill the teacher data.

The meaning and the direction must stay. Only the original wording is ever used for evaluation.

    python scripts/factor/paraphrase_ideas.py --out task/ideas_v2_paraphrases.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os

from openai import AsyncOpenAI

ASK = """Rewrite this investment idea in {n} different ways. Each must be one plain English sentence a trader might say.

Rules:
- Keep exactly the same meaning and the same direction: what goes with better or worse returns must not change.
- Keep every window or time span it names, like "ten days" or "the last six hours".
- Add nothing new. No formulas, no field names, no numbers it doesn't already have.
- Vary the wording and the sentence structure.

Idea: {idea}

Reply with a JSON list of {n} strings and nothing else."""


async def rewrite(client, model: str, idea: dict, n: int, gate) -> dict:
    async with gate:
        for _ in range(5):
            reply = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": ASK.format(n=n, idea=idea["en"])}], max_tokens=3000)
            text = reply.choices[0].message.content or ""
            try:
                out = json.loads(text[text.index("[") : text.rindex("]") + 1])
            except ValueError:
                continue
            if isinstance(out, list) and len(out) == n and all(isinstance(x, str) and x.strip() for x in out):
                return {"id": idea["id"], "en": idea["en"], "paraphrases": [x.strip() for x in out]}
    raise SystemExit(f"no usable paraphrases for {idea['id']}")


async def main_async(args) -> list[dict]:
    client = AsyncOpenAI(base_url=args.base_url, api_key=os.environ[args.api_key_env], timeout=120)
    with open(args.ideas) as f:
        ideas = [r for r in map(json.loads, f) if r["split"] == "train"]
    gate = asyncio.Semaphore(8)
    return await asyncio.gather(*(rewrite(client, args.model, i, args.n, gate) for i in ideas))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ideas", default="task/ideas_v2.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com/v1")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    args = ap.parse_args()
    rows = asyncio.run(main_async(args))
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} ideas, {args.n} wordings each")


if __name__ == "__main__":
    main()
