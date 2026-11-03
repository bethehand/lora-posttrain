"""Check that training reads OLMo 3 the way evaluation does (prereg 02 setup).

Evaluation sends the episode's messages to vLLM's chat endpoint, which renders them with OLMo's
chat template and reads tool calls with vLLM's olmo3 tool parser. Training renders the same
messages with renderers' DefaultRenderer and reads completions with setup/renderers_olmo3.py,
through prime-rl's /inference/v1/generate. For prompts taken from a recorded OLMo eval run (one
random assistant turn from each of n random episodes), this compares:

- prompt: the renderer's token ids against the ids vLLM's chat endpoint used;
- parse: on the chat endpoint's own completion, the tool calls vLLM found against the ones the
  OLMo parser finds in the completion's token ids;
- stop: whether completions from the training path end on <|im_end|> or the length limit.

It first runs both parsers on hand-written edge cases. Prints counts and every mismatch. Needs the
model served by prime-rl's inference server with --vllm.tool-call-parser olmo3, and
setup/install_olmo3_parser.sh.

    uv run python $REPO/scripts/olmo3_parity.py <traces.jsonl> <server base url> [n=300] [seed=0]
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from collections import Counter

import httpx
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from vllm.tool_parsers.olmo3_tool_parser import Olmo3PythonicToolParser

import verifiers.v1 as vf
from renderers.base import ToolCallParseStatus
from renderers.client import generate
from renderers.configs import DefaultRendererConfig
from renderers.default import DefaultRenderer
from renderers.olmo3_tool_parser import parse_text
from verifiers.v1.clients.train import tool_to_wire
from verifiers.v1.dialects.chat import message_to_wire

MODEL = "/mnt/nvme1/bruce/models/Olmo-3-7B-Instruct"
MAX_TOKENS = 512
CONCURRENCY = 16

EDGE_CASES = [
    '<function_calls>wiki_search_pages(query="Taylor Swift")</function_calls>',
    '<function_calls>wiki_read_section(section_ "taylor_swift:career")</function_calls>',
    '<function_calls>wiki_read_section(section_id="taylor_swift:career")</function_calls>',
    '<function_calls>wiki_view_sections(page_id="a")\nwiki_view_sections(page_id="b")</function_calls>',
    'Let me look.\n<function_calls>wiki_view_sections(page_id="x")</function_calls>',
    '<function_calls>wiki_view_sections(page_id="x")</function_calls> and then more text',
    '<function_calls>wiki_view_sections(page_id="x")',
    'wiki_search_pages(query="x")',
    'The answer is Dublin.',
    'Paris (France)',
    '<function_calls>f(a=null, b=true, c=[1, 2], d={"k": "v"})</function_calls>',
    '<function_calls>f(a=x)</function_calls>',
    '<function_calls>f("positional")</function_calls>',
    '<function_calls>a(x="1")</function_calls> text <function_calls>b(y="2")</function_calls>',
    '<function_calls>functions.wiki_search_pages(query="x")</function_calls>',
    '<function_calls>f(a=b"bytes")</function_calls>',
    '<function_calls>wiki_search_pages(query="Zürich")</function_calls>',
    '<function_calls>f(a="He said \\"hi\\"")</function_calls>',
    '<function_calls></function_calls>',
    '<function_calls>f(a="line1\nline2")</function_calls>',
    '<function_calls>wiki_search_pages(query=\'single quotes\')</function_calls>',
    '<function_calls>wiki_search_pages( query = "spaced" )</function_calls>',
]


def vllm_calls(parser: Olmo3PythonicToolParser, text: str) -> list | None:
    info = parser.extract_tool_calls(text, None)
    if not info.tools_called:
        return None
    return [(c.function.name, json.loads(c.function.arguments)) for c in info.tool_calls]


def edge_cases(tokenizer) -> int:
    parser = Olmo3PythonicToolParser(tokenizer)
    bad = 0
    for text in EDGE_CASES:
        theirs, ours = vllm_calls(parser, text), parse_text(text)
        if theirs != ours:
            bad += 1
            print(f"EDGE MISMATCH {text!r}\n  vllm {theirs}\n  ours {ours}")
    print(f"edge cases: {len(EDGE_CASES) - bad}/{len(EDGE_CASES)} agree")
    return bad


def sample_prompts(path: str, n: int, seed: int) -> list[tuple[list[dict], list[dict]]]:
    records = [json.loads(line) for line in open(path)]
    rng = random.Random(seed)
    prompts = []
    for record in rng.sample(records, min(n, len(records))):
        trace = vf.Trace.model_validate(record["traces"][0])
        turns = [i for i, node in enumerate(trace.nodes) if node.message.role == "assistant"]
        k = rng.choice(turns)
        messages = [message_to_wire(node.message) for node in trace.nodes[:k]]
        prompts.append((messages, [tool_to_wire(t) for t in trace.tools]))
    return prompts


async def check(prompt, renderer, tokenizer, http, client, sem, tally):
    messages, tools = prompt
    async with sem:
        response = await http.post("/chat/completions", json={
            "model": MODEL, "messages": messages, "tools": tools, "max_tokens": MAX_TOKENS,
            "return_token_ids": True})
        response.raise_for_status()
        data = response.json()
        trained = await generate(client=client, renderer=renderer, messages=messages, model=MODEL,
                                 tools=tools, sampling_params={"max_tokens": MAX_TOKENS})
    choice = data["choices"][0]

    ours_prompt = renderer.render(messages, tools=tools, add_generation_prompt=True).token_ids
    if ours_prompt == data["prompt_token_ids"]:
        tally["prompt match"] += 1
    else:
        tally["prompt mismatch"] += 1
        first = next((i for i, (a, b) in enumerate(zip(ours_prompt, data["prompt_token_ids"])) if a != b),
                     min(len(ours_prompt), len(data["prompt_token_ids"])))
        print(f"PROMPT MISMATCH at token {first}: renderer {tokenizer.decode(ours_prompt[first:first + 30])!r}"
              f" vs vLLM {tokenizer.decode(data['prompt_token_ids'][first:first + 30])!r}")

    theirs = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
              for c in choice["message"].get("tool_calls") or []] or None
    content_ids, parsed = renderer._tool_parser.extract(choice["token_ids"])
    ours = [(p.name, p.arguments) for p in parsed if p.status == ToolCallParseStatus.OK] or None
    tally["vllm called tools" if theirs else "vllm no tool call"] += 1
    if theirs == ours:
        tally["parse match"] += 1
    else:
        tally["parse mismatch"] += 1
        print(f"PARSE MISMATCH {tokenizer.decode(choice['token_ids'])!r}\n  vllm {theirs}\n  ours {ours}")

    ids = trained["completion_ids"]
    im_start, im_end = tokenizer.convert_tokens_to_ids(["<|im_start|>", "<|im_end|>"])
    if im_start in ids:
        tally["training path ran past the turn"] += 1
    elif ids and ids[-1] == im_end:
        tally["training path stopped on <|im_end|>"] += 1
    elif trained["finish_reason"] == "length":
        tally["training path hit the length limit"] += 1
    else:
        tally[f"training path ended on {tokenizer.convert_ids_to_tokens(ids[-1:])}"] += 1


async def main() -> None:
    path, base_url = sys.argv[1], sys.argv[2].rstrip("/")
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    bad = edge_cases(tokenizer)

    renderer = DefaultRenderer(tokenizer, DefaultRendererConfig(tool_parser="olmo3"))
    tally = Counter()
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(base_url=base_url, timeout=600) as http:
        client = AsyncOpenAI(base_url=base_url, api_key="unused")
        await asyncio.gather(*(check(p, renderer, tokenizer, http, client, sem, tally)
                               for p in sample_prompts(path, n, seed)))
    for key, count in sorted(tally.items()):
        print(f"  {count:5d}  {key}")
    if bad or tally["prompt mismatch"] or tally["parse mismatch"] or tally["training path ran past the turn"]:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
