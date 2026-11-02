"""Diagnose OLMo 3's malformed read_section calls on wiki-search.

Replays held-out episodes up to the turn where the model should call wiki_read_section (after
a search and a view_sections call) and samples that turn again under variants of the harness,
one change at a time, then counts how often the call comes back well formed.

Step 0 ("raw") skips the tool parser: the prompt is tokenized by the server's own chat
template (/tokenize) and sampled through /v1/completions, so the text is exactly what the
model wrote.

    uv run python $REPO/scripts/diagnose_tool_call.py <traces.jsonl> <engine base url> <out dir>
"""

import copy
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

EPISODES = 20
SAMPLES = 4  # per episode and variant at the model's default sampling, plus one greedy


def post(url: str, body: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=600))


def parse_objects(text: str) -> list[dict]:
    """Tool outputs are JSON objects printed one after another."""
    decoder, objects, i = json.JSONDecoder(), [], 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        obj, i = decoder.raw_decode(text, i)
        objects.append(obj)
    return objects


def load_episodes(path: str) -> list[dict]:
    episodes = []
    for record in map(json.loads, open(path)):
        trace = (record.get("traces") or [{}])[0] or {}
        messages = [n["message"] for n in trace.get("nodes") or []]
        assistant = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        if len(assistant) < 3:
            continue
        first, second = (messages[assistant[0]].get("tool_calls") or []), (messages[assistant[1]].get("tool_calls") or [])
        if not (first and second and first[0]["name"].endswith("search_pages") and second[0]["name"].endswith("view_sections")):
            continue
        history = []
        for m in messages[: assistant[2]]:
            m = dict(m)
            if m.get("tool_calls"):
                m["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in m["tool_calls"]
                ]
            history.append(m)
        tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in trace["tools"]]
        try:
            sections = parse_objects(history[-1]["content"])
        except json.JSONDecodeError:
            continue  # view_sections itself errored (e.g. a wrong page_id), so there is no read step to replay
        if not sections or "section_id" not in sections[0]:
            continue
        episodes.append({"idx": record["task"]["data"]["idx"], "history": history, "tools": tools, "sections": sections,
                         "model": trace["agent"]["config"]["model"]})
        if len(episodes) == EPISODES:
            break
    return episodes


def read_tool(tools: list[dict]) -> dict:
    return next(t for t in tools if t["function"]["name"].endswith("read_section"))


# Each variant changes one thing and returns (history, tools, expected argument name)
def baseline(ep):
    return ep["history"], ep["tools"], "section_id"


def json_list(ep):
    history = copy.deepcopy(ep["history"])
    for m in history:
        if m["role"] == "tool":
            m["content"] = json.dumps(parse_objects(m["content"]), indent=2)
    return history, ep["tools"], "section_id"


def plain_text(ep):
    history = copy.deepcopy(ep["history"])
    history[-1]["content"] = "\n".join(f"{s['section_name']} | {s['section_id']}" for s in ep["sections"])
    return history, ep["tools"], "section_id"


def rename_param(ep):
    tools = copy.deepcopy(ep["tools"])
    params = read_tool(tools)["function"]["parameters"]
    params["properties"] = {"section": params["properties"].pop("section_id")}
    params["required"] = ["section"]
    return ep["history"], tools, "section"


def rename_both(ep):
    history, tools, _ = rename_param(ep)
    history = copy.deepcopy(history)
    history[-1]["content"] = "\n".join(
        json.dumps({"section": s["section_id"], "section_name": s["section_name"]}, indent=2) for s in ep["sections"]
    )
    return history, tools, "section"


def plain_description(ep):
    tools = copy.deepcopy(ep["tools"])
    read_tool(tools)["function"]["description"] = "Read the text of one section of a page."
    return ep["history"], tools, "section_id"


def minimal(ep):
    section = ep["sections"][min(1, len(ep["sections"]) - 1)]["section_id"]
    history = [{"role": "user", "content": f"Use the tool to read the section with id {section}."}]
    return history, [read_tool(ep["tools"])], "section_id"


VARIANTS = [baseline, json_list, plain_text, rename_param, rename_both, plain_description, minimal]


def classify(message: dict, arg: str) -> str:
    calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    for call in calls:
        if call["function"]["name"].endswith("read_section"):
            args = json.loads(call["function"]["arguments"] or "{}")
            return "well formed" if args.get(arg) else "parsed, wrong argument"
    if calls:
        return "other tool"
    if "<function_calls>" in content or "read_section(" in content:
        return "unparsed call"
    return "no call"


def main() -> None:
    traces, engine, out = sys.argv[1], sys.argv[2].rstrip("/"), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(traces)
    model = episodes[0]["model"]
    print(f"{len(episodes)} episodes with search -> view_sections -> (read step)")

    # Step 0: raw text, no tool parser
    raw = Counter()
    for ep in episodes:
        tokens = post(f"{engine}/tokenize", {"model": model, "messages": ep["history"], "tools": ep["tools"],
                                              "add_generation_prompt": True})["tokens"]
        completions = post(f"{engine}/v1/completions", {"model": model, "prompt": tokens, "max_tokens": 80,
                                                         "temperature": 0, "n": 1})
        text = completions["choices"][0]["text"]
        raw["section_id=" if "section_id=" in text else "section_ \"" if 'section_ "' in text else "other"] += 1
        if ep is episodes[0]:
            prompt_text = post(f"{engine}/detokenize", {"model": model, "tokens": tokens})["prompt"]
            (out / "rendered_prompt_example.txt").write_text(prompt_text)
            print("raw greedy output, first episode:", repr(text))
    print("step 0, raw greedy text:", dict(raw))

    results = {"raw_greedy": dict(raw), "variants": {}}
    for variant in VARIANTS:
        counts = Counter()
        for ep in episodes:
            history, tools, arg = variant(ep)
            for temperature, n in [(None, SAMPLES), (0.0, 1)]:
                body = {"model": model, "messages": history, "tools": tools, "max_tokens": 200, "n": n}
                if temperature is not None:
                    body["temperature"] = temperature
                for choice in post(f"{engine}/v1/chat/completions", body)["choices"]:
                    counts[classify(choice["message"], arg)] += 1
        total = sum(counts.values())
        results["variants"][variant.__name__] = {k: v / total for k, v in counts.items()} | {"samples": total}
        print(f"{variant.__name__:18s} well formed {counts['well formed'] / total:6.1%}  {dict(counts)}")
    (out / "summary.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
