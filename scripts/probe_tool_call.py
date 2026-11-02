"""Replay an eval episode up to its last assistant turn and sample that turn again.

Used to tell a model's own malformed tool calls apart from rendering or parsing bugs: the
conversation (including earlier tool calls and results) goes back to the server with the
episode's tool schemas, and each sample's raw content and parsed tool calls are printed.

    uv run python $REPO/scripts/probe_tool_call.py <traces.jsonl> <question idx> <base url> [n]
"""

import json
import sys
import urllib.request


def main() -> None:
    traces_path, idx, base_url = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    record = next(r for r in map(json.loads, open(traces_path)) if r["task"]["data"]["idx"] == idx)
    trace = record["traces"][0]
    model = trace["agent"]["config"]["model"]
    messages = [node["message"] for node in trace["nodes"]]
    last = max(i for i, m in enumerate(messages) if m["role"] == "assistant")

    history = []
    for m in messages[:last]:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in m["tool_calls"]
            ]
        history.append(m)
    tools = [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
        for t in trace["tools"]
    ]
    print("original:", repr(messages[last].get("content")))

    for temperature, samples in [(None, n), (0.0, 1)]:
        body = {"model": model, "messages": history, "tools": tools, "max_tokens": 200, "n": samples}
        if temperature is not None:
            body["temperature"] = temperature
        request = urllib.request.Request(
            f"{base_url}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        for choice in json.load(urllib.request.urlopen(request, timeout=300))["choices"]:
            message = choice["message"]
            calls = [(c["function"]["name"], c["function"]["arguments"]) for c in message.get("tool_calls") or []]
            label = "default" if temperature is None else f"temp {temperature}"
            print(f"{label:8s} content={(message.get('content') or '')[:150]!r} calls={calls}")


if __name__ == "__main__":
    main()
