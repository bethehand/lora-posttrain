"""Write random scored episodes of an eval run as Markdown, for reading by hand (prereg 01: 50
trained-model trajectories read for judge gaming, such as listing several candidate answers).

Each episode shows the question, the gold answer, the judge's reward, every tool call with its
arguments and what came back (the error text, or "ok" and the output's length), whether any tool
output contains the gold answer (navigation_metrics.py's check), and the final reply in full.
`correct-unseen` samples only episodes judged right whose gold answer is in no tool output.

    python3 scripts/sample_trajectories.py <run dir> <out.md> [n=50] [seed=0] [all|correct-unseen]
"""

import json
import random
import sys
from pathlib import Path

from navigation_metrics import norm


def main() -> None:
    run_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    only = sys.argv[5] if len(sys.argv) > 5 else "all"
    records = []
    for record in map(json.loads, (run_dir / "traces.jsonl").open()):
        trace = (record.get("traces") or [{}])[0] or {}
        reward = ((trace.get("rewards") or {}).get("wiki-search") or {}).get("score")
        if not (record.get("ok") and reward is not None):
            continue
        messages = [nd.get("message") or {} for nd in trace.get("nodes") or []]
        tool_text = norm(" ".join(str(m.get("content") or "") for m in messages if m.get("role") == "tool"))
        gold_seen = norm(str(record["task"]["data"]["answer"])) in tool_text
        if only == "correct-unseen" and not (reward > 0 and not gold_seen):
            continue
        records.append((record, messages, reward, gold_seen))
    sample = random.Random(seed).sample(range(len(records)), min(n, len(records)))

    title = "random episodes" if only == "all" else "random episodes judged right with the gold answer in no tool output"
    lines = [f"# {len(sample)} {title} of `{run_dir.name}` (seed {seed})", ""]
    for k, i in enumerate(sorted(sample), 1):
        record, messages, reward, gold_seen = records[i]
        data = record["task"]["data"]
        outputs = {m.get("tool_call_id"): str(m.get("content") or "") for m in messages if m.get("role") == "tool"}
        calls = []
        for m in messages:
            for c in m.get("tool_calls") or []:
                result = outputs.get(c.get("id"), "")
                status = result[:110].replace("\n", " ") if result.lower().startswith("error") else f"ok, {len(result)} chars"
                calls.append(f"`{c.get('name')}` {c.get('arguments')} → {status}")
        replies = [m for m in messages if m.get("role") == "assistant"]
        final = (replies[-1].get("content") or "").strip() if replies else ""
        gold_seen = "yes" if gold_seen else "no"
        lines += [f"## {k}. idx {data['idx']}, reward {reward:g}", "",
                  f"**Q:** {data['question']}", "", f"**Gold:** {data['answer']}", "",
                  "**Tool calls:** " + ("; ".join(calls) if calls else "none"), "",
                  f"**Gold in tool output:** {gold_seen}", "",
                  "**Final reply:**", "", "> " + final.replace("\n", "\n> "), ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(sample)} episodes of {len(records)} to {out}")


if __name__ == "__main__":
    main()
