"""Write random scored episodes of an eval run as Markdown, for reading by hand (prereg 01: 50
trained-model trajectories read for judge gaming, such as listing several candidate answers).

Each episode shows the question, the gold answer, the judge's reward, every tool call with its
arguments, whether any tool output contains the gold answer (navigation_metrics.py's check), and
the final reply in full.

    python3 scripts/sample_trajectories.py <run dir> <out.md> [n=50] [seed=0]
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
    records = []
    for record in map(json.loads, (run_dir / "traces.jsonl").open()):
        trace = (record.get("traces") or [{}])[0] or {}
        reward = ((trace.get("rewards") or {}).get("wiki-search") or {}).get("score")
        if record.get("ok") and reward is not None:
            records.append((record, trace, reward))
    sample = random.Random(seed).sample(range(len(records)), n)

    lines = [f"# {n} random episodes of `{run_dir.name}` (seed {seed})", ""]
    for k, i in enumerate(sorted(sample), 1):
        record, trace, reward = records[i]
        data = record["task"]["data"]
        messages = [nd.get("message") or {} for nd in trace.get("nodes") or []]
        calls = [f"`{c.get('name')}` {c.get('arguments')}" for m in messages for c in m.get("tool_calls") or []]
        replies = [m for m in messages if m.get("role") == "assistant"]
        final = (replies[-1].get("content") or "").strip() if replies else ""
        tool_text = norm(" ".join(str(m.get("content") or "") for m in messages if m.get("role") == "tool"))
        gold_seen = "yes" if norm(str(data["answer"])) in tool_text else "no"
        lines += [f"## {k}. idx {data['idx']}, reward {reward:g}", "",
                  f"**Q:** {data['question']}", "", f"**Gold:** {data['answer']}", "",
                  "**Tool calls:** " + ("; ".join(calls) if calls else "none"), "",
                  f"**Gold in tool output:** {gold_seen}", "",
                  "**Final reply:**", "", "> " + final.replace("\n", "\n> "), ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {n} episodes of {len(records)} to {out}")


if __name__ == "__main__":
    main()
