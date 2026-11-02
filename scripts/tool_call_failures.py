"""Break down how episodes fail on wiki-search: malformed tool calls, tool errors, wrong answers.

An unparsed call left in the final reply means the parser could not read the model's
<function_calls> text, so the episode ended on it and the judge graded the markup.

    uv run python $REPO/scripts/tool_call_failures.py <traces.jsonl>
"""

import json
import sys
from collections import Counter


def main() -> None:
    kinds, questions_solved, n = Counter(), set(), 0
    for record in map(json.loads, open(sys.argv[1])):
        trace = (record.get("traces") or [{}])[0] or {}
        nodes = trace.get("nodes") or []
        reward = ((trace.get("rewards") or {}).get("wiki-search") or {}).get("score")
        if reward is None:
            continue
        n += 1
        replies = [(nd.get("message") or {}).get("content") or "" for nd in nodes if (nd.get("message") or {}).get("role") == "assistant"]
        tool_outputs = [str((nd.get("message") or {}).get("content")) for nd in nodes if (nd.get("message") or {}).get("role") == "tool"]
        final = replies[-1] if replies else ""
        if reward > 0:
            questions_solved.add(record["task"]["data"]["idx"])
            kinds["correct"] += 1
        elif "<function_calls>" in final or "(section_" in final:
            kinds["ended on an unparsed tool call"] += 1
        elif any(out.startswith("Error executing tool") for out in tool_outputs):
            kinds["tool error, then a wrong answer"] += 1
        elif not tool_outputs:
            kinds["no tool call, wrong answer"] += 1
        else:
            kinds["used tools, wrong answer"] += 1
    print(f"{n} scored episodes; questions with at least one correct rollout: {len(questions_solved)}")
    for kind, count in kinds.most_common():
        print(f"  {count:5d}  {count / n:6.1%}  {kind}")


if __name__ == "__main__":
    main()
