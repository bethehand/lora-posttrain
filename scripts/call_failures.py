"""H4 of prereg 02: whether RL fixes OLMo's tool calls, from wiki-search eval traces.

A call failure is a scored episode with reward 0 that either ended on a tool call the parser could
not read (the final reply still holds <function_calls>: an unreadable call stays in the reply and
ends the episode), or got an error back from a tool ("Error executing tool ...", or "error: unknown
tool ..." for a tool name that does not exist). Per question, the share of its rollouts that are
call failures, and the same for the two parts; overall, the mean over questions, as for accuracy.
Errored episodes are dropped, as in summarize_eval.py.

Summarize one run (writes results/evals/<name>/call_failures.jsonl):
    python3 scripts/call_failures.py <run dir> <name>
Compare two summarized runs, b - a with b the trained model: 95% paired bootstrap CIs over
questions, and the H4 verdict:
    python3 scripts/call_failures.py --compare <name a> <name b>
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

BOOTSTRAP = 10_000
RESULTS = Path(__file__).resolve().parents[1] / "results/evals"
TOOL_ERRORS = ("Error executing tool", "error: unknown tool")
METRICS = ("call_failure", "unparsed", "tool_error")


def episode(record: dict) -> tuple[int, bool, bool] | None:
    """(question idx, ended on an unreadable call, got a tool error) for a failed episode; both
    False for a correct one; None for an errored one."""
    trace = (record.get("traces") or [{}])[0] or {}
    reward = ((trace.get("rewards") or {}).get("wiki-search") or {}).get("score")
    if not record.get("ok") or reward is None:
        return None
    idx = record["task"]["data"]["idx"]
    if reward > 0:
        return idx, False, False
    messages = [n.get("message") or {} for n in trace.get("nodes") or []]
    replies = [m.get("content") or "" for m in messages if m.get("role") == "assistant"]
    unparsed = "<function_calls>" in (replies[-1] if replies else "")
    tool_error = any(str(m.get("content") or "").startswith(TOOL_ERRORS)
                     for m in messages if m.get("role") == "tool")
    return idx, unparsed, tool_error


def summarize(run_dir: Path, name: str) -> None:
    per_question = defaultdict(list)
    for line in (run_dir / "traces.jsonl").open():
        if ep := episode(json.loads(line)):
            per_question[ep[0]].append(ep[1:])
    rows = [
        {"idx": idx, "rollouts": len(eps),
         "call_failure": sum(u or e for u, e in eps) / len(eps),
         "unparsed": sum(u for u, _ in eps) / len(eps),
         "tool_error": sum(e and not u for u, e in eps) / len(eps)}
        for idx, eps in sorted(per_question.items())
    ]
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "call_failures.jsonl").open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    means = {key: sum(row[key] for row in rows) / len(rows) for key in METRICS}
    print(f"{name}: {len(rows)} questions; call failures {means['call_failure']:.3f} "
          f"(unparsed {means['unparsed']:.3f}, tool error {means['tool_error']:.3f})")


def load(name: str) -> dict[int, dict]:
    return {row["idx"]: row for row in map(json.loads, (RESULTS / name / "call_failures.jsonl").open())}


def compare(name_a: str, name_b: str) -> None:
    a, b = load(name_a), load(name_b)
    shared = sorted(a.keys() & b.keys())
    n = len(shared)
    print(f"{n} shared questions; a = {name_a}, b = {name_b}")
    result = {}
    for key in METRICS:
        diffs = [b[i][key] - a[i][key] for i in shared]
        rng = random.Random(0)
        boot = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(BOOTSTRAP))
        lo, hi = boot[int(0.025 * BOOTSTRAP)], boot[int(0.975 * BOOTSTRAP)]
        mean_a, mean_b = (sum(run[i][key] for i in shared) / n for run in (a, b))
        result[key] = (mean_b - mean_a, lo, hi)
        print(f"{key:13s} a {mean_a:.3f}  b {mean_b:.3f}  b - a {mean_b - mean_a:+.3f}  95% paired CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"H4 {'passes' if result['call_failure'][2] < 0 else 'fails'}: call-failure CI below 0")


def main() -> None:
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        summarize(Path(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
