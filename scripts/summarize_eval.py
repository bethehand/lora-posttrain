"""Summarize a prime-rl/verifiers eval run of wiki-search-heldout.

Reads <run dir>/traces.jsonl and writes, under results/evals/<name>/:
- questions.jsonl: one row per question (idx, mean reward over rollouts, rollouts, tool calls)
- summary.json: accuracy with a 95% bootstrap CI over questions, the slide100 subset
  (question-bank rows 0-99), tool use, truncation and error rates, judge token usage

    uv run python $REPO/scripts/summarize_eval.py <run dir> <name>
"""

import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

BOOTSTRAP = 10_000


def episode(record: dict) -> dict:
    # Errored episodes carry None for most fields
    trace = (record.get("traces") or [{}])[0] or {}
    nodes = trace.get("nodes") or []
    replies = [n["message"].get("content") or "" for n in nodes if (n.get("message") or {}).get("role") == "assistant"]
    calls = trace.get("calls") or []
    reward = ((trace.get("rewards") or {}).get("wiki-search") or {}).get("score")
    judge_usage = trace.get("extra_usage") or []
    errors = (record.get("errors") or []) + (trace.get("errors") or [])
    return {
        "idx": record["task"]["data"]["idx"],
        "ok": bool(record.get("ok")) and reward is not None,
        "error": str(errors[0])[:120] if errors else None,
        "reward": reward,
        "tool_calls": sum(1 for n in nodes if (n.get("message") or {}).get("role") == "tool"),
        "turns": len(calls),
        "truncated": any(c.get("finish_reason") == "length" for c in calls),
        "final_reply_chars": len(replies[-1]) if replies else 0,
        "stop": trace.get("stop_condition"),
        "judge_tokens": sum(u.get("prompt_tokens", 0) + u.get("completion_tokens", 0) for u in judge_usage),
    }


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    rng = random.Random(0)
    n = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(n)) / n for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP)]


def accuracy(per_question: dict[int, list[float]]) -> dict:
    means = [sum(v) / len(v) for v in per_question.values()]
    lo, hi = bootstrap_ci(means)
    return {"questions": len(means), "accuracy": sum(means) / len(means), "ci95": [lo, hi]}


def main() -> None:
    run_dir, name = Path(sys.argv[1]), sys.argv[2]
    episodes = [episode(json.loads(line)) for line in (run_dir / "traces.jsonl").open()]
    scored = [e for e in episodes if e["ok"]]

    per_question = defaultdict(list)
    for e in scored:
        per_question[e["idx"]].append(e["reward"])
    slide100 = {i: v for i, v in per_question.items() if i < 100}

    n = len(scored)
    summary = {
        "run_dir": str(run_dir),
        "episodes": len(episodes),
        "errored": len(episodes) - n,
        "error_kinds": dict(Counter(e["error"] for e in episodes if not e["ok"]).most_common(5)),
        "all": accuracy(per_question),
        "slide100": accuracy(slide100) if slide100 else None,
        "mean_tool_calls": sum(e["tool_calls"] for e in scored) / n,
        "no_tool_call_rate": sum(e["tool_calls"] == 0 for e in scored) / n,
        "truncation_rate": sum(e["truncated"] for e in scored) / n,
        "median_final_reply_chars": sorted(e["final_reply_chars"] for e in scored)[n // 2],
        "stop_conditions": dict(sorted({s: sum(e["stop"] == s for e in episodes) for s in {e["stop"] for e in episodes}}.items(), key=lambda kv: str(kv[0]))),
        "judge_tokens_total": sum(e["judge_tokens"] for e in scored),
    }

    out = Path(os.environ["REPO"]) / "results/evals" / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "questions.jsonl").open("w") as f:
        for idx in sorted(per_question):
            rows = [e for e in scored if e["idx"] == idx]
            f.write(json.dumps({"idx": idx, "mean_reward": sum(per_question[idx]) / len(per_question[idx]),
                                "rollouts": len(rows), "tool_calls": [e["tool_calls"] for e in rows]}) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
