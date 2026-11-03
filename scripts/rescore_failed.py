"""Retry the judge calls that failed during a wiki-search eval run (a prereg 02 deviation).

The DeepSeek API timed out for most episodes that started between 16:50 and 18:20 on
2026-11-03, during prereg 02's evals. Those rollouts finished; only their scoring failed. This
grades them again with the run's own judge (the production WikiSearchJudge on the task's question,
the reference answer and the last assistant message, as rejudge.py does), keeps every other
verdict exactly as the run gave it, and writes the run with the retried verdicts filled in to
<out run dir>/traces.jsonl, so summarize_eval.py and the analysis scripts run on it unchanged.
Retried records are marked "rescored", with their original errors kept there. Episodes that
failed for a reason other than a scoring error stay failed.

To check that a retried verdict means the same as an original one, it also re-grades --check
episodes the run scored (random, seed 0) and reports the agreement with their original verdicts.

On the server, from $PRIME_RL_DIR:
    uv run python $REPO/scripts/rescore_failed.py <run dir> <out run dir> [--model deepseek-v4-pro]
"""

import argparse
import asyncio
import json
import random
from pathlib import Path

from judge_agreement import grade
from wiki_search.judge import WikiSearchJudge, WikiSearchJudgeConfig


def last_reply(trace: dict) -> str | None:
    replies = [n["message"] for n in trace.get("nodes") or [] if (n.get("message") or {}).get("role") == "assistant"]
    return (replies[-1].get("content") or "").strip() if replies else None


def scoring_failed(record: dict) -> bool:
    """Every error is a scoring error and the rollout itself is there."""
    trace = (record.get("traces") or [{}])[0] or {}
    errors = (record.get("errors") or []) + (trace.get("errors") or [])
    return bool(errors) and all("scoring:" in str(e) for e in errors) and last_reply(trace) is not None


def reward(record: dict) -> float | None:
    trace = (record.get("traces") or [{}])[0] or {}
    return ((trace.get("rewards") or {}).get("wiki-search") or {}).get("score")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key-var", default="DEEPSEEK_API_KEY")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--check", type=int, default=200)
    args = parser.parse_args()

    records = [json.loads(line) for line in (args.run_dir / "traces.jsonl").open()]
    scored = [i for i, r in enumerate(records) if r.get("ok") and reward(r) is not None]
    retry = [i for i, r in enumerate(records) if not (r.get("ok") and reward(r) is not None) and scoring_failed(r)]
    check = random.Random(0).sample(scored, min(args.check, len(scored)))

    judge = WikiSearchJudge(WikiSearchJudgeConfig(model=args.model, base_url=args.base_url, api_key_var=args.api_key_var))
    sem = asyncio.Semaphore(args.concurrency)

    def ask(i: int):
        data, trace = records[i]["task"]["data"], records[i]["traces"][0]
        return grade(judge, sem, data["question"], str(data["answer"]), last_reply(trace))

    verdicts = await asyncio.gather(*(ask(i) for i in retry + check))
    retried, checked = verdicts[: len(retry)], verdicts[len(retry):]

    filled = 0
    for i, v in zip(retry, retried):
        if not isinstance(v, float):
            continue
        record = records[i]
        trace = record["traces"][0]
        trace["rewards"] = {**(trace.get("rewards") or {}), "wiki-search": {"score": v}}
        record["rescored"] = {"judge": args.model, "errors": (record.get("errors") or []) + (trace.get("errors") or [])}
        record["errors"], trace["errors"], record["ok"] = [], [], True
        filled += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "traces.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    pairs = [(reward(records[i]), v) for i, v in zip(check, checked) if isinstance(v, float)]
    summary = {
        "run_dir": str(args.run_dir), "out_dir": str(args.out_dir), "judge": args.model,
        "episodes": len(records), "scored_by_run": len(scored), "scoring_failed": len(retry),
        "filled": filled, "still_failed": len(records) - len(scored) - filled,
        "check_episodes": len(pairs),
        "check_agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
    }
    (args.out_dir / "rescore_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    asyncio.run(main())
