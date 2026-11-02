"""Re-grade a wiki-search eval run with another judge (prereg 01: every eval rollout is also
judged by deepseek-flash, the training judge, besides the primary deepseek-v4-pro).

Grades the episodes summarize_eval.py scored, with the production judge
(wiki_search.judge.WikiSearchJudge) and the production inputs: the task's question and reference
answer, and the last assistant message, stripped; an empty one scores 0 without a call. Writes
results/evals/<name>/questions.jsonl in summarize_eval.py's format, so compare_evals.py can pair
it, plus summary.json with the accuracy and the episode-level agreement with the run's own judge.

On the server, from $PRIME_RL_DIR:
    uv run python $REPO/scripts/rejudge.py <run dir> <name> [--model deepseek-flash]
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

from judge_agreement import cohen_kappa, grade
from summarize_eval import accuracy
from wiki_search.judge import WikiSearchJudge, WikiSearchJudgeConfig


def episode(record: dict) -> dict | None:
    trace = (record.get("traces") or [{}])[0] or {}
    reward = ((trace.get("rewards") or {}).get("wiki-search") or {}).get("score")
    if not record.get("ok") or reward is None:
        return None
    data = record["task"]["data"]
    replies = [n["message"] for n in trace.get("nodes") or [] if (n.get("message") or {}).get("role") == "assistant"]
    return {"idx": data["idx"], "question": data["question"], "answer": str(data["answer"]),
            "reply": (replies[-1].get("content") or "").strip() if replies else "", "original": reward}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("name")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key-var", default="DEEPSEEK_API_KEY")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--limit", type=int, help="grade only the first N episodes, for a smoke test")
    args = parser.parse_args()

    episodes = [e for e in map(episode, map(json.loads, (args.run_dir / "traces.jsonl").open())) if e][: args.limit]
    judge = WikiSearchJudge(WikiSearchJudgeConfig(model=args.model, base_url=args.base_url, api_key_var=args.api_key_var))
    sem = asyncio.Semaphore(args.concurrency)
    verdicts = await asyncio.gather(*(grade(judge, sem, e["question"], e["answer"], e["reply"]) for e in episodes))
    graded = [(e, v) for e, v in zip(episodes, verdicts) if isinstance(v, float)]

    per_question = defaultdict(list)
    for e, v in graded:
        per_question[e["idx"]].append(v)
    out = Path(os.environ["REPO"]) / "results/evals" / args.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "questions.jsonl").open("w") as f:
        for idx in sorted(per_question):
            f.write(json.dumps({"idx": idx, "mean_reward": sum(per_question[idx]) / len(per_question[idx]),
                                "rollouts": len(per_question[idx])}) + "\n")
    original, ours = [e["original"] for e, _ in graded], [v for _, v in graded]
    summary = {
        "run_dir": str(args.run_dir),
        "judge": args.model,
        "episodes": len(episodes),
        "unparseable": verdicts.count("unparseable"),
        "failed": verdicts.count("failed"),
        "all": accuracy(per_question),
        "agreement_with_run_judge": sum(a == b for a, b in zip(original, ours)) / len(graded),
        "kappa_with_run_judge": cohen_kappa(original, ours),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
