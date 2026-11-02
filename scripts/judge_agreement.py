"""How often does our judge agree with the judge behind the published wiki-search numbers?

Prime Intellect's published eval (verifiers baea257, November 2025) graded 7 models x 100
questions with gpt-4.1-mini. This script re-grades the same (question, reference answer,
final reply) triples with our judge, through the production code path
(wiki_search.judge.WikiSearchJudge), and reports agreement, Cohen's kappa, and each
model's accuracy under both judges.

On the server, from $PRIME_RL_DIR:
    uv run python $REPO/scripts/judge_agreement.py --model deepseek-flash
"""

import argparse
import asyncio
import json
import os
import urllib.request
from pathlib import Path

import openai
from wiki_search.judge import WikiSearchJudge, WikiSearchJudgeConfig

RETRYABLE = (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)

COMMIT = "baea257d230e0b981d787dc051254901ef641cc6"
# model directory name -> run id, as printed on the slide
RUNS = {
    "gpt-5-mini": "dc83cc41",
    "willcb--qwen3-4b-wiki-search": "094df25d",
    "gpt-4.1": "50519885",
    "gpt-4.1-mini": "052a81ae",
    "gpt-5-nano": "9a37a8bd",
    "gpt-4.1-nano": "8d8aecd4",
    "qwen--qwen3-4b-instruct-2507": "e762fd98",
}
URL = (
    "https://raw.githubusercontent.com/PrimeIntellect-ai/verifiers/{commit}/environments/"
    "wiki_search/outputs/evals/wiki-search--{model}/{run}/results.jsonl"
)


def load_rows(model: str, run: str, cache: Path) -> list[dict]:
    path = cache / f"{model}.jsonl"
    if not path.exists():
        urllib.request.urlretrieve(URL.format(commit=COMMIT, model=model, run=run), path)
    return [json.loads(line) for line in path.read_text().splitlines()]


def triple(row: dict) -> tuple[str, str, str]:
    question = next(m["content"] for m in reversed(row["prompt"]) if m["role"] == "user")
    replies = [m["content"] for m in row["completion"] if m["role"] == "assistant" and m.get("content")]
    return question, str(row["answer"]), replies[-1] if replies else ""


async def grade(judge: WikiSearchJudge, sem: asyncio.Semaphore, q: str, a: str, r: str) -> float | str:
    """1.0 or 0.0, else "unparseable" (no yes/no in the reply) or "failed" (API errors persisted)."""
    if not r.strip():
        return 0.0  # the production judge scores an empty reply 0 without a call
    async with sem:
        for attempt in range(5):
            try:
                result = await judge.evaluate(question=q, answer=a, response=r, positive="yes", negative="no")
                return float(result.parsed)
            except ValueError:
                return "unparseable"
            except RETRYABLE:
                await asyncio.sleep(10 * 2**attempt)
    return "failed"


def cohen_kappa(a: list[float], b: list[float]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--api-key-var", default="DEEPSEEK_API_KEY")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    root = Path(os.environ["ROOT"])
    cache = root / "cache/original_evals"
    cache.mkdir(parents=True, exist_ok=True)
    out = Path(os.environ["REPO"]) / "results/judge_agreement" / args.model
    out.mkdir(parents=True, exist_ok=True)

    judge = WikiSearchJudge(
        WikiSearchJudgeConfig(model=args.model, base_url=args.base_url, api_key_var=args.api_key_var)
    )
    sem = asyncio.Semaphore(args.concurrency)

    theirs, ours, per_model, disagreements = [], [], {}, []
    for model, run in RUNS.items():
        rows = load_rows(model, run, cache)
        triples = [triple(row) for row in rows]
        verdicts = await asyncio.gather(*(grade(judge, sem, *t) for t in triples))
        records = [
            {"model": model, "example_id": row["example_id"], "question": q, "answer": a,
             "reply": r, "theirs": row["judge_reward_func"], "ours": v}
            for row, (q, a, r), v in zip(rows, triples, verdicts)
        ]
        with (out / f"{model}.jsonl").open("w") as f:
            f.writelines(json.dumps(rec) + "\n" for rec in records)
        ok = [rec for rec in records if isinstance(rec["ours"], float)]
        disagreements += [rec for rec in ok if rec["ours"] != rec["theirs"]]
        per_model[model] = {
            "n": len(rows),
            "unparseable": verdicts.count("unparseable"),
            "failed": verdicts.count("failed"),
            "acc_theirs": sum(rec["theirs"] for rec in ok) / len(ok),
            "acc_ours": sum(rec["ours"] for rec in ok) / len(ok),
        }
        theirs += [rec["theirs"] for rec in ok]
        ours += [rec["ours"] for rec in ok]
        print(f"{model}: {per_model[model]}", flush=True)

    with (out / "disagreements.jsonl").open("w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in disagreements)

    summary = {
        "judge": {"model": args.model, "base_url": args.base_url},
        "reference_judge": "gpt-4.1-mini (verifiers baea257 eval outputs)",
        "n": len(ours),
        "agreement": sum(t == o for t, o in zip(theirs, ours)) / len(ours),
        "cohen_kappa": cohen_kappa(theirs, ours),
        "theirs_yes_ours_no": sum(t == 1 and o == 0 for t, o in zip(theirs, ours)),
        "theirs_no_ours_yes": sum(t == 0 and o == 1 for t, o in zip(theirs, ours)),
        "per_model": per_model,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
