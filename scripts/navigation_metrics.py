"""H4 of prereg 01: how far into a page the model reads, from wiki-search eval traces.

Per scored episode: the number of wiki_read_section calls, and whether it is a wrong-section
failure, meaning reward 0 after reading at least one section of the question's source page while
no tool output contains the gold answer (both lowercased, punctuation removed). Per question, the
mean over its rollouts; overall, the mean over questions, as for accuracy. Errored episodes are
dropped, as in summarize_eval.py.

Summarize one run (on the server, from $PRIME_RL_DIR; writes results/evals/<name>/navigation.jsonl):
    uv run python $REPO/scripts/navigation_metrics.py <run dir> <name>
Compare two summarized runs, b - a with b the trained model: 95% paired bootstrap CIs over
questions, and the H4 verdict:
    python3 scripts/navigation_metrics.py --compare <name a> <name b>
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

BOOTSTRAP = 10_000
RESULTS = Path(__file__).resolve().parents[1] / "results/evals"
BANK_REVISION = "bd2138213717fa69d5df34514169d6bec0f5b6b3"
METRICS = ("section_reads", "wrong_section")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def section_arg(arguments) -> str:
    """The section a wiki_read_section call asked for; the renamed environment calls it `section`."""
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments or {}
        return str(args.get("section_id") or args.get("section") or "")
    except (json.JSONDecodeError, AttributeError):
        return ""


def source_pages() -> dict[int, str | None]:
    """Page id of each question-bank row's source page, None if the page is not in the corpus."""
    from datasets import load_dataset

    bank = load_dataset("willcb/wiki-trivia-questions", split="train", revision=BANK_REVISION)
    pages = load_dataset("willcb/rare-wiki-pages", split="train").select_columns(["title", "id"])
    page_id = dict(zip(pages["title"], pages["id"]))
    return {i: page_id.get(name.removesuffix(".md")) for i, name in enumerate(bank["filename"])}


def episode(record: dict, page_of: dict[int, str | None]) -> tuple[int, int, bool] | None:
    trace = (record.get("traces") or [{}])[0] or {}
    reward = ((trace.get("rewards") or {}).get("wiki-search") or {}).get("score")
    if not record.get("ok") or reward is None:
        return None
    data = record["task"]["data"]
    nodes = [n.get("message") or {} for n in trace.get("nodes") or []]
    reads = [
        call
        for m in nodes
        if m.get("role") == "assistant"
        for call in m.get("tool_calls") or []
        if call.get("name") == "wiki_read_section"
    ]
    page = page_of.get(data["idx"])
    read_source = page is not None and any(section_arg(c.get("arguments")).startswith(f"{page}:") for c in reads)
    tool_text = norm(" ".join(str(m.get("content") or "") for m in nodes if m.get("role") == "tool"))
    wrong_section = reward == 0 and read_source and norm(str(data["answer"])) not in tool_text
    return data["idx"], len(reads), wrong_section


def summarize(run_dir: Path, name: str) -> None:
    page_of = source_pages()
    per_question = defaultdict(list)
    for line in (run_dir / "traces.jsonl").open():
        if ep := episode(json.loads(line), page_of):
            per_question[ep[0]].append(ep[1:])
    rows = [
        {"idx": idx, "rollouts": len(eps), "section_reads": sum(r for r, _ in eps) / len(eps),
         "wrong_section": sum(w for _, w in eps) / len(eps)}
        for idx, eps in sorted(per_question.items())
    ]
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "navigation.jsonl").open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    missing = sum(page_of.get(idx) is None for idx in per_question)
    means = {key: sum(row[key] for row in rows) / len(rows) for key in METRICS}
    print(f"{name}: {len(rows)} questions, {missing} without their source page in the corpus; "
          f"section reads {means['section_reads']:.3f}, wrong-section failures {means['wrong_section']:.3f}")


def load(name: str) -> dict[int, dict]:
    return {row["idx"]: row for row in map(json.loads, (RESULTS / name / "navigation.jsonl").open())}


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
        result[key] = (mean_b, lo, hi)
        print(f"{key:14s} a {mean_a:.3f}  b {mean_b:.3f}  b - a {mean_b - mean_a:+.3f}  95% paired CI [{lo:+.3f}, {hi:+.3f}]")
    reads, wrong = result["section_reads"], result["wrong_section"]
    passes = reads[0] > 1.5 and reads[1] > 0 and wrong[2] < 0
    print(f"H4 {'passes' if passes else 'fails'}: b reads > 1.5 {reads[0] > 1.5}, reads CI above 0 {reads[1] > 0}, "
          f"wrong-section CI below 0 {wrong[2] < 0}")


def main() -> None:
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        summarize(Path(sys.argv[1]), sys.argv[2])


if __name__ == "__main__":
    main()
