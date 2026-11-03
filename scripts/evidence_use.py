"""Where a wiki-search eval run's correct answers come from, and how its tool use fails
(exploratory, prereg 02's trajectory audit, results/audit/olmo_rl35_heldout_labels.md).

Per scored episode: judged right or not; whether the gold answer is in any tool output (the
normalized substring check of navigation_metrics.py, so spelling variants count as unseen);
whether any wiki_read_section call returned text rather than an error ("Section not found" and
bare headings count as text); whether the episode listed the page's sections before reading;
whether the final reply is garbled (over 800 characters with under 35% distinct words, or ten or
more "##"); the longest section read; and each kind of tool error. Writes
results/evals/<name>/evidence_use.json and prints it.

    python3 scripts/evidence_use.py <run dir> <name>
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

from navigation_metrics import RESULTS, norm

TOOL_ERRORS = ("Error executing tool", "error: unknown tool")


def error_kind(text: str) -> str:
    text = re.sub(r"'[^']*'", "X", text[:90])
    return re.sub(r"for \w+Arguments.*", "validation error", text, flags=re.S).strip()


def main() -> None:
    run_dir, name = Path(sys.argv[1]), sys.argv[2]
    counts, errors, error_episodes = Counter(), Counter(), Counter()
    n = 0
    for record in map(json.loads, (run_dir / "traces.jsonl").open()):
        trace = (record.get("traces") or [{}])[0] or {}
        reward = ((trace.get("rewards") or {}).get("wiki-search") or {}).get("score")
        if not record.get("ok") or reward is None:
            continue
        n += 1
        messages = [nd.get("message") or {} for nd in trace.get("nodes") or []]
        outputs = {m.get("tool_call_id"): str(m.get("content") or "") for m in messages if m.get("role") == "tool"}
        calls = [c for m in messages for c in m.get("tool_calls") or []]
        names = [c.get("name") for c in calls]
        reads = [outputs.get(c.get("id"), "") for c in calls if c.get("name") == "wiki_read_section"]
        read_ok = [r for r in reads if not r.startswith(TOOL_ERRORS)]
        seen = set()
        for out in outputs.values():
            if out.startswith(TOOL_ERRORS):
                errors[error_kind(out)] += 1
                seen.add(error_kind(out))
        error_episodes.update(seen)

        gold = norm(str(record["task"]["data"]["answer"]))
        gold_seen = bool(gold) and gold in norm(" ".join(outputs.values()))
        replies = [m.get("content") or "" for m in messages if m.get("role") == "assistant"]
        final = replies[-1] if replies else ""
        words = re.findall(r"\w+", final)
        garbled = len(final) > 800 and (len(set(words)) / max(1, len(words)) < 0.35 or final.count("##") >= 10)
        longest = max((len(r) for r in read_ok), default=0)

        outcome = "correct" if reward > 0 else "wrong"
        counts[f"{outcome}, gold {'in' if gold_seen else 'not in'} tool output, "
               f"{'a read returned text' if read_ok else 'no read returned text'}"] += 1
        counts["listed sections"] += "wiki_view_sections" in names
        counts["read"] += bool(reads)
        counts["read without listing sections"] += bool(reads) and "wiki_view_sections" not in names
        counts["garbled reply"] += garbled
        counts["read over 4,000 characters"] += longest > 4000
        counts["garbled after a read over 4,000 characters"] += garbled and longest > 4000

    result = {"run_dir": str(run_dir), "episodes": n, "counts": dict(sorted(counts.items())),
              "error_episodes": dict(error_episodes.most_common()), "error_calls": dict(errors.most_common())}
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence_use.json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
