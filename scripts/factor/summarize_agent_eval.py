#!/usr/bin/env python3
"""Summarize an eval run of factor-idea from its traces.jsonl.

    python scripts/factor/summarize_agent_eval.py <run dir or its parent> <name> [--show N]

Writes results/factor/agent/<name>.jsonl (one row per episode: replies, tests, the final
factor, its verdict and IC) and <name>.summary.json.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from statistics import mean, median


def episode(record: dict) -> dict:
    trace = (record.get("traces") or [{}])[0] or {}
    metrics = trace.get("metrics") or {}
    info = (trace.get("info") or {}).get("factor") or {}
    nodes = trace.get("nodes") or []
    steps = []
    for node in nodes:
        message = node.get("message") or {}
        role = message.get("role")
        if role == "assistant":
            calls = message.get("tool_calls") or []
            steps.append(dict(role=role, content=message.get("content") or "", calls=[c.get("arguments") if isinstance(c, dict) else str(c) for c in calls]))
        elif role == "tool":
            steps.append(dict(role=role, content=message.get("content") or ""))
    errors = (record.get("errors") or []) + (trace.get("errors") or [])
    calls = trace.get("calls") or []
    return dict(
        idea=info.get("idea"),
        family=info.get("family"),
        ok=bool(record.get("ok")),
        error=str(errors[0])[:200] if errors else None,
        stop=trace.get("stop_condition"),
        turns=len(calls),
        truncated_turns=sum(1 for c in calls if c.get("finish_reason") == "length"),
        tests=int(metrics.get("num_tests", 0)),
        has_answer=bool(metrics.get("has_answer")),
        parsed=bool(metrics.get("parsed")),
        computes=bool(metrics.get("computes")),
        matches_idea=bool(metrics.get("matches_idea")),
        ic=metrics.get("ic"),
        raw_ic=metrics.get("raw_ic"),
        reward=((trace.get("rewards") or {}).get("factor_reward") or {}).get("score"),
        reward_ic=metrics.get("reward_ic"),
        factor=info.get("expression"),
        skeleton=info.get("skeleton"),
        reason=info.get("reason"),
        steps=steps,
    )


def tool_result_kind(text: str) -> str:
    if text.startswith("computes: yes"):
        return "computed"
    if text.startswith("does not compute"):
        return "does not compute"
    if text.startswith("error"):
        return "expression error"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("name")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--out", default="results/factor/agent")
    args = ap.parse_args()
    paths = glob.glob(os.path.join(args.run, "traces.jsonl")) or glob.glob(os.path.join(args.run, "*", "traces.jsonl"))
    if len(paths) != 1:
        raise SystemExit(f"expected one traces.jsonl under {args.run}, found {paths}")
    with open(paths[0]) as f:
        rows = [episode(json.loads(line)) for line in f]

    n = len(rows)
    passed = [r for r in rows if r["matches_idea"]]
    ics = [r["ic"] for r in passed if r["ic"] is not None]
    tool_kinds = collections.Counter(tool_result_kind(s["content"]) for r in rows for s in r["steps"] if s["role"] == "tool")
    first_ok = sum(any(s["role"] == "tool" and tool_result_kind(s["content"]) == "computed" for s in r["steps"]) for r in rows)
    parse_reasons = collections.Counter(
        (r["reason"] or "").split(":")[0] if (r["reason"] or "").startswith("parse") else (r["reason"] or "")
        for r in rows
        if not r["matches_idea"]
    )
    summary = dict(
        name=args.name,
        episodes=n,
        errors=sum(r["error"] is not None for r in rows),
        stops=collections.Counter(str(r["stop"]) for r in rows).most_common(),
        has_answer=sum(r["has_answer"] for r in rows) / n,
        parsed=sum(r["parsed"] for r in rows) / n,
        computes=sum(r["computes"] for r in rows) / n,
        passed=len(passed) / n,
        reward_mean=mean(r["reward"] or 0.0 for r in rows),
        ic_mean_among_passed=mean(ics) if ics else None,
        ic_median_among_passed=median(ics) if ics else None,
        tests_per_episode=mean(r["tests"] for r in rows),
        tests_distribution=sorted(collections.Counter(r["tests"] for r in rows).items()),
        episodes_with_a_computed_test=first_ok / n,
        tool_results=tool_kinds.most_common(),
        turns_truncated=sum(r["truncated_turns"] for r in rows),
        turns=sum(r["turns"] for r in rows),
        failure_reasons=parse_reasons.most_common(),
        passed_by_idea={i: mean(r["matches_idea"] for r in rows if r["idea"] == i) for i in sorted({r["idea"] for r in rows if r["idea"]})},
    )
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{args.name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{args.out}/{args.name}.summary.json", "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "passed_by_idea"}, indent=1, ensure_ascii=False))
    shown = 0
    for r in sorted(rows, key=lambda r: -r["tests"]):
        if shown >= args.show:
            break
        shown += 1
        print(f"\n===== {r['idea']} tests={r['tests']} passed={r['matches_idea']} ic={r['ic']} reason={r['reason']!r}")
        for s in r["steps"]:
            print(f"[{s['role']}] {s['content'][:420]!r} {s.get('calls') or ''}")


if __name__ == "__main__":
    main()
