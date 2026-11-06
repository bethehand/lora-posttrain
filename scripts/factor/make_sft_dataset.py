#!/usr/bin/env python3
"""Turn teacher trajectories into the dataset prime-rl's SFT trainer reads.

Only trajectories worth imitating go in: on a family training may see, past the gates, handing in a
formula they tested, and never talking about the manual the student doesn't have. The saved file
keeps a turn's "train": false, so a trace can start from a student's mistake without teaching it.

    python scripts/factor/make_sft_dataset.py results/factor/teacher/pilot_v2.jsonl --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs", "factor_idea"))

from teacher_traces import TOOL  # noqa: E402


# Words from the teacher's own brief, or another language: the student would learn them as part of the job.
_OFF_SCRIPT = re.compile(r"[\u4e00-\u9fff]|as instructed|i was told|i am told|i'm told|the instructions", re.IGNORECASE)


def off_script(row: dict) -> bool:
    """The student is told one call per turn, and has no brief to quote."""
    turns = [m for m in row["messages"] if m["role"] == "assistant" and m.get("train") is not False]
    return (
        bool(row.get("leaks_the_manual"))
        or any(_OFF_SCRIPT.search(m["content"] or "") for m in turns)
        or any(len(m.get("calls") or []) > 1 for m in turns)
    )


def wire(history: list[dict]) -> list[dict]:
    """The file's messages in the form a live run sends to the server."""
    out, pending = [], []
    for m in history:
        if m["role"] == "user":
            out.append({"role": "user", "content": m["content"]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": pending.pop(0), "content": m["content"]})
        else:
            # Every assistant turn says, or the dataset loader's schema from the first rows won't fit the later ones.
            one = {"role": "assistant", "content": m["content"] or None, "train": m.get("train") is not False}
            if m["calls"]:
                ids = [f"call_{len(out)}_{i}" for i in range(len(m["calls"]))]
                one["tool_calls"] = [
                    {"id": i, "type": "function", "function": {"name": "test_factor", "arguments": json.dumps({"expression": e})}}
                    for i, e in zip(ids, m["calls"])
                ]
                pending.extend(ids)
            out.append(one)
    # An episode that ran out of turns ends on a tool result: no model generates one, and the trainer refuses it.
    while out[-1]["role"] == "tool":
        out.pop()
    return out


def keep(row: dict, split_of: dict[str, str], split: str) -> str:
    """Why a trajectory stays out, or "" if it goes in."""
    if split_of.get(row["idea"]) != split:
        return f"not in {split}"
    if not row["matches_idea"]:
        return "failed the gates"
    if not row["final_was_tested"]:
        return "handed in something untested"
    if off_script(row):
        return "off script"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--ideas", default="task/ideas_v2.jsonl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.ideas) as f:
        split_of = {r["id"]: r["split"] for r in map(json.loads, f)}
    tools = json.dumps([TOOL])
    rows, dropped = [], collections.Counter()
    for path in args.inputs:
        with open(path) as f:
            for row in map(json.loads, f):
                why = keep(row, split_of, args.split)
                if why:
                    dropped[why] += 1
                    continue
                rows.append({
                    "messages": wire(row["messages"]),
                    "tools": tools,
                    "weight": float(row.get("weight", 1.0)),
                    "idea": row["idea"],
                    "family": row["family"],
                })
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "train.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"kept {len(rows)} trajectories on {len({r['idea'] for r in rows})} ideas; dropped {dict(dropped)}")


if __name__ == "__main__":
    main()
