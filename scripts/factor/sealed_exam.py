#!/usr/bin/env python3
"""The sealed exam of prereg 03: the held-out answers of S1, of each arm's final model and of DeepSeek, on 2026.

Rehearses every file on 2025 first, and opens the sealed period only if each gives back its own verdicts and
ICs. A file already scored is left alone, so a restart can't score it twice. The originals are never written.

    tmux new-session -d -s lp-sealed "bash -c 'source setup/env.sh; cd \\$REPO; \\$PRIME_RL_DIR/.venv/bin/python scripts/factor/sealed_exam.py --confirm \\"<phrase in panel.py>\\"'"
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_stages import interval  # noqa: E402
from run_stages import ONLINE_ORDER, PY, REPO, ROOT  # noqa: E402

AGENT = "results/factor/agent"
OUT = "results/factor/sealed"
IDEAS = "task/ideas_v2.jsonl"
SAME = 1e-9


# Kept pure so it can be tested.
def differences(rehearsed: list[dict], original: list[dict]) -> list[str]:
    """Where a rehearsal on 2025 doesn't give back the file."""
    if len(rehearsed) != len(original):
        return [f"{len(rehearsed)} rows against {len(original)}"]
    out = []
    for i, (a, b) in enumerate(zip(rehearsed, original)):
        if bool(a.get("matches_idea")) != bool(b.get("matches_idea")):
            out.append(f"row {i}: verdict")
        for name in ("ic", "raw_ic"):
            x, y = a.get(name), b.get(name)
            if (x is None) != (y is None) or (x is not None and abs(x - y) > SAME):
                out.append(f"row {i}: {name} {x} against {y}")
    return out


def per_idea(rows: list[dict], name: str) -> list[float]:
    """Mean of a field among passes per idea, for the ideas with a pass."""
    ideas = collections.defaultdict(list)
    for row in rows:
        if row.get("matches_idea") and row.get(name) is not None:
            ideas[row["idea"]].append(row[name])
    return [sum(v) / len(v) for v in ideas.values()]


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def answers() -> dict[str, str]:
    """Name of each file to score, and where its held-out answers are."""
    os.makedirs(OUT, exist_ok=True)
    teacher = f"{OUT}/inputs-deepseek-heldout.jsonl"
    if not os.path.exists(teacher):
        split_of = {r["id"]: r["split"] for r in read_jsonl(IDEAS)}
        write_jsonl(teacher, [r for r in read_jsonl(f"{AGENT}/v2-deepseek-tools.jsonl") if split_of.get(r.get("idea")) == "heldout"])
    files = {"s1": f"{AGENT}/s1-heldout.jsonl", "deepseek": teacher}
    for arm, seed in ONLINE_ORDER:
        files[f"{arm}-seed{seed}"] = f"{AGENT}/factor-s4-{arm}-seed{seed}-heldout.jsonl"
    return files


def rescore(path: str, out: str, *extra: str) -> None:
    code = subprocess.run([PY, "scripts/factor/rescore_eval.py", path, "--out", out, *extra], cwd=REPO).returncode
    if code:
        sys.exit(f"rescore_eval.py ended with {code} on {path}")


def rehearse(files: dict[str, str]) -> None:
    os.makedirs(f"{ROOT}/tmp/sealed-rehearsal", exist_ok=True)
    for name, path in files.items():
        out = f"{ROOT}/tmp/sealed-rehearsal/{name}.jsonl"
        rescore(path, out, "--period", "reward")
        found = differences(read_jsonl(out), read_jsonl(path))
        if found:
            sys.exit(f"{name} does not reproduce on 2025, so the sealed period stays closed: {found[:5]}")
    print(f"all {len(files)} files give back their verdicts and ICs on 2025", flush=True)


def report(files: dict[str, str]) -> None:
    arm = lambda name: ",".join(f"{OUT}/{name}-seed{seed}.jsonl" for seed in (1, 2))  # noqa: E731
    with open(f"{OUT}/comparisons.txt", "w") as out:
        out.write("Held-out ideas on the sealed period, 2026-01-01 to 2026-10-16. Each arm's seeds are pooled.\n\n")
        out.write("Mean IC among passes per idea, 95% CI over ideas, with the same answers on 2025 beside it\n")
        groups = {
            "S1": ["s1"],
            "S4 arm A": ["gates-seed1", "gates-seed2"],
            "S4 arm B": ["ic-seed1", "ic-seed2"],
            "S4 arm C": ["judged-seed1", "judged-seed2"],
            "DeepSeek": ["deepseek"],
        }
        for label, names in groups.items():
            sealed = [r for n in names for r in read_jsonl(f"{OUT}/{n}.jsonl")]
            before = [r for n in names for r in read_jsonl(files[n])]
            passes = sum(bool(r.get("matches_idea")) for r in sealed)
            out.write(f"  {label}: {passes}/{len(sealed)} pass, {sum(bool(r.get('matches_idea')) for r in before)} on 2025\n")
            for field in ("ic", "raw_ic"):
                now = interval(per_idea(sealed, field))
                then = interval(per_idea(before, field))
                out.write(f"    {field}: 2026 {now[0]:+.4f} ({now[1]:+.4f} to {now[2]:+.4f}), 2025 {then[0]:+.4f} ({then[1]:+.4f} to {then[2]:+.4f})\n")
        out.write("\n")
        out.flush()
        for label, later, earlier in (
            ("H8", arm("ic"), f"{OUT}/s1.jsonl"),
            ("Arm A against S1", arm("gates"), f"{OUT}/s1.jsonl"),
            ("Arm C against S1", arm("judged"), f"{OUT}/s1.jsonl"),
            ("Arm B against arm A", arm("ic"), arm("gates")),
            ("Arm C against arm B", arm("judged"), arm("ic")),
            ("Arm B against the teacher", arm("ic"), f"{OUT}/deepseek.jsonl"),
            ("S1 against the teacher", f"{OUT}/s1.jsonl", f"{OUT}/deepseek.jsonl"),
        ):
            out.write(f"== {label}: {later} against {earlier}\n")
            out.flush()
            subprocess.run([PY, "scripts/factor/compare_stages.py", later, earlier, "--split", "heldout", "--ideas", IDEAS], stdout=out, stderr=subprocess.STDOUT, cwd=REPO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", required=True)
    args = ap.parse_args()
    os.chdir(REPO)
    files = answers()
    rehearse(files)
    for name, path in files.items():
        out = f"{OUT}/{name}.jsonl"
        if not os.path.exists(out):
            rescore(path, out, "--period", "sealed", "--confirm", args.confirm)
    report(files)
    print(f"the sealed exam is written to {OUT}", flush=True)


if __name__ == "__main__":
    if not (ROOT and REPO):
        sys.exit("source setup/env.sh first")
    main()
