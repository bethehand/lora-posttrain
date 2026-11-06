#!/usr/bin/env python3
"""Online RL's models on the held-out ideas, once run_stages.py has ended, with the comparisons of prereg 03.

Waits for the driver's tmux session to end, evaluates each kept adapter on two pairs of GPUs, asks the
judge, then writes H5, H6 and what H7 and H9 need to results/factor/heldout_online_comparisons.txt. A run
with no adapter is left out, and the log says so. An arm's seeds are pooled. The audit that settles H7 and
H9 is read by hand afterwards. The sealed period is never read here.

    tmux new-session -d -s lp-heldout-online "bash -c 'source setup/env.sh; cd \\$REPO; \\$PRIME_RL_DIR/.venv/bin/python scripts/factor/heldout_online.py'"
"""
from __future__ import annotations

import collections
import json
import os
import re
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_stages import OFFLINE, ONLINE_ORDER, PY, REPO, ROOT, measure, note, session_alive, shell, wait_gpus_free  # noqa: E402

AGENT = "results/factor/agent"
OUT = "results/factor/heldout_online_comparisons.txt"
DEEPSEEK = "v2-deepseek-tools"


# Kept pure so it can be tested.
def unsigned(formula: str) -> str:
    """A formula without spaces, its sign or the brackets round it."""
    text = re.sub(r"\s+", "", formula)
    while True:
        if text.startswith("-1*"):
            text = text[3:]
        elif text.startswith("-"):
            text = text[1:]
        elif text.startswith("(") and text.endswith(")") and _closes_at_end(text):
            text = text[1:-1]
        else:
            return text


def _closes_at_end(text: str) -> bool:
    depth = 0
    for i, ch in enumerate(text):
        depth += (ch == "(") - (ch == ")")
        if depth == 0 and i < len(text) - 1:
            return False
    return True


def mirrored(ideas: list[dict], split: str = "heldout") -> set[str]:
    """Ideas of a split whose reference formula another idea of the split shares up to sign."""
    groups = collections.defaultdict(list)
    for idea in ideas:
        if idea["split"] == split:
            groups[unsigned(idea["reference"])].append(idea)
    return {i["id"] for group in groups.values() if len({i["direction"] for i in group}) > 1 for i in group}


def wait_for_driver() -> None:
    """A driver restarted by hand leaves its session gone for a moment, so look twice."""
    while True:
        if not session_alive("lp-stages"):
            time.sleep(180)
            if not session_alive("lp-stages"):
                return
        time.sleep(60)


def evaluate_all(runs: list[str]) -> list[str]:
    queue, done, lock = list(runs), [], threading.Lock()

    def worker(gpus: str, port: int) -> None:
        while True:
            with lock:
                if not queue:
                    return
                run = queue.pop(0)
            name = f"{run}-heldout"
            for attempt in (1, 2):
                if measure(name) is not None:
                    break
                wait_gpus_free(gpus)
                code = shell(f"bash {REPO}/scripts/factor/eval_stage.sh {name} {ROOT}/adapters/{run} heldout {gpus} {port} 128", f"eval-{name}")
                note("held-out eval ended", run=name, attempt=attempt, exit=code)
            if measure(name) is None:
                note("no held-out result", run=name)
                continue
            shell(f"cd {REPO} && {PY} scripts/factor/judge_eval.py {AGENT}/{name}.jsonl", f"eval-{name}")
            note("held out", stage=run, result=measure(name))
            with lock:
                done.append(run)

    threads = [threading.Thread(target=worker, args=a) for a in (("0,1", 18200), ("2,3", 18400))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return done


def pooled(arm: str, done: list[str]) -> str | None:
    files = [f"{AGENT}/{run}-heldout.jsonl" for a, seed in ONLINE_ORDER if a == arm for run in [f"factor-s4-{a}-seed{seed}"] if run in done]
    return ",".join(files) or None


def compare(out, label: str, later: str | None, earlier: str | None, ideas: str = "task/ideas_v2.jsonl") -> None:
    out.write(f"== {label}: {later} against {earlier}\n")
    out.flush()
    if not later or not earlier:
        out.write("  left out: a side has no results\n")
        out.flush()
        return
    subprocess.run([PY, "scripts/factor/compare_stages.py", later, earlier, "--split", "heldout", "--ideas", ideas], stdout=out, stderr=subprocess.STDOUT, cwd=REPO)


def comparisons(done: list[str], path: str = OUT) -> None:
    with open(f"{REPO}/task/ideas_v2.jsonl") as f:
        ideas = [json.loads(line) for line in f]
    pairs = mirrored(ideas)
    only_unmirrored = f"{ROOT}/tmp/ideas_unmirrored.jsonl"
    with open(only_unmirrored, "w") as f:
        for idea in ideas:
            f.write(json.dumps({**idea, "split": "mirrored" if idea["id"] in pairs else idea["split"]}, ensure_ascii=False) + "\n")
    stage = lambda name: f"{AGENT}/{name}-heldout.jsonl"  # noqa: E731
    ic, gates, judged = (pooled(arm, done) for arm in ("ic", "gates", "judged"))
    with open(f"{REPO}/{path}", "w") as out:
        out.write(f"Held-out ideas, {len(pairs)} of them in mirrored pairs. Each arm's seeds are pooled. Runs with results: {', '.join(done)}\n")
        compare(out, "H5", ic, stage(f"{OFFLINE}-ic"))
        compare(out, "H6", ic, gates)
        compare(out, "H6 on the ideas outside mirrored pairs", ic, gates, only_unmirrored)
        compare(out, "For H7, judge's view before the audit", ic, stage("s2"))
        compare(out, "For H9, judge's view before the audit", judged, ic)
        compare(out, "Arm A, online against offline", gates, stage(f"{OFFLINE}-gates"))
        compare(out, "Arm C, online against offline", judged, stage(f"{OFFLINE}-judged"))
        compare(out, "Arm B against S1", ic, stage("s1"))
        compare(out, "Arm B against the teacher", ic, f"{AGENT}/{DEEPSEEK}.jsonl")


def main() -> None:
    os.chdir(REPO)
    note("held-out evals for online RL are waiting for the driver", pid=os.getpid())
    wait_for_driver()
    runs = [f"factor-s4-{arm}-seed{seed}" for arm, seed in ONLINE_ORDER]
    ready = [r for r in runs if os.path.exists(f"{ROOT}/adapters/{r}/adapter_model.safetensors") and measure(r)]
    note("the driver has ended", runs_with_adapters=ready, missing=[r for r in runs if r not in ready])
    done = evaluate_all(ready)
    comparisons(done)
    note("online RL's held-out comparisons written", file=OUT, passes={r: round(measure(f"{r}-heldout")["pass"], 3) for r in done})


if __name__ == "__main__":
    if not (ROOT and REPO):
        sys.exit("source setup/env.sh first")
    main()
