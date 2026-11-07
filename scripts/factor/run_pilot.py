#!/usr/bin/env python3
"""One run of prereg/03b-pilot.md: online RL from S2 under prereg 03's guards, then val ideas at 16 rollouts each, then the judge.

    tmux new-session -d -s lp-pilot-p1 "bash -c 'source setup/env.sh; cd \\$REPO; \\$PRIME_RL_DIR/.venv/bin/python scripts/factor/run_pilot.py p1'"
    tmux new-session -d -s lp-pilot-p2 "bash -c 'source setup/env.sh; cd \\$REPO; \\$PRIME_RL_DIR/.venv/bin/python scripts/factor/run_pilot.py p2 --after lp-pilot-p1'"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_stages import ONLINE_STEPS, PY, REPO, ROOT, keep_adapter, measure, note, online_once, run_steps, session_alive, shell, stop, wait_gpus_free  # noqa: E402

START = "factor-s2"
RANK = 32
SEED = 1
ROLLOUTS = 16
RUNS = {"p1": ("factor-p1-s2-ic", "ic"), "p2": ("factor-p2-s2-rank", "rank")}


def wait_for(session: str) -> None:
    """A session restarted by hand is gone for a moment, so look twice."""
    while True:
        if not session_alive(session):
            time.sleep(180)
            if not session_alive(session):
                return
        time.sleep(60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=sorted(RUNS))
    ap.add_argument("--after", default="", help="a tmux session to wait for first")
    args = ap.parse_args()
    if args.after:
        wait_for(args.after)
    name, arm = RUNS[args.run]
    start = f"{ROOT}/adapters/{START}"
    final = f"{ROOT}/adapters/{name}"
    extra, lr_note = f"--inference.vllm.seed {SEED}", None
    for attempt in (1, 2):
        wait_gpus_free("0,1,2,3")
        out = f"{ROOT}/outputs/{name}"
        if os.path.exists(out):
            os.rename(out, f"{out}-attempt{attempt}-{int(time.time())}")
        note("pilot RL", run=name, arm=arm, seed=SEED, start=start, attempt=attempt, lr=lr_note)
        started = time.monotonic()
        ended, directory = online_once(name, start, arm, RANK, extra)
        adapter = f"{directory}/broadcasts/step_{ONLINE_STEPS}" if directory else ""
        if ended == "finished" and os.path.exists(f"{adapter}/adapter_model.safetensors"):
            note("pilot RL finished", run=name, hours=round((time.monotonic() - started) / 3600, 2), table=run_steps(f"{directory}/metrics.jsonl"))
            break
        note("pilot RL ended early", run=name, ended=ended, attempt=attempt)
        if attempt == 2:
            stop("pilot RL failed twice", run=name)
        if ended.startswith("guard"):
            extra, lr_note = f"{extra} --trainer.optim.lr 1e-5", "1e-5"
    keep_adapter(name.removeprefix("factor-"), adapter)
    wait_gpus_free("0,1")
    code = shell(f"ROLLOUTS={ROLLOUTS} bash {REPO}/scripts/factor/eval_stage.sh {name} {final} val 0,1 18200 128", f"eval-{name}")
    if code != 0 or not measure(name):
        stop("evaluating the pilot failed", run=name, exit=code)
    shell(f"cd {REPO} && {PY} scripts/factor/judge_eval.py results/factor/agent/{name}.jsonl", f"eval-{name}")
    note("pilot RL on val", run=name, val=measure(name))


if __name__ == "__main__":
    if not (ROOT and REPO):
        sys.exit("source setup/env.sh first")
    main()
