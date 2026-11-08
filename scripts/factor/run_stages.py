#!/usr/bin/env python3
"""The stages after SFT1's sweep, unattended on the server, with every choice made by the rules in prereg 03.

Waits for the sweep's evaluations, picks rank and steps, then runs S2, S3 for each arm, online RL pilots
and online RL. Each choice goes to results/factor/selection.jsonl with its numbers. A step that fails stops
the run with the GPUs free, so a fault isn't carried forward. Work already done is kept on a restart.
Held-out ideas are read only once results/factor/frozen.txt exists, which the freeze writes. The sealed
period is never read here.

    tmux new-session -d -s lp-stages "bash -c 'source setup/env.sh; cd \\$REPO; \\$PRIME_RL_DIR/.venv/bin/python scripts/factor/run_stages.py'"
"""
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from statistics import mean

ROWS_PER_STEP = 14.4  # 2,021 rows took 140 steps an epoch at 8 sequences of 4,096 tokens
SWEEP_STEPS = (70, 140, 210, 280, 350, 420)
RANK_MARGIN = STEP_MARGIN = 0.05  # val pass rate
REWARD_MARGIN = 0.02  # val reward, for S2 and S3
OFFLINE_FALL = 0.10  # val pass rate lost from S2 that sends an arm's S4 back to S2
ENTROPY_BAND = (0.3, 6.0)
MISMATCH_KL = 0.01
REWARD_FALL = 0.30  # share of its starting reward that online RL may lose
OFFLINE_LR = "3e-5"
OFFLINE = "s3p"  # S3 on its positive rows. The full version, s3, broke the syntax on val and stays on disk to show it
ONLINE_STEPS = 35
ONLINE_ORDER = (("ic", 1), ("judged", 1), ("gates", 1), ("ic", 2), ("judged", 2), ("gates", 2))
FREEZE_WAIT_HOURS = 2


# The rules, kept pure so they can be tested.
def earliest_within(scores: dict[int, float], margin: float) -> int:
    """The earliest checkpoint within ``margin`` of the best."""
    best = max(scores.values())
    return min(step for step, value in scores.items() if value >= best - margin)


def pick_rank(best: dict[int, float], margin: float = RANK_MARGIN) -> int:
    """The smallest rank within ``margin`` of the best."""
    top = max(best.values())
    return min(rank for rank, value in best.items() if value >= top - margin)


def needs_rank_128(best: dict[int, float], margin: float = RANK_MARGIN) -> bool:
    """A bigger adapter is only worth trying if the step from 8 to 32 clearly helped."""
    return best[32] > best[8] + margin


def guard_trips(steps: dict[int, dict[str, float]], band: tuple[float, float] = ENTROPY_BAND, kl: float = MISMATCH_KL) -> str | None:
    """Prereg 02's guard over steps 1 to 10: two consecutive steps out of bounds."""
    for step in range(1, 10):
        now, after = steps.get(step), steps.get(step + 1)
        if not now or not after or not all(k in d for d in (now, after) for k in ("ent", "kl")):
            continue
        if now["kl"] > kl and after["kl"] > kl:
            return f"mismatch_kl above {kl} at steps {step} and {step + 1}"
        if not band[0] <= now["ent"] <= band[1] and not band[0] <= after["ent"] <= band[1]:
            return f"entropy outside {band} at steps {step} and {step + 1}"
    return None


def reward_falls(steps: dict[int, dict[str, float]], fall: float = REWARD_FALL) -> str | None:
    """Skill can go with entropy and KL looking fine. Any five steps running against the first three."""
    rewards = {step: d["reward"] for step, d in steps.items() if "reward" in d}
    if not all(step in rewards for step in (1, 2, 3)):
        return None
    start = mean(rewards[step] for step in (1, 2, 3))
    for step in rewards:
        window = [rewards.get(k) for k in range(step, step + 5)]
        if step > 3 and None not in window and mean(window) < start * (1 - fall):
            return f"reward {mean(window):.3f} over steps {step} to {step + 4}, from {start:.3f} over steps 1 to 3"
    return None


def half_epoch(rows: int) -> int:
    return max(1, round(rows / ROWS_PER_STEP / 2))


def summarize(rows: list[dict]) -> dict[str, float | int]:
    """What a stage is chosen by. A failed episode counts in the denominator."""
    passed = [r for r in rows if r.get("matches_idea")]
    ics = [r["ic"] for r in passed if r.get("ic") is not None]
    return {
        "episodes": len(rows),
        "failed": sum(r.get("idea") is None for r in rows),
        "pass": len(passed) / len(rows),
        "answered": mean(bool(r.get("factor")) for r in rows),
        "reward_ic": mean(r.get("reward_ic") or 0.0 for r in rows),
        "reward_judged": mean((r.get("reward_ic") or 0.0) if r.get("faithful") else 0.0 for r in rows),
        "ic_among_passes": mean(ics) if ics else 0.0,
        "tests": mean(r.get("tests", 0) for r in rows),
    }


ROOT = os.environ.get("ROOT", "")
REPO = os.environ.get("REPO", "")
PY = os.path.join(os.environ.get("PRIME_RL_DIR", ""), ".venv", "bin", "python")
DATA = f"{ROOT}/data/factor_idea_v1/sft"
METRIC = {"ic": "reward_ic", "gates": "pass", "judged": "reward_judged"}
_lock = threading.Lock()


def note(event: str, **fields) -> None:
    line = {"at": dt.datetime.now().strftime("%m-%d %H:%M:%S"), "event": event, **fields}
    with _lock:
        print(json.dumps(line, ensure_ascii=False), flush=True)
        with open(f"{REPO}/results/factor/selection.jsonl", "a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def stop(why: str, **fields) -> None:
    note("stopped", why=why, **fields)
    os._exit(1)


def shell(command: str, log: str) -> int:
    with open(f"{ROOT}/logs/{log}.log", "a") as out:
        return subprocess.run(["bash", "-c", command], stdout=out, stderr=subprocess.STDOUT).returncode


def session_alive(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", f"={name}"], capture_output=True).returncode == 0


def wait_gpus_free(gpus: str, minutes: int = 10) -> None:
    wanted = set(gpus.split(","))
    for _ in range(minutes * 6):
        used = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        busy = [g for g, m in (line.split(", ") for line in used.strip().splitlines()) if g in wanted and int(m) > 1000]
        if not busy:
            return
        time.sleep(10)
    stop("GPUs did not free up", gpus=gpus, busy=busy)


def rows_in(path: str) -> int:
    with open(path) as f:
        return sum(1 for _ in f)


def measure(name: str) -> dict | None:
    path = f"{REPO}/results/factor/agent/{name}.jsonl"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return summarize(rows) if rows else None


def broadcasts(name: str) -> str:
    return f"{ROOT}/outputs/sft/{name}/{name}/broadcasts"


def train(name: str, data: str, steps: int, rank: int, gpus: str, every: int, start: str = "", lr: str | None = None) -> None:
    """One SFT run, unless it already finished. On several GPUs, one failure falls back to two."""
    if os.path.exists(f"{broadcasts(name)}/step_{steps}/.finished"):
        note("kept", run=name)
        return
    while True:
        wait_gpus_free(gpus)
        shutil.rmtree(f"{ROOT}/outputs/sft/{name}", ignore_errors=True)
        extra = f"--optim.lr {lr}" if lr else ""
        note("training", run=name, data=os.path.basename(data), steps=steps, rank=rank, gpus=gpus, every=every, start=start, lr=lr)
        code = shell(f"bash {REPO}/scripts/factor/train_sft.sh {name} {data} {steps} {rank} {gpus} {every} '{start}' {extra}", f"sft-{name}")
        if code == 0 and os.path.exists(f"{broadcasts(name)}/step_{steps}/.finished"):
            note("trained", run=name)
            return
        if gpus.count(",") == 3:
            note("falling back to two GPUs", run=name, exit=code)
            gpus = "0,1"
            continue
        stop("training failed", run=name, exit=code, log=f"sft-{name}.log")


def evaluate(prefix: str, adapters: str, steps: list[int], gpus: str, port: int) -> None:
    """Val evals of a run's adapters from one server, skipping those already done."""
    missing = [s for s in steps if measure(f"{prefix}-step{s}") is None]
    if missing:
        wait_gpus_free(gpus)
        joined = " ".join(map(str, missing))
        code = shell(f"ADAPTER_WAIT=120 bash {REPO}/scripts/factor/eval_steps.sh {prefix} {adapters} val {gpus} {port} {joined}", f"eval-{prefix}")
        if code != 0:
            stop("evaluation failed", run=prefix, exit=code)
    absent = [s for s in steps if measure(f"{prefix}-step{s}") is None]
    if absent:
        stop("evaluation left no result", run=prefix, steps=absent)


def keep_adapter(stage: str, source: str) -> str:
    target = f"{ROOT}/adapters/factor-{stage}"
    os.makedirs(target, exist_ok=True)
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy(f"{source}/{name}", f"{target}/{name}")
    with open(f"{target}/adapter_model.safetensors", "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    with open(f"{REPO}/results/factor/adapters.md", "a") as f:
        f.write(f"- {stage}: `{digest}`, from `{source}`\n")
    note("adapter kept", stage=stage, path=target, sha256=digest)
    return target


def choose_s1() -> tuple[int, str]:
    for session in ("lp-eval-r32", "lp-eval-r8", "lp-sft-r32", "lp-sft-r8"):
        while session_alive(session):
            time.sleep(30)
    sweep = {}
    for rank in (8, 32):
        found = {s: measure(f"s1-r{rank}-step{s}") for s in SWEEP_STEPS}
        sweep[rank] = {s: m for s, m in found.items() if m}
        if not sweep[rank]:
            stop("no results for a rank of the sweep", rank=rank)
    best = {r: max(m["pass"] for m in v.values()) for r, v in sweep.items()}
    note("sweep", best=best, curve={r: {s: round(m["pass"], 3) for s, m in v.items()} for r, v in sweep.items()},
         answered={r: {s: round(m["answered"], 3) for s, m in v.items()} for r, v in sweep.items()})
    if needs_rank_128(best):
        train("s1-r128", f"{DATA}/s1", SWEEP_STEPS[-1], 128, "0,1,2,3", SWEEP_STEPS[0])
        evaluate("s1-r128", broadcasts("s1-r128"), list(SWEEP_STEPS), "2,3", 18400)
        sweep[128] = {s: measure(f"s1-r128-step{s}") for s in SWEEP_STEPS}
        best[128] = max(m["pass"] for m in sweep[128].values())
    rank = pick_rank(best)
    step = earliest_within({s: m["pass"] for s, m in sweep[rank].items()}, STEP_MARGIN)
    note("chose S1", rank=rank, step=step, best=best, val=sweep[rank][step])
    return rank, keep_adapter("s1", f"{broadcasts(f's1-r{rank}')}/step_{step}")


def choose_stage(stage: str, data: str, rank: int, start: str, gpus: str, port: int, metric: str, lr: str | None = None) -> tuple[str, dict]:
    every = half_epoch(rows_in(f"{data}/train.jsonl"))
    steps = [every * k for k in range(1, 7)]
    train(stage, data, steps[-1], rank, gpus, every, start, lr)
    eval_gpus = "2,3" if gpus.count(",") == 3 else gpus
    evaluate(stage, broadcasts(stage), steps, eval_gpus, port)
    names = [f"{stage}-step{s}" for s in steps]
    if metric == "reward_judged":
        files = " ".join(f"results/factor/agent/{n}.jsonl" for n in names)
        if shell(f"cd {REPO} && {PY} scripts/factor/judge_eval.py {files}", f"eval-{stage}") != 0:
            stop("the judge failed on val", stage=stage)
    found = {s: measure(f"{stage}-step{s}") for s in steps}
    step = earliest_within({s: m[metric] for s, m in found.items()}, REWARD_MARGIN)
    note(f"chose {stage}", step=step, metric=metric, curve={s: round(m[metric], 3) for s, m in found.items()},
         passes={s: round(m["pass"], 3) for s, m in found.items()}, val=found[step])
    return keep_adapter(stage, f"{broadcasts(stage)}/step_{step}"), found[step]


def wait_for_freeze() -> bool:
    """Online RL needs no held-out ideas, so it goes ahead after a while. Held-out evals don't."""
    flag = f"{REPO}/results/factor/frozen.txt"
    deadline = time.monotonic() + FREEZE_WAIT_HOURS * 3600
    note("waiting for the freeze", flag=flag, hours=FREEZE_WAIT_HOURS)
    while time.monotonic() < deadline:
        if os.path.exists(flag):
            note("frozen", commit=open(flag).read().strip())
            return True
        time.sleep(30)
    note("no freeze in time: held-out evals skipped, online RL goes on")
    return False


def heldout(models: dict[str, str]) -> None:
    """Each stage so far on the held-out ideas, with the judge's view, and the comparisons that exist yet."""
    queue, lock = list(models.items()), threading.Lock()

    def worker(gpus: str, port: int) -> None:
        while True:
            with lock:
                if not queue:
                    return
                stage, adapter = queue.pop(0)
            name = f"{stage}-heldout"
            if measure(name) is None:
                wait_gpus_free(gpus)
                code = shell(f"bash {REPO}/scripts/factor/eval_stage.sh {name} {adapter} heldout {gpus} {port} 128", f"eval-{name}")
                if code != 0 or not measure(name):
                    stop("a held-out eval failed", run=name, exit=code)
            shell(f"cd {REPO} && {PY} scripts/factor/judge_eval.py results/factor/agent/{name}.jsonl", f"eval-{name}")
            note("held out", stage=stage, result=measure(name))

    threads = [threading.Thread(target=worker, args=a) for a in (("0,1", 18200), ("2,3", 18400))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    agent = "results/factor/agent"
    with open(f"{REPO}/results/factor/heldout_comparisons.txt", "a") as out:
        for label, later, earlier in (("H1", "s1-heldout", "v2-olmo-tools"), ("H3", "s2-heldout", "s1-heldout"), ("H4", f"{OFFLINE}-ic-heldout", "s2-heldout")):
            out.write(f"== {label}: {later} against {earlier}\n")
            out.flush()
            subprocess.run([PY, "scripts/factor/compare_stages.py", f"{agent}/{later}.jsonl", f"{agent}/{earlier}.jsonl", "--split", "heldout"], stdout=out, stderr=subprocess.STDOUT, cwd=REPO)
    note("held-out comparisons written", file="results/factor/heldout_comparisons.txt")


def run_steps(metrics: str) -> dict[int, dict[str, float]]:
    keys = {"entropy/all/mean": "ent", "mismatch_kl/all/mean": "kl", "train/agg/all/agent/reward/mean": "reward"}
    steps: dict[int, dict[str, float]] = {}
    if os.path.exists(metrics):
        with open(metrics) as f:
            for row in map(json.loads, f):
                if row.get("step") is not None:
                    steps.setdefault(row["step"], {}).update({short: row[k] for k, short in keys.items() if k in row})
    return steps


def online_once(name: str, start: str, arm: str, rank: int, extra: str) -> tuple[str, str | None]:
    """Returns how it ended, and the run's directory."""
    out = f"{ROOT}/outputs/{name}"
    command = f"bash {REPO}/scripts/factor/train_rl.sh {name} {start} {arm} --trainer.model.lora.rank {rank} {extra}"
    with open(f"{ROOT}/logs/rl-{name}.log", "a") as log:
        process = subprocess.Popen(["bash", "-c", command], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    while process.poll() is None:
        time.sleep(60)
        found = glob.glob(f"{out}/*/metrics.jsonl")
        table = run_steps(found[0]) if found else {}
        trip = guard_trips(table) or reward_falls(table)
        if trip:
            os.killpg(process.pid, signal.SIGTERM)
            time.sleep(60)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            return f"guard: {trip}", os.path.dirname(found[0])
    found = glob.glob(f"{out}/*/metrics.jsonl")
    return ("finished" if process.returncode == 0 else f"exit {process.returncode}"), os.path.dirname(found[0]) if found else None


def pilot(arm: str, start: str, rank: int, steps: int) -> bool:
    name = f"pilot-factor-{arm}"
    shutil.rmtree(f"{ROOT}/outputs/{name}", ignore_errors=True)
    wait_gpus_free("0,1,2,3")
    note("pilot", arm=arm, steps=steps, start=start)
    started = time.monotonic()
    ended, run = online_once(name, start, arm, rank, f"--max-steps {steps} --no-orchestrator.eval")
    if ended != "finished" or not run:
        note("pilot failed", arm=arm, ended=ended, log=f"rl-{name}.log")
        return False
    logged = subprocess.run(["bash", "-c", f"grep -rl 'Started the adapter' {run} | head -1"], capture_output=True, text=True).stdout.strip()
    table = run_steps(f"{run}/metrics.jsonl")
    healthy = bool(logged) and bool(table) and all(
        ENTROPY_BAND[0] <= d.get("ent", 1.0) <= ENTROPY_BAND[1] and d.get("kl", 0.0) <= MISMATCH_KL for s, d in table.items() if s >= 1
    )
    note("pilot done", arm=arm, healthy=healthy, started_from_adapter=bool(logged), minutes=round((time.monotonic() - started) / 60), table=table)
    return healthy


def online(arm: str, seed: int, start: str, rank: int) -> None:
    name = f"factor-s4-{arm}-seed{seed}"
    final = f"{ROOT}/adapters/{name}"
    if os.path.exists(f"{final}/adapter_model.safetensors") and measure(name):
        note("kept", run=name)
        return
    extra, lr_note = f"--inference.vllm.seed {seed}", None
    for attempt in (1, 2):
        wait_gpus_free("0,1,2,3")
        out = f"{ROOT}/outputs/{name}"
        if os.path.exists(out):
            os.rename(out, f"{out}-attempt{attempt}-{int(time.time())}")
        note("online RL", run=name, arm=arm, seed=seed, start=start, attempt=attempt, lr=lr_note)
        started = time.monotonic()
        ended, run = online_once(name, start, arm, rank, extra)
        adapter = f"{run}/broadcasts/step_{ONLINE_STEPS}" if run else ""
        if ended == "finished" and os.path.exists(f"{adapter}/adapter_model.safetensors"):
            note("online RL finished", run=name, hours=round((time.monotonic() - started) / 3600, 2), table=run_steps(f"{run}/metrics.jsonl"))
            break
        note("online RL ended early", run=name, ended=ended, attempt=attempt)
        if attempt == 2:
            stop("online RL failed twice", run=name)
        # Prereg 02's rules: the guard reruns once at the lower rate, a crash once as it was.
        if ended.startswith("guard"):
            extra, lr_note = f"{extra} --trainer.optim.lr 1e-5", "1e-5"
    keep_adapter(f"s4-{arm}-seed{seed}", adapter)
    wait_gpus_free("0,1")
    code = shell(f"bash {REPO}/scripts/factor/eval_stage.sh {name} {final} val 0,1 18200 128", f"eval-{name}")
    if code != 0 or not measure(name):
        stop("evaluating online RL failed", run=name, exit=code)
    if arm == "judged":
        shell(f"cd {REPO} && {PY} scripts/factor/judge_eval.py results/factor/agent/{name}.jsonl", f"eval-{name}")
    note("online RL on val", run=name, val=measure(name))


def main() -> None:
    os.chdir(REPO)
    note("start", pid=os.getpid())
    rank, s1 = choose_s1()

    s2, s2_val = choose_stage("s2", f"{DATA}/s2", rank, s1, "0,1,2,3", 18400, "reward_ic")

    if not os.path.exists(f"{DATA}/{OFFLINE}-judged/train.jsonl"):
        code = shell(f"cd {REPO} && {PY} scripts/factor/make_stage_datasets.py results/factor/teacher/strategies.jsonl --ideas task/ideas_v2.jsonl --offline {DATA} --arms judged", "stage-data-judged")
        if code != 0:
            stop("building arm C's offline data failed")
    note("offline data", rows={arm: rows_in(f"{DATA}/{OFFLINE}-{arm}/train.jsonl") for arm in METRIC})

    queue, starts, lock = ["ic", "gates", "judged"], {}, threading.Lock()

    def worker(gpus: str, port: int) -> None:
        while True:
            with lock:
                if not queue:
                    return
                arm = queue.pop(0)
            adapter, val = choose_stage(f"{OFFLINE}-{arm}", f"{DATA}/{OFFLINE}-{arm}", rank, s2, gpus, port, METRIC[arm], OFFLINE_LR)
            fell = s2_val["pass"] - val["pass"]
            # Prereg 03's stop rule: a stage that loses the skill is reported, and online RL starts before it.
            starts[arm] = s2 if fell > OFFLINE_FALL else adapter
            note("offline RL done", arm=arm, s2_pass=s2_val["pass"], pass_now=val["pass"], starts_online_from=starts[arm])

    threads = [threading.Thread(target=worker, args=a) for a in (("0,1", 18200), ("2,3", 18400))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if wait_for_freeze():
        heldout({"s1": s1, "s2": s2, **{f"{OFFLINE}-{arm}": f"{ROOT}/adapters/factor-{OFFLINE}-{arm}" for arm in METRIC}})

    if not pilot("ic", starts["ic"], rank, 3):
        stop("the online RL pilot failed, so no main run started")
    judged_ok = pilot("judged", starts["judged"], rank, 2)
    for arm, seed in ONLINE_ORDER:
        if arm == "judged" and not judged_ok:
            note("skipped", run=f"{arm}-{seed}", why="the arm C pilot failed")
            continue
        online(arm, seed, starts[arm], rank)
    note("all stages finished")


if __name__ == "__main__":
    if not (ROOT and REPO):
        sys.exit("source setup/env.sh first")
    main()
