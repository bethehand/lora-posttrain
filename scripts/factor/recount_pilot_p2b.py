"""Recount pilot 03b's three rules for factor-p2b-s2-rank from raw files, sharing no code with pilot_compare.py.
Rule 3 scores each answer-less episode on the formula it tested best on the research window, on 2025 only. Run on GS01:
training IC comes from the raw rollout traces under /mnt/nvme1/bruce/outputs, which are not in the repo, and the
sandbox reads the dev panel, which ends with 2025."""
import collections, glob, json, pathlib, re, sys
import multiprocessing as mp
import numpy as np
REPO = str(pathlib.Path(__file__).resolve().parents[2])
sys.dont_write_bytecode = True
sys.path.insert(0, f"{REPO}/envs/factor_idea")
from factor_idea import sandbox as sb  # noqa: E402
from factor_idea.panel import load_dev  # noqa: E402
O, A = "/mnt/nvme1/bruce/outputs", f"{REPO}/results/factor/agent"
IDEAS = {r["id"]: r for r in map(json.loads, open(f"{REPO}/task/ideas_v2.jsonl"))}
SEEN = re.compile(r"incremental IC after removing known styles: ([+-]?[0-9.]+) \(t = ")
BOX = sb.Sandbox(load_dev())
B = ["factor-s4-ic-seed1", "factor-s4-ic-seed2"]

def train_ic(runs, first, last):
    per = collections.defaultdict(list)
    for run in runs:
        for path in glob.glob(f"{O}/{run}/*/rollouts/step_*/train/all/traces.jsonl"):
            step = int(re.search(r"step_(\d+)/train", path).group(1))
            if first <= step <= last:
                for line in open(path):
                    r = json.loads(line); m = r["traces"][0]["metrics"]
                    if m["has_answer"] > 0 and m["reward_gates"] > 0 and m.get("ic") is not None:
                        per[r["task"]["data"]["idea_id"]].append(m["ic"])
    return {k: np.mean(v) for k, v in per.items()}

def tested(steps):
    out, waiting = [], []
    for s in steps or []:
        if s.get("role") == "assistant":
            for c in s.get("calls") or []:
                try: waiting.append(json.loads(c).get("expression", ""))
                except (ValueError, AttributeError, TypeError): waiting.append("")
        elif s.get("role") == "tool" and waiting:
            e = waiting.pop(0); m = SEEN.search(s.get("content") or "")
            if m: out.append((e, float(m.group(1))))
    return out

def score(job):
    s = BOX.score(*job[:3], also=job[3])
    return bool(s.matches_idea), (s.ic.incremental if s.ic else None)

def val(names, fallback=False):
    rows = [json.loads(l) for n in names for l in open(f"{A}/{n}.jsonl")]
    rec = [[r["idea"], bool(r.get("factor")), bool(r.get("matches_idea") and r.get("factor")), r.get("ic")] for r in rows]
    stats = collections.Counter()
    if fallback:
        slots, jobs = [], []
        for i, r in enumerate(rows):
            if not r.get("factor"):
                stats["no answer"] += 1
                t = tested(r.get("steps"))
                if t:
                    best = max(t, key=lambda x: x[1])[0]; idea = IDEAS[r["idea"]]
                    slots.append(i); jobs.append((best, idea["key"], idea["direction"], tuple(idea.get("also", ()))))
        stats["no answer, tested"] = len(jobs)
        with mp.get_context("fork").Pool(8) as pool:
            for i, (ok, ic) in zip(slots, pool.map(score, jobs)):
                rec[i][2], rec[i][3] = ok, ic; stats["best test passes"] += ok
    ic, rate = collections.defaultdict(list), collections.defaultdict(list)
    for idea, answered, ok, x in rec:
        rate[idea].append(ok)
        if ok and x is not None and (answered or fallback): ic[idea].append(x)
    return {k: np.mean(v) for k, v in ic.items()}, {k: np.mean(v) for k, v in rate.items()}, stats

def paired(a, b):
    keys = sorted(set(a) & set(b)); d = np.array([a[k] - b[k] for k in keys])
    boot = d[np.random.default_rng(1).integers(0, len(d), (20000, len(d)))].mean(1)
    return f"{d.mean():+.4f} ({np.quantile(boot, .025):+.4f} to {np.quantile(boot, .975):+.4f}) over {len(keys)} ideas"

base_ic, base_rate, _ = val(B)
late = train_ic(["factor-p2b-s2-rank"], 26, 35)
own_ic, own_rate, _ = val(["factor-p2b-s2-rank"])
_, fb_rate, stats = val(["factor-p2b-s2-rank"], fallback=True)
print("factor-p2b-s2-rank")
print("  rule 1", paired(late, train_ic(B, 26, 35)))
print("  rule 2", paired(own_ic, base_ic))
print("  rule 3 with best test", paired(fb_rate, base_rate), "| plain", f"{np.mean(list(fb_rate.values())):.3f} vs {np.mean(list(base_rate.values())):.3f}", dict(stats))
print("  rule 3 own answers only", paired(own_rate, base_rate), "| plain", f"{np.mean(list(own_rate.values())):.3f}")
print("  steps 1-10 to 26-35", paired(late, train_ic(["factor-p2b-s2-rank"], 1, 10)))
print("  minus p1 on rule 1", paired(late, train_ic(["factor-p1-s2-ic"], 26, 35)))
