"""Recount pilot 03b's three rules for factor-p1-s2-ic and factor-p2-s2-rank from raw files, sharing no code with
pilot_compare.py. Run on GS01: training IC comes from the raw rollout traces under /mnt/nvme1/bruce/outputs, which are
not in the repo. Its bootstrap is its own, so intervals can differ from pilot_compare.py's in the last digit."""
import collections, glob, json, pathlib, re
import numpy as np
O, A = "/mnt/nvme1/bruce/outputs", f"{pathlib.Path(__file__).resolve().parents[2]}/results/factor/agent"

def train_ic(runs, first, last):
    per = collections.defaultdict(list)
    for run in runs:
        for path in glob.glob(f"{O}/{run}/*/rollouts/step_*/train/all/traces.jsonl"):
            step = int(re.search(r"step_(\d+)/train", path).group(1))
            if first <= step <= last:
                for line in open(path):
                    r = json.loads(line); m = r["traces"][0]["metrics"]
                    if m.get("has_answer", 0) > 0 and m.get("reward_gates", 0) > 0 and m.get("ic") is not None:
                        per[r["task"]["data"]["idea_id"]].append(m["ic"])
    return {k: np.mean(v) for k, v in per.items()}

def val(names):
    ic, rate = collections.defaultdict(list), collections.defaultdict(list)
    for n in names:
        for line in open(f"{A}/{n}.jsonl"):
            r = json.loads(line)
            ok = bool(r.get("matches_idea") and r.get("factor"))
            rate[r["idea"]].append(ok)
            if ok and r.get("ic") is not None:
                ic[r["idea"]].append(r["ic"])
    return {k: np.mean(v) for k, v in ic.items()}, {k: np.mean(v) for k, v in rate.items()}

def paired(a, b):
    keys = sorted(set(a) & set(b)); d = np.array([a[k] - b[k] for k in keys])
    boot = d[np.random.default_rng(1).integers(0, len(d), (20000, len(d)))].mean(1)
    return f"{d.mean():+.4f} ({np.quantile(boot, .025):+.4f} to {np.quantile(boot, .975):+.4f}) over {len(keys)} ideas"

base_train = train_ic(["factor-s4-ic-seed1", "factor-s4-ic-seed2"], 26, 35)
base_ic, base_rate = val(["factor-s4-ic-seed1", "factor-s4-ic-seed2"])
for run in ["factor-p1-s2-ic", "factor-p2-s2-rank"]:
    ic, rate = val([run])
    print(run)
    print("  rule 1", paired(train_ic([run], 26, 35), base_train))
    print("  rule 2", paired(ic, base_ic))
    print("  rule 3", paired(rate, base_rate), "| plain pass rates", f"{np.mean(list(rate.values())):.3f} vs {np.mean(list(base_rate.values())):.3f}")
    print("  steps 1-10 to 26-35", paired(train_ic([run], 26, 35), train_ic([run], 1, 10)))
