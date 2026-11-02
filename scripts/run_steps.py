"""Per-step summary of a prime-rl run and the prereg 01 instability guard (steps 1-10).

    python3 scripts/run_steps.py <run dir>/metrics.jsonl
"""

import json
import sys

KEYS = [("train/agg/all/agent/reward/mean", "reward"), ("train/agg/all/agent/is_truncated/mean", "trunc"),
        ("train/agg/all/agent/num_turns/mean", "turns"), ("entropy/all/mean", "ent"), ("mismatch_kl/all/mean", "kl"),
        ("perf/peak_memory", "mem"), ("time/forward_backward", "fb")]
steps = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r.get("step") is None:
        continue
    d = steps.setdefault(r["step"], {})
    d.update({short: r[k] for k, short in KEYS if k in r})
    if r.get("producer") == "orch" and "time/step" in r:
        d["orch_s"] = r["time/step"]
for s in sorted(steps):
    print(f"step {s:>2}: " + " ".join(f"{k}={v:.4g}" for k, v in steps[s].items()))
done = [s for s in sorted(steps) if 1 <= s <= 10 and "ent" in steps[s] and "kl" in steps[s]]
kl_bad = {s: steps[s]["kl"] > 0.01 for s in done}
ent_bad = {s: not 0.02 <= steps[s]["ent"] <= 0.5 for s in done}
trig = [s for s in done if (kl_bad[s] and kl_bad.get(s + 1)) or (ent_bad[s] and ent_bad.get(s + 1))]
if trig:
    print("GUARD TRIGGER at steps", trig)
elif 10 in done:
    print("GUARD PASSED for steps 1-10")
else:
    print(f"guard: {len(done)} of 10 steps so far")
