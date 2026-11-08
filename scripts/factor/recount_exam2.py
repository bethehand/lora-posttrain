"""Recount prereg 04's E1 to E4 from the answer files (2025) and the 2026 files, sharing no code with second_exam.py
or what it imports. Its bootstrap and keep-one draws are its own, so CIs and E3 may differ in the last digit.
DRY=1 runs it on prereg 03's stored held-out answers and 2026 scores instead, as a check of the recount itself."""
import collections, json, os, pathlib, re

import numpy as np

REPO = str(pathlib.Path(__file__).resolve().parents[2])
AG, EX = f"{REPO}/results/factor/agent", f"{REPO}/results/factor/exam2"
MODELS = {"S1": ["s1"], "S2": ["s2"], "S3 B": ["s3p-ic"], "S4 A": ["s4-gates-seed1", "s4-gates-seed2"],
          "S4 B": ["s4-ic-seed1", "s4-ic-seed2"], "S4 C": ["s4-judged-seed1", "s4-judged-seed2"], "DeepSeek": ["deepseek"]}
SEEN = re.compile(r"incremental IC after removing known styles: ([+-]?[0-9.]+) \(t = ")
COVER = "too_few_coins_or_constant_when_scored"
IDEAS = [json.loads(l)["id"] for l in open(f"{REPO}/task/ideas_fresh.jsonl")]
PAIRS = {m: [(f"{AG}/exam2-{n}.jsonl", f"{EX}/{n}.jsonl") for n in names] for m, names in MODELS.items()}
DRY = os.environ.get("DRY") == "1"
if DRY:  # prereg 03's stored answers and 2026 scores; S2 wasn't scored on 2026 then, so S1 stands in for it. Opens no panel
    S = f"{REPO}/results/factor/sealed"
    arm = lambda x: [(f"{AG}/factor-s4-{x}-seed{s}-heldout.jsonl", f"{S}/{x}-seed{s}.jsonl") for s in (1, 2)]
    PAIRS = {"S2": [(f"{AG}/s1-heldout.jsonl", f"{S}/s1.jsonl")], "S4 A": arm("gates"), "S4 B": arm("ic"), "S4 C": arm("judged"),
             "DeepSeek": [(f"{S}/inputs-deepseek-heldout.jsonl", f"{S}/deepseek.jsonl")]}
    IDEAS = [r["id"] for r in map(json.loads, open(f"{REPO}/task/ideas_v2.jsonl")) if r["split"] == "heldout"]
squash = lambda s: "".join((s or "").split())

def load(answers, scored):
    a = [json.loads(l) for l in open(answers)]
    b = [json.loads(l) for l in open(scored)]
    name = answers
    assert len(a) == len(b), name
    rows = []
    for x, y in zip(a, b):
        assert x["idea"] == y["idea"] and x.get("factor") == y.get("factor"), name
        f, research, waiting = x.get("factor"), None, []
        for s in x.get("steps") or []:
            if s.get("role") == "assistant":
                for c in s.get("calls") or []:
                    try: waiting.append(json.loads(c).get("expression", ""))
                    except (ValueError, AttributeError, TypeError): waiting.append("")
            elif s.get("role") == "tool" and waiting:
                e = waiting.pop(0); m = SEEN.search(s.get("content") or "")
                if m and f and squash(e) == squash(f): research = float(m.group(1))
        gates = bool(f) and not x.get("over_limit") and (bool(x.get("matches_idea")) or x.get("reason") == COVER)
        p25, p26 = bool(x.get("matches_idea")), bool(y.get("matches_idea"))
        rows.append(dict(idea=x["idea"], answered=bool(f), gates=gates, research=research, p25=p25, p26=p26,
                         ic25=x.get("ic") if p25 else None, ic26=y.get("ic") if p26 else None))
    return rows

def per(rows, key):
    v = collections.defaultdict(list)
    for r in rows:
        if r is None: continue
        if r[key] is not None: v[r["idea"]].append(float(r[key]))
    return {k: np.mean(x) for k, x in v.items()}

def rate(rows, key="p26"):
    v = collections.defaultdict(list)
    for idea, r in rows:
        v[idea].append(bool(r and r[key]))
    return {k: np.mean(x) for k, x in v.items()}

def paired(a, b, seed=7):
    keys = sorted(set(a) & set(b)); d = np.array([a[k] - b[k] for k in keys])
    m = d[np.random.default_rng(seed).integers(0, len(d), (10000, len(d)))].mean(1)
    return d.mean(), np.quantile(m, .025), np.quantile(m, .975), len(keys)

def keep(rows, k, seed=3, draws=2000):
    rng, by = np.random.default_rng(seed), collections.defaultdict(list)
    for r in rows: by[r["idea"]].append(r)
    out = []
    for idea in sorted(by):
        m = by[idea]
        subsets = [[r] for r in m] if k == 1 else ([m] if k >= len(m) else [[m[i] for i in rng.choice(len(m), k, replace=False)] for _ in range(draws)])
        for s in subsets:
            ok = [r for r in s if r["research"] is not None and r["gates"]]
            if ok: out.append((idea, max(ok, key=lambda r: r["research"])))
            else:
                ans = [r for r in s if r["answered"]]
                out.append((idea, ans[rng.integers(len(ans))] if ans else None))
    return out

def within(models, tag="26"):
    sums = collections.defaultdict(lambda: np.zeros(3)); pairs = 0
    for rows in models.values():
        g = collections.defaultdict(list)
        for r in rows:
            if r["p" + tag] and r["research"] is not None and r["ic" + tag] is not None: g[r["idea"]].append((r["research"], r["ic" + tag]))
        for idea, pts in g.items():
            if len(pts) < 2: continue
            c = np.array(pts) - np.array(pts).mean(0); pairs += len(pts)
            sums[idea] += [c[:, 0] @ c[:, 0], c[:, 1] @ c[:, 1], c[:, 0] @ c[:, 1]]
    keys = sorted(sums); t = np.array([sums[k] for k in keys])
    d = t[np.random.default_rng(11).integers(0, len(keys), (10000, len(keys)))].sum(1)
    boot = d[:, 2] / np.sqrt(d[:, 0] * d[:, 1]); tot = t.sum(0)
    return tot[2] / np.sqrt(tot[0] * tot[1]), np.quantile(boot, .025), np.quantile(boot, .975), pairs

models = {m: [r for a, s in files for r in load(a, s)] for m, files in PAIRS.items()}
for m, rows in models.items():
    counts = collections.Counter(r["idea"] for r in rows)
    assert set(counts) == set(IDEAS) and (DRY or set(counts.values()) == {16}), (m, counts)
    print(f"{m:9s} pass26 {np.mean(list(rate([(r['idea'], r) for r in rows]).values())):.3f} IC26 {np.mean(list(per(rows, 'ic26').values())):+.4f} "
          f"pass25 {np.mean(list(rate([(r['idea'], r) for r in rows], 'p25').values())):.3f} IC25 {np.mean(list(per(rows, 'ic25').values())):+.4f}")
fmt = lambda t, p=False: (f"{t[0]:+.1%} ({t[1]:+.1%} to {t[2]:+.1%})" if p else f"{t[0]:+.4f} ({t[1]:+.4f} to {t[2]:+.4f})") + f" over {t[3]} ideas"
s2, b = models["S2"], models["S4 B"]
e1 = paired(per(s2, "ic26"), per(b, "ic26")); print("E1 S2 - S4 B, IC 2026:", fmt(e1), "PASSES" if e1[1] > 0 else "does not pass")
e2 = paired(rate([(r["idea"], r) for r in b]), rate([(r["idea"], r) for r in s2])); print("E2 S4 B - S2, pass 2026:", fmt(e2, True), "PASSES" if e2[1] > 0 else "does not pass")
k4, k1 = keep(b, 4), keep(b, 1)
e3 = paired(per([r for _, r in k4], "ic26"), per([r for _, r in k1], "ic26")); print("E3 S4 B k=4 - k=1, IC 2026:", fmt(e3), "PASSES" if e3[1] > 0 else "does not pass",
    f"| k=4 pass {np.mean(list(rate(k4).values())):.3f}, k=1 pass {np.mean(list(rate(k1).values())):.3f}")
e4 = within(models); print(f"E4 within correlation 2026: {e4[0]:+.3f} ({e4[1]:+.3f} to {e4[2]:+.3f}) over {e4[3]} passes", "PASSES" if e4[1] > 0.3 else "does not pass")
e1b = paired(per(s2, "ic25"), per(b, "ic25")); print("2025 beside: E1", fmt(e1b), "| E4", "%+.3f" % within(models, "25")[0])
