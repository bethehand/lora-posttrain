#!/usr/bin/env python3
"""What prereg 03's answers and rollouts show once its exam is over. Exploration, and reported as such.

Reads stored answers, stored 2026 scores and the rollouts under $ROOT/outputs. The sealed panel is never opened.
`pick` and `push` score tested formulas again, on 2025 only.

    python scripts/factor/explore_after_exam.py reward     # how much of the reward's spread is IC; IC on training ideas during RL
    python scripts/factor/explore_after_exam.py student    # student against teacher, search within an episode, failures
    python scripts/factor/explore_after_exam.py persist    # does research IC predict later IC within an idea?
    python scripts/factor/explore_after_exam.py pick       # keep one of k answers by research IC
    python scripts/factor/explore_after_exam.py push       # what each reward says about the number of tests
    python scripts/factor/explore_after_exam.py features   # what a formula's form says about its IC within an idea
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import multiprocessing as mp
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "envs", "factor_idea"))

from compare_stages import interval  # noqa: E402

ROOT = os.environ.get("ROOT", "")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
AGENT = f"{REPO}/results/factor/agent"
SEALED = f"{REPO}/results/factor/sealed"
REPORTED_IC = re.compile(r"incremental IC after removing known styles: ([+-]?[0-9.]+) \(t = ")
ARMS = {"S4 A": "gates", "S4 B": "ic", "S4 C": "judged"}
FILES = {
    "val": {
        "S1": ["s1-r32-step350"],
        "S2": ["s2-step35"],
        "S3 A": ["s3p-gates-step16"],
        "S3 B": ["s3p-ic-step64"],
        "S3 C": ["s3p-judged-step34"],
        **{m: [f"factor-s4-{a}-seed{s}" for s in (1, 2)] for m, a in ARMS.items()},
        "DeepSeek": ["v2-deepseek-tools"],
    },
    "heldout": {
        "S1": ["s1-heldout"],
        "S2": ["s2-heldout"],
        "S3 A": ["s3p-gates-heldout"],
        "S3 B": ["s3p-ic-heldout"],
        "S3 C": ["s3p-judged-heldout"],
        **{m: [f"factor-s4-{a}-seed{s}-heldout" for s in (1, 2)] for m, a in ARMS.items()},
        "DeepSeek": ["v2-deepseek-tools"],
    },
}
EXAM = {"S1": ["s1"], **{m: [f"{a}-seed{s}" for s in (1, 2)] for m, a in ARMS.items()}, "DeepSeek": ["deepseek"]}


# Kept pure so it can be tested.
def norm(text: str | None) -> str:
    return re.sub(r"\s+", "", text or "")


def trials(steps: list[dict] | None) -> list[tuple[str, float]]:
    """The tests that computed, in order: the expression and the research IC the student saw."""
    out, pending = [], []
    for step in steps or []:
        if step.get("role") == "assistant":
            for call in step.get("calls") or []:
                try:
                    pending.append(json.loads(call).get("expression", ""))
                except (ValueError, AttributeError, TypeError):
                    pending.append("")
        elif step.get("role") == "tool" and pending:
            expression = pending.pop(0)
            found = REPORTED_IC.search(step.get("content") or "")
            if found:
                out.append((expression, float(found.group(1))))
    return out


def answer_research_ic(factor: str | None, tested: list[tuple[str, float]]) -> float | None:
    """The research IC the answer showed in a test, or None if it was never tested."""
    found = [ic for expression, ic in tested if norm(expression) == norm(factor)]
    return found[-1] if found else None


def ic_share(groups: list[tuple[list[float], list[bool]]]) -> float:
    """Of the spread of reward within groups, the share that lies among passes, since a failure always earns 0."""
    total, among = 0.0, 0.0
    for rewards, passed in groups:
        r, p = np.asarray(rewards, float), np.asarray(passed, bool)
        total += float(r.var())
        if p.any():
            among += float(((r[p] - r[p].mean()) ** 2).sum()) / len(r)
    return among / total if total else float("nan")


def percentile(table: np.ndarray, ic: float) -> float:
    return float(np.searchsorted(table, ic, side="right")) / len(table) if len(table) else 0.5


def pick_one(rollouts: list[dict], rule: str, rng: np.random.Generator) -> dict:
    """The rollout a process keeps. `research` sees only reported IC, `gates` also the gates, which read data through 2024."""
    tested = [r for r in rollouts if r["research"] is not None]
    if rule == "gates":
        tested = [r for r in tested if r["passed"]]
    elif rule == "gates+judge":
        passing = [r for r in tested if r["passed"]]
        tested = [r for r in passing if r["faithful"]] or passing
    if tested:
        return max(tested, key=lambda r: r["research"])
    answered = [r for r in rollouts if r["answered"]]
    return answered[rng.integers(len(answered))] if answered else rollouts[0]


def within_slope(pairs: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, float]:
    """y per unit of x within groups, with whole groups resampled."""
    keys = sorted(pairs)
    xy = np.array([float(pairs[k][0] @ pairs[k][1]) for k in keys])
    xx = np.array([float(pairs[k][0] @ pairs[k][0]) for k in keys])
    draws = np.random.default_rng(0).integers(0, len(keys), size=(10_000, len(keys)))
    slopes = xy[draws].sum(axis=1) / np.maximum(xx[draws].sum(axis=1), 1e-12)
    return float(xy.sum() / xx.sum()), float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


IDEAS = {r["id"]: r for r in read_jsonl(f"{REPO}/task/ideas_v2.jsonl")}


def ci(values) -> str:
    mean, low, high = interval(list(values))
    return f"{mean:+.4f} ({low:+.4f} to {high:+.4f})"


def answers(model: str, split: str) -> list[dict]:
    """A model's answers on a split, each with what it showed in its tests and, for the exam's files, its 2026 score."""
    out = []
    for k, name in enumerate(FILES[split][model]):
        rows = [r for r in read_jsonl(f"{AGENT}/{name}.jsonl") if IDEAS.get(r.get("idea"), {}).get("split") == split]
        exam = read_jsonl(f"{SEALED}/{EXAM[model][k]}.jsonl") if split == "heldout" and model in EXAM else [{}] * len(rows)
        assert len(exam) == len(rows), name
        for row, later in zip(rows, exam):
            tested = trials(row.get("steps"))
            out.append(
                {
                    **row,
                    "seed": k,
                    "tested": tested,
                    "research": answer_research_ic(row.get("factor"), tested),
                    "answered": bool(row.get("factor")),
                    "passed": bool(row.get("matches_idea") and row.get("factor")),
                    "ic25": row.get("ic"),
                    "ic26": later.get("ic"),
                    "reward25": row.get("reward_ic") or 0.0,
                    "reward26": later.get("reward_ic") or 0.0,
                }
            )
    return out


def by_idea(rows: list[dict]) -> dict[str, list[dict]]:
    out = collections.defaultdict(list)
    for row in rows:
        out[row["idea"]].append(row)
    return out


def rollouts(arm: str, seed: int, first: int = 1, last: int = 99, with_tests: bool = False) -> list[dict]:
    return run_rollouts(f"factor-s4-{arm}-seed{seed}", first, last, with_tests)


def run_rollouts(run: str, first: int = 1, last: int = 99, with_tests: bool = False) -> list[dict]:
    """Every finished rollout of a run on training ideas."""
    out = []
    for path in glob.glob(f"{ROOT}/outputs/{run}/*/rollouts/step_*/train/all/traces.jsonl"):
        step = int(re.search(r"step_(\d+)", path).group(1))
        if not first <= step <= last:
            continue
        for row in read_jsonl(path):
            trace = row["traces"][0]
            metrics = trace["metrics"]
            item = {
                "step": step,
                "idea": row["task"]["data"]["idea_id"],
                "group": (run, step, row["group"]["id"]),
                "passed": metrics["reward_gates"] > 0,
                "ic": metrics.get("ic"),
                "reward": metrics["reward_ic"],
                "tests": int(metrics["num_tests"]),
                "answered": metrics["has_answer"] > 0,
                "expression": ((trace.get("info") or {}).get("factor") or {}).get("expression"),
            }
            if with_tests:
                item["tested"] = trials(trace_steps(trace))
            out.append(item)
    return out


def trace_steps(trace: dict) -> list[dict]:
    steps = []
    for node in trace.get("nodes") or []:
        message = node.get("message") or {}
        if message.get("role") == "assistant":
            calls = [c.get("arguments") if isinstance(c.get("arguments"), str) else json.dumps(c.get("arguments")) for c in message.get("tool_calls") or []]
            steps.append({"role": "assistant", "calls": calls})
        elif message.get("role") == "tool":
            steps.append({"role": "tool", "content": message.get("content") or ""})
    return steps


def groups_of(rows: list[dict]) -> list[list[dict]]:
    out = collections.defaultdict(list)
    for row in rows:
        out[row["group"]].append(row)
    return [members for members in out.values() if len(members) >= 4]


def show_reward() -> None:
    store = {arm: [r for seed in (1, 2) for r in rollouts(arm, seed)] for arm in ARMS.values()}

    def change(rows):
        early, late = collections.defaultdict(list), collections.defaultdict(list)
        for r in rows:
            if r["passed"] and r["ic"] is not None:
                if 1 <= r["step"] <= 10:
                    early[r["idea"]].append(r["ic"])
                elif 26 <= r["step"] <= 35:
                    late[r["idea"]].append(r["ic"])
        shared = sorted(set(early) & set(late))
        return {i: np.mean(late[i]) - np.mean(early[i]) for i in shared}, np.mean([np.mean(early[i]) for i in shared]), np.mean([np.mean(late[i]) for i in shared])

    print("IC among passes on training ideas, steps 1-10 to steps 26-35, per idea")
    changes = {}
    for arm, rows in store.items():
        changes[arm], before, after = change(rows)
        print(f"  arm {arm:7s} {before:+.4f} to {after:+.4f}, change {ci(changes[arm].values())} over {len(changes[arm])} ideas")
    shared = sorted(set(changes["ic"]) & set(changes["gates"]))
    print(f"  arm B's change minus arm A's {ci(changes['ic'][i] - changes['gates'][i] for i in shared)}")

    table = collections.defaultdict(list)
    for arm in ("gates", "judged"):
        for r in store[arm]:
            if r["passed"] and r["ic"] is not None:
                table[r["idea"]].append(r["ic"])
    table = {k: np.sort(v) for k, v in table.items()}

    def linear(r, _):
        return 0.3 + 0.7 * float(np.clip((r["ic"] + 0.03) / 0.06, 0, 1))

    def rank_in_group(floor):
        def score(r, members):
            ics = sorted(x["ic"] for x in members if x["passed"] and x["ic"] is not None)
            return floor + (1 - floor) * (ics.index(r["ic"]) / (len(ics) - 1) if len(ics) > 1 else 0.5)

        return score

    schemes = {"arm B's": linear, **{f"table, floor {f}": (lambda r, _, f=f: f + (1 - f) * percentile(table[r["idea"]], r["ic"])) for f in (0.3, 0.2, 0.1)}}
    schemes.update({f"rank within group, floor {f}": rank_in_group(f) for f in (0.3, 0.1)})
    print("Share of arm B's within-group spread that is IC among passes, under its reward and others, with a table from arms A and C")
    for first, last in ((1, 5), (31, 35)):
        members = groups_of([r for r in store["ic"] if first <= r["step"] <= last])
        for name, score in schemes.items():
            groups = []
            for rows in members:
                ok = [r["passed"] and r["ic"] is not None for r in rows]
                groups.append(([score(r, rows) if good else 0.0 for r, good in zip(rows, ok)], ok))
            print(f"  steps {first}-{last}, {name:32s} {ic_share(groups):6.1%}")
    spread = [np.std([r["ic"] for r in rows if r["passed"] and r["ic"] is not None]) for rows in groups_of(store["ic"]) if sum(r["passed"] and r["ic"] is not None for r in rows) >= 2]
    print(f"Typical spread of IC among passes in a group {np.median(spread):.4f}, which is {0.7 * np.median(spread) / 0.06:.3f} of arm B's reward")


def show_student() -> None:
    teacher = {"val": by_idea(answers("DeepSeek", "val")), "heldout": by_idea(answers("DeepSeek", "heldout"))}

    def compare(label, per, other):
        ic_of = lambda rows: [r["ic25"] for r in rows if r["passed"] and r["ic25"] is not None]  # noqa: E731
        ids = sorted(set(per) & set(other))
        both = [i for i in ids if ic_of(per[i]) and ic_of(other[i])]
        print(
            f"  {label:24s} pass {np.mean([np.mean([r['passed'] for r in per[i]]) for i in ids]):6.1%} against {np.mean([np.mean([r['passed'] for r in other[i]]) for i in ids]):6.1%}; "
            f"IC {np.mean([np.mean(ic_of(per[i])) for i in both]):+.4f} against {np.mean([np.mean(ic_of(other[i])) for i in both]):+.4f}, minus the teacher {ci(np.mean(ic_of(per[i])) - np.mean(ic_of(other[i])) for i in both)} over {len(both)} ideas"
        )

    print("Student against teacher, same ideas, 2025, per idea")
    teacher_train = collections.defaultdict(list)
    for row in read_jsonl(f"{AGENT}/v2-deepseek-tools.jsonl"):
        if IDEAS.get(row.get("idea"), {}).get("split") == "train":
            teacher_train[row["idea"]].append({"passed": bool(row.get("matches_idea") and row.get("factor")), "ic25": row.get("ic")})
    print(" Training ideas, rollouts at temperature 1.0 against the teacher's")
    for model, arm in (("B", "ic"), ("A", "gates")):
        for label, first, last in ((f"S3 {model}, steps 1-5", 1, 5), (f"S4 {model}, steps 31-35", 31, 35)):
            rows = [{"idea": r["idea"], "passed": r["passed"], "ic25": r["ic"]} for seed in (1, 2) for r in rollouts(arm, seed, first, last)]
            compare(label, by_idea(rows), teacher_train)
    for split in ("val", "heldout"):
        print(f" {split}")
        for model in ("S1", "S2", "S3 B", "S4 B"):
            compare(model, by_idea(answers(model, split)), teacher[split])

    print("Search within an episode, passing answers that were tested: research IC of the first test and the answer")
    for split in ("val", "heldout"):
        print(f" {split}")
        for model in FILES[split]:
            rows = [r for r in answers(model, split) if r["passed"] and r["research"] is not None and r["tested"]]
            first = [r["tested"][0][1] for r in rows]
            final = [r["research"] for r in rows]
            distinct = [len({norm(e) for e, _ in r["tested"]}) for r in rows]
            best = [r["research"] >= max(ic for _, ic in r["tested"]) - 1e-9 for r in rows]
            gain = np.mean(final) - np.mean(first)
            print(f"  {model:9s} first {np.mean(first):+.4f} answer {np.mean(final):+.4f} gain {gain:+.4f}; distinct tests {np.mean(distinct):.2f}, gain per extra test {gain / max(1e-9, np.mean(distinct) - 1):+.4f}; answer is its best test {np.mean(best):.0%}")

    def bucket(row):
        reason = row.get("reason") or ""
        if row["passed"]:
            return "pass"
        if not row["answered"]:
            return "no answer"
        if "against" in reason:
            return "wrong direction"
        if "does_not_respond" in reason:
            return "does not respond"
        if reason.startswith("parse") or "too_few" in reason or "too_slow" in reason or "computes" in reason:
            return "does not parse or compute"
        return "other gate"

    print("How episodes end")
    for split in ("val", "heldout"):
        print(f" {split}")
        for model in ("S1", "S2", "S3 B", "S4 B", "DeepSeek"):
            rows = answers(model, split)
            counts = collections.Counter(bucket(r) for r in rows)
            print(f"  {model:9s} " + "  ".join(f"{k} {counts[k] / len(rows):.0%}" for k in ("pass", "no answer", "wrong direction", "does not respond", "does not parse or compute", "other gate")))

    print("Within an idea and a model, IC among passes per extra test, 2025, held-out ideas")
    groups = {"S1, S2, S3": ["S1", "S2", "S3 A", "S3 B", "S3 C"], "S4": list(ARMS), "DeepSeek": ["DeepSeek"]}
    for label, models in groups.items():
        pairs = {}
        for model in models:
            for idea, rows in by_idea([r for r in answers(model, "heldout") if r["passed"] and r["ic25"] is not None]).items():
                if len(rows) >= 2:
                    x = np.array([float(r.get("tests") or 0) for r in rows])
                    y = np.array([r["ic25"] for r in rows])
                    pairs[(model, idea)] = (x - x.mean(), y - y.mean())
        by_group = collections.defaultdict(lambda: [np.zeros(0), np.zeros(0)])
        for (_, idea), (x, y) in pairs.items():
            by_group[idea][0] = np.concatenate([by_group[idea][0], x])
            by_group[idea][1] = np.concatenate([by_group[idea][1], y])
        mean, low, high = within_slope({k: (v[0], v[1]) for k, v in by_group.items()})
        print(f"  {label:12s} {mean:+.5f} ({low:+.5f} to {high:+.5f}) over {len(by_group)} ideas")


def show_persist() -> None:
    print("Passing answers that were tested: research IC against 2025 and 2026, per episode and within an idea")
    for model in FILES["heldout"]:
        rows = [r for r in answers(model, "heldout") if r["passed"] and r["research"] is not None and r["ic25"] is not None]
        line = f"  {model:9s} n {len(rows):3d} research {np.mean([r['research'] for r in rows]):+.4f} 2025 {np.mean([r['ic25'] for r in rows]):+.4f}"
        has_exam = rows[0]["ic26"] is not None
        if has_exam:
            line += f" 2026 {np.mean([r['ic26'] for r in rows]):+.4f}"
        x, y25, y26 = [], [], []
        for members in by_idea(rows).values():
            if len(members) >= 2:
                a = np.array([r["research"] for r in members])
                x += list(a - a.mean())
                b = np.array([r["ic25"] for r in members])
                y25 += list(b - b.mean())
                if has_exam:
                    c = np.array([r["ic26"] for r in members], float)
                    y26 += list(c - c.mean())
        line += f"; within an idea, correlation with 2025 {np.corrcoef(x, y25)[0, 1]:+.2f}"
        if has_exam:
            line += f", with 2026 {np.corrcoef(x, y26)[0, 1]:+.2f}"
        print(line)


def keep(per: dict[str, list[dict]], k: int, rule: str, draws: int = 2000) -> dict[str, dict]:
    rng = np.random.default_rng(0)
    out = {}
    for idea, rows in per.items():
        n = len(rows)
        if k == 1:
            subsets = [[r] for r in rows]
        elif k >= n:
            subsets = [rows]
        else:
            subsets = [[rows[i] for i in rng.choice(n, k, replace=False)] for _ in range(draws)]
        chosen, tests = [], []
        for subset in subsets:
            chosen.append(pick_one(subset, rule, rng))
            tests.append(sum(r.get("tests") or 0 for r in subset))
        passes = [r for r in chosen if r["passed"]]
        judged = [r["faithful"] for r in passes if r.get("faithful") is not None]
        later = [r["ic26"] for r in passes if r["ic26"] is not None]
        out[idea] = {
            "passed": np.mean([r["passed"] for r in chosen]),
            "reward25": np.mean([r["reward25"] for r in chosen]),
            "reward26": np.mean([r["reward26"] for r in chosen]),
            "ic25": np.mean([r["ic25"] for r in passes]) if passes else None,
            "ic26": np.mean(later) if later else None,
            "faithful": np.mean(judged) if judged else None,
            "tests": np.mean(tests),
        }
    return out


def average(result: dict[str, dict], name: str) -> float:
    values = [v[name] for v in result.values() if v[name] is not None]
    return float(np.mean(values)) if values else float("nan")


def paired(a: dict[str, dict], b: dict[str, dict], name: str) -> str:
    shared = [i for i in a if a[i][name] is not None and b[i][name] is not None]
    return f"{ci(a[i][name] - b[i][name] for i in shared)} over {len(shared)} ideas"


def show_keep(label: str, result: dict[str, dict], base: dict[str, dict], exam: bool) -> None:
    line = f"  {label:24s} tests {average(result, 'tests'):4.1f}, pass {average(result, 'passed'):6.1%}, faithful {average(result, 'faithful'):4.0%}, IC 2025 {average(result, 'ic25'):+.4f}, gain {paired(result, base, 'ic25')}, reward 2025 {average(result, 'reward25'):.3f}"
    if exam:
        own = [v["ic26"] for v in result.values() if v["ic26"] is not None]
        line += f"; IC 2026 {ci(own)}, gain {paired(result, base, 'ic26')}, reward 2026 {average(result, 'reward26'):.3f}"
    print(line)


def score_formula(job: tuple) -> tuple[bool, float | None, float]:
    text, key, direction, also = job
    result = _BOX.score(text, key, direction, also=also)
    return bool(result.matches_idea), (result.ic.incremental if result.ic else None), result.reward("ic")


def score_all(jobs: list[tuple]) -> list[tuple]:
    global _BOX
    from factor_idea import sandbox as sb
    from factor_idea.panel import load_dev

    _BOX = sb.Sandbox(load_dev())
    with mp.get_context("fork").Pool(10) as pool:
        return pool.map(score_formula, jobs, chunksize=4)


def job_for(idea_id: str, expression: str) -> tuple:
    idea = IDEAS[idea_id]
    return expression, idea["key"], idea["direction"], tuple(idea.get("also", ()))


def show_pick() -> None:
    print("Held-out ideas. One of k answers is kept by its research IC")
    for model, ks, rules in (("S4 B", (2, 3, 4, 8), ("research", "gates", "gates+judge")), ("S4 A", (2, 4, 8), ("research", "gates", "gates+judge")), ("S4 C", (2, 4, 8), ("research", "gates", "gates+judge")), ("S1", (2, 4), ("research", "gates")), ("S2", (2, 4), ("research", "gates")), ("DeepSeek", (2, 4), ("research", "gates"))):
        per = by_idea(answers(model, "heldout"))
        exam = model in EXAM
        base = keep(per, 1, "research")
        print(model)
        show_keep("one episode", base, base, exam)
        for rule in rules:
            for k in ks:
                show_keep(f"{rule}, k={k}", keep(per, k, rule), base, exam)

    student, teacher = by_idea(answers("S4 B", "heldout")), by_idea(answers("DeepSeek", "heldout"))
    print("S4 arm B against the teacher, by the number of tests each spends, 2026")
    for label, a, b in (("one episode each", keep(student, 1, "research"), keep(teacher, 1, "research")), ("about 10 tests, gates: k=4 and k=2", keep(student, 4, "gates"), keep(teacher, 2, "gates")), ("about 20 tests, gates: k=8 and k=4", keep(student, 8, "gates"), keep(teacher, 4, "gates"))):
        print(f"  {label:36s} reward {ci(a[i]['reward26'] - b[i]['reward26'] for i in a)}, IC {paired(a, b, 'ic26')}, pass {ci(a[i]['passed'] - b[i]['passed'] for i in a)}")

    print("Each seed of S4 arm B alone, k=4, research only, 2026")
    for seed in (0, 1):
        per = by_idea([r for r in answers("S4 B", "heldout") if r["seed"] == seed])
        print(f"  seed {seed + 1}: gain {paired(keep(per, 4, 'research'), keep(per, 1, 'research'), 'ic26')}")

    print("Val ideas, 2025. Research only and gates, k=8 for S4, k=2 and k=4 for S2 and the teacher")
    for model, ks in (("S4 A", (8,)), ("S4 B", (2, 4, 8)), ("S4 C", (8,)), ("S2", (2, 4)), ("DeepSeek", (2, 4))):
        per = by_idea(answers(model, "val"))
        base = keep(per, 1, "research")
        print(model)
        show_keep("one episode", base, base, False)
        for k in ks:
            show_keep(f"research, k={k}", keep(per, k, "research"), base, False)
            show_keep(f"gates, k={k}", keep(per, k, "gates"), base, False)

    print("The same tests and no more episodes: hand in the formula an episode tested best, scored on 2025, held-out ideas")
    for model in ("S1", "S2", "S3 B", "S4 A", "S4 B", "S4 C", "DeepSeek"):
        rows = answers(model, "heldout")
        slots = [i for i, r in enumerate(rows) if r["tested"]]
        scored = dict(zip(slots, score_all([job_for(rows[i]["idea"], max(rows[i]["tested"], key=lambda t: t[1])[0]) for i in slots])))
        own, fallback, best = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
        for i, row in enumerate(rows):
            mine = (row["passed"], row["ic25"], row["reward25"])
            found = scored.get(i, (False, None, 0.0))
            own[row["idea"]].append(mine)
            fallback[row["idea"]].append(mine if row["answered"] else found)
            best[row["idea"]].append(found)

        def summary(table):
            ic = {i: [v[1] for v in vals if v[0] and v[1] is not None] for i, vals in table.items()}
            return np.mean([np.mean([v[0] for v in vals]) for vals in table.values()]), np.mean([np.mean([v[2] for v in vals]) for vals in table.values()]), {i: np.mean(v) for i, v in ic.items() if v}

        a, b, c = summary(own), summary(fallback), summary(best)
        shared = sorted(set(a[2]) & set(c[2]))
        print(f"  {model:9s} own: pass {a[0]:.1%}, reward {a[1]:.3f}, IC {np.mean(list(a[2].values())):+.4f} | best test when no answer: pass {b[0]:.1%}, reward {b[1]:.3f}, IC {np.mean(list(b[2].values())):+.4f} | always best test: pass {c[0]:.1%}, reward {c[1]:.3f}, IC {np.mean(list(c[2].values())):+.4f}, IC gain {ci(c[2][i] - a[2][i] for i in shared)}")


def show_push() -> None:
    table = collections.defaultdict(list)
    early = []
    for arm in ARMS.values():
        for seed in (1, 2):
            for r in rollouts(arm, seed):
                if r["passed"] and r["ic"] is not None:
                    table[r["idea"]].append(r["ic"])
            early += rollouts(arm, seed, 1, 5, with_tests=True)
    table = {k: np.sort(v) for k, v in table.items()}
    slots = [i for i, r in enumerate(early) if not r["answered"] and r["tested"]]
    scored = score_all([job_for(early[i]["idea"], max(early[i]["tested"], key=lambda t: t[1])[0]) for i in slots])
    for i, (passed, ic, _) in zip(slots, scored):
        early[i]["fallback"] = (passed, ic)
    print(f"Steps 1-5: {len(early)} rollouts, {sum(not r['answered'] for r in early)} without an answer, {len(slots)} of those tested something, and {sum(s[0] for s in scored)} of those formulas pass")

    def outcome(r, fallback):
        return r["passed"], r["ic"] if r["answered"] or not fallback else r.get("fallback", (False, None))

    def outcome_pair(r, fallback):
        if r["answered"] or not fallback:
            return r["passed"], r["ic"]
        return r.get("fallback", (False, None))

    def linear(r, fallback):
        passed, ic = outcome_pair(r, fallback)
        return 0.3 + 0.7 * float(np.clip((ic + 0.03) / 0.06, 0, 1)) if passed and ic is not None else 0.0

    def ranked(r, fallback):
        passed, ic = outcome_pair(r, fallback)
        return 0.1 + 0.9 * percentile(table[r["idea"]], ic) if passed and ic is not None else 0.0

    schemes = {
        "arm A's": lambda r: 1.0 if r["passed"] else 0.0,
        "arm B's": lambda r: linear(r, False),
        "arm B's, best test if no answer": lambda r: linear(r, True),
        "table, floor 0.1, best test if no answer": lambda r: ranked(r, True),
    }
    print("Mean advantage by number of tests")
    groups = groups_of(early)
    for name, score in schemes.items():
        by_tests = collections.defaultdict(list)
        for members in groups:
            rewards = np.array([score(x) for x in members])
            for x, advantage in zip(members, rewards - rewards.mean()):
                by_tests[x["tests"]].append(advantage)
        print(f"  {name:42s} " + " ".join(f"{k}: {np.mean(by_tests[k]):+.3f}" for k in sorted(by_tests)))
    print("Share of episodes and how they end, by number of tests, across all six runs")
    for first, last in ((1, 5), (31, 35)):
        rows = [r for arm in ARMS.values() for seed in (1, 2) for r in rollouts(arm, seed, first, last)]
        per = collections.defaultdict(list)
        for r in rows:
            per[r["tests"]].append(r)
        print(f"  steps {first}-{last}: " + " | ".join(f"{k}: {len(per[k]) / len(rows):.0%} of episodes, answered {np.mean([r['answered'] for r in per[k]]):.0%}, passed {np.mean([r['passed'] for r in per[k]]):.0%}" for k in sorted(per)))


def show_features() -> None:
    form = {
        "more tests": lambda r: r["tests"],
        "longer, per 50 characters": lambda r: len(norm(r["expression"])) / 50,
        "uses ln": lambda r: float(".ln(" in r["expression"]),
        "uses sec_rank": lambda r: float(".sec_rank(" in r["expression"]),
        "uses ts_rank": lambda r: float(".ts_rank(" in r["expression"]),
        "uses EMA": lambda r: float(".EMA(" in r["expression"] or "ewm" in r["expression"]),
        "uses ts_std": lambda r: float(".ts_std(" in r["expression"]),
        "uses a ratio": lambda r: float("safe_div" in r["expression"]),
        "two different windows": lambda r: float(len(set(re.findall(r"window=(\d+)", r["expression"]))) >= 2),
        "longest window, per 20 days": lambda r: max([int(w) for w in re.findall(r"(?:window|span|periods)=(\d+)", r["expression"])] or [0]) / 20,
        "two or more fields": lambda r: float(len(set(re.findall(r"Factor\('(\w+)'\)", r["expression"]))) >= 2),
    }

    def slopes(rows):
        per = by_idea([r for r in rows if r["passed"] and r["ic"] is not None and r["expression"]])
        out = {}
        for name, fn in form.items():
            pairs = {}
            for idea, members in per.items():
                if len(members) >= 2:
                    x = np.array([fn(m) for m in members], float)
                    y = np.array([m["ic"] for m in members])
                    pairs[idea] = (x - x.mean(), y - y.mean())
            out[name] = within_slope(pairs)
        return out

    student = [r for arm in ARMS.values() for seed in (1, 2) for r in rollouts(arm, seed)]
    val = []
    for model in ("S1", "S2", "S3 A", "S3 B", "S3 C", *ARMS):
        val += [{"idea": r["idea"], "passed": r["passed"], "ic": r["ic25"], "tests": r.get("tests") or 0, "expression": r.get("factor")} for r in answers(model, "val")]
    teacher = [
        {"idea": r["idea"], "passed": bool(r.get("matches_idea") and r.get("factor")), "ic": r.get("ic"), "tests": r.get("tests") or 0, "expression": r.get("factor")}
        for r in read_jsonl(f"{AGENT}/v2-deepseek-tools.jsonl")
        if IDEAS.get(r.get("idea"), {}).get("split") == "train"
    ]
    a, b, c = slopes(student), slopes(val), slopes(teacher)
    print("IC among passes per unit of a formula's feature, within an idea, 2025. No held-out idea here")
    print(f"  {'feature':30s} {'students, training ideas':36s} {'students, val ideas':36s} teacher, training ideas")
    for name in form:
        print(f"  {name:30s} " + "  ".join(f"{t[name][0]:+.5f} ({t[name][1]:+.5f},{t[name][2]:+.5f})" for t in (a, b, c)))
    per = by_idea([r for r in student if r["passed"] and r["ic"] is not None])
    grand = np.mean([r["ic"] for rows in per.values() for r in rows])
    between = sum(len(rows) * (np.mean([r["ic"] for r in rows]) - grand) ** 2 for rows in per.values())
    within = sum(float(((np.array([r["ic"] for r in rows]) - np.mean([r["ic"] for r in rows])) ** 2).sum()) for rows in per.values())
    print(f"Spread of IC among passes on training ideas: between ideas {between / (between + within):.0%}, within an idea {within / (between + within):.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["reward", "student", "persist", "pick", "push", "features"])
    args = ap.parse_args()
    {"reward": show_reward, "student": show_student, "persist": show_persist, "pick": show_pick, "push": show_push, "features": show_features}[args.what]()


if __name__ == "__main__":
    main()
