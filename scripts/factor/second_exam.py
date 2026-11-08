#!/usr/bin/env python3
"""The second exam of prereg/04-second-exam.md: every model answers ideas none has read, on 2025 and then on 2026.

    python scripts/factor/second_exam.py answers --pilots <runs that passed 03b>    # answers on 2025, then the judge
    python scripts/factor/second_exam.py score --pilots <runs> --confirm "<phrase in panel.py>"   # rehearses on 2025, then 2026, once per file
    python scripts/factor/second_exam.py report --pilots <runs>    # E1 to E5, and what is reported beside them

Only `score` opens the sealed panel, and only once every file is whole and gives back its own verdicts and ICs on
2025. A file already scored is left alone, so a restart can't score it twice. The answer files are never written
after the judge. Run from a tmux session with setup/env.sh sourced.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_stages import BOOTSTRAP, interval  # noqa: E402
from explore_after_exam import answer_research_ic, pick_one, read_jsonl, trials  # noqa: E402
from heldout_online import mirrored  # noqa: E402
from run_stages import PY, REPO, ROOT, note, shell, wait_gpus_free  # noqa: E402
from sealed_exam import differences, rescore  # noqa: E402

IDEAS = "task/ideas_fresh.jsonl"
PREREG = "prereg/04-second-exam.md"
ADAPTERS = "results/factor/adapters.md"
AGENT = "results/factor/agent"
OUT = "results/factor/exam2"
TEACHER = "DeepSeek"
ROLLOUTS = 16
DRAWS = 2000
SEEDS = (1, 2)
COVERAGE = "too_few_coins_or_constant_when_scored"
MOST_PASS_LOST = 0.05
E4_FLOOR = 0.3


# Kept pure so it can be tested.
def roster(pilots: list[str]) -> dict[str, list[str]]:
    """Each model and the names of its answer files. A name is also its adapter's, after factor-."""
    table = {"S1": ["s1"], "S2": ["s2"], "S3 B": ["s3p-ic"]}
    for model, arm in (("S4 A", "gates"), ("S4 B", "ic"), ("S4 C", "judged")):
        table[model] = [f"s4-{arm}-seed{seed}" for seed in SEEDS]
    for run in pilots:
        short = run.removeprefix("factor-")
        table[short] = [short]
    table[TEACHER] = ["deepseek"]
    return table


def frozen_problems(prereg: str, ideas_sha: str, pilots: list[str]) -> list[str]:
    """What keeps a model from reading the ideas, by the text of the prereg."""
    found = []
    if "Status: FROZEN" not in prereg:
        found.append("prereg 04 is not frozen")
    if ideas_sha not in prereg:
        found.append("the ideas file is not the one pinned")
    return found + [f"{run} is not named in prereg 04" for run in pilots if run not in prereg]


def pinned(pins: str, name: str, sha: str) -> bool:
    """The adapter matches the last line written for its name in adapters.md."""
    found = re.findall(rf"^- {re.escape(name)}: `([0-9a-f]+)`", pins, flags=re.M)
    return bool(found) and found[-1] == sha


def whole(rows: list[dict], ideas: set[str], rollouts: int) -> bool:
    """Every idea has its rollouts and no episode failed: a failed episode has no idea on it."""
    counts = collections.Counter(row.get("idea") for row in rows)
    return set(counts) == ideas and set(counts.values()) == {rollouts}


def year(row: dict, tag: str) -> dict:
    """How an answer scores on one year. IC counts only for a pass on that year."""
    passed = bool(row.get("matches_idea"))
    return {
        f"pass{tag}": passed,
        f"ic{tag}": row.get("ic") if passed else None,
        f"raw{tag}": row.get("raw_ic") if passed else None,
        f"reward{tag}": row.get("reward_ic") or 0.0,
    }


def merged(before: list[dict], after: list[dict]) -> list[dict]:
    """One row per episode. `passed` and `research` are what a process at the end of 2024 can see: the gates' verdict,
    without the check that the formula still has coins on the year it is scored on, and its own test report."""
    out = []
    for row, later in zip(before, after, strict=True):
        if row.get("idea") != later.get("idea") or row.get("factor") != later.get("factor"):
            raise ValueError("the two years' files do not hold the same answers")
        answered = bool(row.get("factor"))
        out.append(
            {
                "idea": row["idea"],
                "answered": answered,
                "tests": row.get("tests") or 0,
                "passed": answered and not row.get("over_limit") and (bool(row.get("matches_idea")) or row.get("reason") == COVERAGE),
                "research": answer_research_ic(row.get("factor"), trials(row.get("steps"))),
                "faithful": row.get("faithful"),
                **year(row, "25"),
                **year(later, "26"),
            }
        )
    return out


def per_idea(rows: list[dict], field: str) -> dict[str, float]:
    """A field's mean per idea over the rows that have it, so an idea with no pass has no IC."""
    values = collections.defaultdict(list)
    for row in rows:
        if row[field] is not None:
            values[row["idea"]].append(float(row[field]))
    return {idea: float(np.mean(v)) for idea, v in values.items()}


def gap(later: dict[str, float], earlier: dict[str, float], only: set[str] | None = None) -> tuple[float, float, float, int]:
    """Later minus earlier over the ideas both have, with whole ideas resampled: the difference, its CI, how many ideas."""
    shared = set(later) & set(earlier)
    shared = sorted(shared if only is None else shared & only)
    if not shared:
        return float("nan"), float("nan"), float("nan"), 0
    return (*interval([later[i] - earlier[i] for i in shared]), len(shared))


def keep_one(rows: list[dict], k: int, draws: int = DRAWS, seed: int = 0) -> list[dict]:
    """What the process keeps from k episodes of an idea. k=1 takes each episode once. `spent` is the tests it took."""
    rng = np.random.default_rng(seed)
    per = collections.defaultdict(list)
    for row in rows:
        per[row["idea"]].append(row)
    kept = []
    for idea in sorted(per):
        members = per[idea]
        if k == 1:
            subsets = [[row] for row in members]
        elif k >= len(members):
            subsets = [members]
        else:
            subsets = [[members[i] for i in rng.choice(len(members), k, replace=False)] for _ in range(draws)]
        kept += [{**pick_one(subset, "gates", rng), "spent": sum(row["tests"] for row in subset)} for subset in subsets]
    return kept


def within_points(models: dict[str, list[dict]], tag: str, only: set[str] | None = None) -> dict[str, list[list[tuple[float, float]]]]:
    """For each idea, each model's passes on it that were tested: research IC and the year's IC."""
    points = collections.defaultdict(list)
    for rows in models.values():
        groups = collections.defaultdict(list)
        for row in rows:
            if row[f"pass{tag}"] and row["research"] is not None and row[f"ic{tag}"] is not None and (only is None or row["idea"] in only):
                groups[row["idea"]].append((row["research"], row[f"ic{tag}"]))
        for idea, group in groups.items():
            points[idea].append(group)
    return points


def within_correlation(points: dict[str, list[list[tuple[float, float]]]], seed: int = 0) -> tuple[float, float, float, int]:
    """Correlation after taking out each group's means, pooled, with whole ideas resampled. Also how many pairs."""
    keys = sorted(points)
    sums, pairs = [], 0
    for idea in keys:
        xx = yy = xy = 0.0
        for group in points[idea]:
            if len(group) < 2:
                continue
            centred = np.asarray(group, dtype=float)
            centred = centred - centred.mean(axis=0)
            xx += float(centred[:, 0] @ centred[:, 0])
            yy += float(centred[:, 1] @ centred[:, 1])
            xy += float(centred[:, 0] @ centred[:, 1])
            pairs += len(group)
        sums.append((xx, yy, xy))
    if not pairs:
        return float("nan"), float("nan"), float("nan"), 0
    table = np.asarray(sums)
    total = table.sum(axis=0)
    drawn = table[np.random.default_rng(seed).integers(0, len(keys), size=(BOOTSTRAP, len(keys)))].sum(axis=1)
    values = drawn[:, 2] / np.sqrt(np.maximum(drawn[:, 0] * drawn[:, 1], 1e-24))
    return float(total[2] / np.sqrt(total[0] * total[1])), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), pairs


def e5_passes(ic_low: float, pass_gap: float) -> bool:
    return ic_low > 0 and pass_gap >= -MOST_PASS_LOST - 1e-12


def hypotheses(models: dict[str, list[dict]], pilots: list[str], tag: str = "26", only: set[str] | None = None) -> list[dict]:
    """E1 to E5 on one year. Each entry holds its numbers, and whether it passes by the prereg's rule."""
    ic, rate = f"ic{tag}", f"pass{tag}"
    s2, b = models["S2"], models["S4 B"]
    e1 = gap(per_idea(s2, ic), per_idea(b, ic), only)
    e2 = gap(per_idea(b, rate), per_idea(s2, rate), only)
    e3 = gap(per_idea(keep_one(b, 4), ic), per_idea(keep_one(b, 1), ic), only)
    e4 = within_correlation(within_points(models, tag, only))
    out = [
        {"id": "E1", "what": "S2 minus S4 B, IC among passes", "ic": e1, "passes": e1[1] > 0},
        {"id": "E2", "what": "S4 B minus S2, pass rate", "rate": e2, "passes": e2[1] > 0},
        {"id": "E3", "what": "S4 B, one of four minus one episode, IC among passes", "ic": e3, "passes": e3[1] > 0},
        {"id": "E4", "what": "research IC against the year's IC, within one model's passes on one idea", "correlation": e4, "passes": e4[1] > E4_FLOOR},
    ]
    for run in pilots:
        short = run.removeprefix("factor-")
        ic_gap = gap(per_idea(models[short], ic), per_idea(b, ic), only)
        rate_gap = gap(per_idea(models[short], rate), per_idea(b, rate), only)
        out.append({"id": "E5", "what": f"{short} minus S4 B", "ic": ic_gap, "rate": rate_gap, "passes": e5_passes(ic_gap[1], rate_gap[0])})
    return out


def show_ic(value: tuple[float, float, float, int]) -> str:
    return f"{value[0]:+.4f} ({value[1]:+.4f} to {value[2]:+.4f}) over {value[3]} ideas"


def show_rate(value: tuple[float, float, float, int]) -> str:
    return f"{value[0]:+.1%} ({value[1]:+.1%} to {value[2]:+.1%}) over {value[3]} ideas"


def show_entry(entry: dict) -> str:
    parts = []
    if "ic" in entry:
        parts.append(f"IC {show_ic(entry['ic'])}")
    if "rate" in entry:
        parts.append(f"pass rate {show_rate(entry['rate'])}")
    if "correlation" in entry:
        r, low, high, pairs = entry["correlation"]
        parts.append(f"correlation {r:+.3f} ({low:+.3f} to {high:+.3f}) over {pairs} passes")
    return "; ".join(parts)


def mean_over_ideas(rows: list[dict], field: str) -> float:
    values = per_idea(rows, field)
    return float(np.mean(list(values.values()))) if values else float("nan")


def describe(rows: list[dict]) -> str:
    """One model or process, averaged per idea and then over ideas."""

    def ic(field: str) -> str:
        values = list(per_idea(rows, field).values())
        if not values:
            return "no pass"
        mean, low, high = interval(values)
        return f"{mean:+.4f} ({low:+.4f} to {high:+.4f}), {len(values)} ideas"

    faithful = per_idea(rows, "faithful")
    return (
        f"answered {mean_over_ideas(rows, 'answered'):.1%}, tests {mean_over_ideas(rows, 'tests'):.2f}, faithful {f'{np.mean(list(faithful.values())):.1%}' if faithful else 'not judged'}\n"
        f"      2026: pass {mean_over_ideas(rows, 'pass26'):.1%}, reward {mean_over_ideas(rows, 'reward26'):.3f}, IC {ic('ic26')}, raw IC {ic('raw26')}\n"
        f"      2025: pass {mean_over_ideas(rows, 'pass25'):.1%}, reward {mean_over_ideas(rows, 'reward25'):.3f}, IC {ic('ic25')}, raw IC {ic('raw25')}"
    )


def report(models: dict[str, list[dict]], ideas: list[dict], pilots: list[str], say) -> None:
    """Everything prereg 04 says it reports. Only the first block decides anything."""
    pairs = mirrored(ideas)
    outside = {i["id"] for i in ideas} - pairs
    say(f"{len(ideas)} ideas, {len(pairs)} of them in mirrored pairs. Each model has {ROLLOUTS} rollouts per idea, and two seeds give half each.\n")

    say("Hypotheses, on 2026")
    for entry in hypotheses(models, pilots):
        say(f"  {entry['id']}: {entry['what']}: {show_entry(entry)}: {'PASSES' if entry['passes'] else 'does not pass'}")
    say("\nThe same on 2025, deciding nothing")
    for entry in hypotheses(models, pilots, "25"):
        say(f"  {entry['id']}: {entry['what']}: {show_entry(entry)}")
    say("\nThe same on 2026, outside mirrored pairs, deciding nothing")
    for entry in hypotheses(models, pilots, only=outside):
        say(f"  {entry['id']}: {entry['what']}: {show_entry(entry)}")
    say("\nThe same on 2026, by family, deciding nothing")
    for family in sorted({i["family"] for i in ideas}):
        members = {i["id"] for i in ideas if i["family"] == family}
        say(f"  {family}, {len(members)} ideas")
        for entry in hypotheses(models, pilots, only=members):
            say(f"    {entry['id']}: {entry['what']}: {show_entry(entry)}")

    say("\nEach model, per idea")
    for model, rows in models.items():
        say(f"  {model}: {len(rows)} episodes, {describe(rows)}")
    say("\nResearch IC against each year's IC, within one model's passes on one idea")
    for model in models:
        late = within_correlation(within_points({model: models[model]}, "26"))
        early = within_correlation(within_points({model: models[model]}, "25"))
        say(f"  {model}: 2026 {late[0]:+.3f} ({late[1]:+.3f} to {late[2]:+.3f}) over {late[3]} passes; 2025 {early[0]:+.3f} ({early[1]:+.3f} to {early[2]:+.3f}) over {early[3]} passes")

    say("\nAgainst the teacher")
    teacher = models[TEACHER]
    for model in ["S2", "S4 B", *[run.removeprefix("factor-") for run in pilots]]:
        for tag, label in (("26", "2026"), ("25", "2025")):
            say(
                f"  {model} minus DeepSeek, {label}: IC {show_ic(gap(per_idea(models[model], f'ic{tag}'), per_idea(teacher, f'ic{tag}')))}; "
                f"pass rate {show_rate(gap(per_idea(models[model], f'pass{tag}'), per_idea(teacher, f'pass{tag}')))}; "
                f"reward {show_ic(gap(per_idea(models[model], f'reward{tag}'), per_idea(teacher, f'reward{tag}')))}"
            )

    say("\nKeeping one of k answers by research IC")
    kept = {}
    for model, sizes in (("S4 B", (1, 2, 4, 8)), ("S2", (1, 2, 4)), (TEACHER, (1, 2, 4))):
        kept[model] = {k: keep_one(models[model], k) for k in sizes}
        for k, rows in kept[model].items():
            say(f"  {model}, k={k}, tests spent {mean_over_ideas(rows, 'spent'):.1f}: {describe(rows)}")
            if k > 1:
                say(f"      gain over one episode: 2026 IC {show_ic(gap(per_idea(rows, 'ic26'), per_idea(kept[model][1], 'ic26')))}; 2025 IC {show_ic(gap(per_idea(rows, 'ic25'), per_idea(kept[model][1], 'ic25')))}")
    say("\nThe process against the teacher's, at about the same number of tests, 2026")
    for label, a, b in (("S4 B k=4 minus DeepSeek k=2", kept["S4 B"][4], kept[TEACHER][2]), ("S2 k=2 minus DeepSeek k=2", kept["S2"][2], kept[TEACHER][2])):
        say(f"  {label}: IC {show_ic(gap(per_idea(a, 'ic26'), per_idea(b, 'ic26')))}; pass rate {show_rate(gap(per_idea(a, 'pass26'), per_idea(b, 'pass26')))}; reward {show_ic(gap(per_idea(a, 'reward26'), per_idea(b, 'reward26')))}")


def answer_file(name: str) -> str:
    return f"{AGENT}/exam2-{name}.jsonl"


def rollouts_of(table: dict[str, list[str]], name: str) -> int:
    return ROLLOUTS // next(len(names) for names in table.values() if name in names)


def idea_rows() -> list[dict]:
    return read_jsonl(IDEAS)


def is_whole(name: str, ideas: set[str], rollouts: int) -> bool:
    return os.path.exists(answer_file(name)) and whole(read_jsonl(answer_file(name)), ideas, rollouts)


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_frozen(table: dict[str, list[str]], pilots: list[str]) -> None:
    with open(PREREG) as f:
        problems = frozen_problems(f.read(), sha256(IDEAS), pilots)
    with open(ADAPTERS) as f:
        pins = f.read()
    for name in (n for names in table.values() for n in names if n != "deepseek"):
        path = f"{ROOT}/adapters/factor-{name}/adapter_model.safetensors"
        if not os.path.exists(path):
            problems.append(f"no adapter at {path}")
        elif not pinned(pins, name, sha256(path)):
            problems.append(f"the adapter of {name} is not the one pinned")
    if problems:
        sys.exit("no model reads the ideas yet: " + "; ".join(problems))


def judge(name: str) -> None:
    shell(f"cd {REPO} && {PY} scripts/factor/judge_eval.py {answer_file(name)} --ideas {IDEAS}", f"eval-exam2-{name}")


def run_students(names: list[str], ideas: set[str], table: dict[str, list[str]]) -> None:
    queue, lock = list(names), threading.Lock()

    def worker(gpus: str, port: int) -> None:
        while True:
            with lock:
                if not queue:
                    return
                name = queue.pop(0)
            rollouts = rollouts_of(table, name)
            for attempt in (1, 2):
                if is_whole(name, ideas, rollouts):
                    break
                wait_gpus_free(gpus)
                code = shell(f"IDEAS={REPO}/{IDEAS} ROLLOUTS={rollouts} bash {REPO}/scripts/factor/eval_stage.sh exam2-{name} {ROOT}/adapters/factor-{name} heldout {gpus} {port} 128", f"eval-exam2-{name}")
                note("exam 2 answers", run=name, attempt=attempt, exit=code)
            if is_whole(name, ideas, rollouts):
                judge(name)
                note("exam 2 judged", run=name)
            else:
                note("exam 2 file is not whole", run=name)

    threads = [threading.Thread(target=worker, args=a) for a in (("0,1", 18200), ("2,3", 18400))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def run_teacher(ideas: set[str]) -> None:
    """As in run_baselines.sh, with more rollouts on these ideas. The key stays in the environment."""
    name, out = "deepseek", f"{ROOT}/outputs/evals/factor/exam2-deepseek"
    for attempt in (1, 2):
        if is_whole(name, ideas, ROLLOUTS):
            return
        code = shell(
            f"rm -rf {out}; export FACTOR_IDEA_WORKERS=16 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; "
            f"(cd {os.environ['PRIME_RL_DIR']} && uv run eval @ {REPO}/configs/factor-idea/eval.toml -m deepseek-flash -r {ROLLOUTS} -c 16 --output-dir {out} --no-rich "
            f"--client.base-url https://api.deepseek.com/v1 --client.api-key-var DEEPSEEK_API_KEY --sampling.max-tokens 4000 "
            f"--env.taskset.ideas {REPO}/{IDEAS} --env.taskset.split heldout --env.taskset.use-tools true) && "
            f"cd {REPO} && {PY} scripts/factor/summarize_agent_eval.py {out} exam2-deepseek > /dev/null",
            "eval-exam2-deepseek",
        )
        note("exam 2 answers", run=name, attempt=attempt, exit=code)
    if not is_whole(name, ideas, ROLLOUTS):
        note("exam 2 file is not whole", run=name)


def answers(pilots: list[str]) -> None:
    table = roster(pilots)
    check_frozen(table, pilots)
    ideas = {i["id"] for i in idea_rows()}
    names = [n for model in ("S2", "S4 B") for n in table[model]]
    names += [n for model in table if model not in ("S2", "S4 B", TEACHER) for n in table[model]]
    note("exam 2 answers begin", models=list(table))
    teacher = threading.Thread(target=run_teacher, args=(ideas,))
    teacher.start()
    run_students(names, ideas, table)
    teacher.join()
    note("exam 2 answers end", whole={n: is_whole(n, ideas, rollouts_of(table, n)) for names in table.values() for n in names})


def score(pilots: list[str], confirm: str) -> None:
    """Rehearse every file on 2025 first, as prereg 03's exam did, and open 2026 only if each gives itself back."""
    table = roster(pilots)
    ideas = {i["id"] for i in idea_rows()}
    names = [n for group in table.values() for n in group]
    broken = [n for n in names if not is_whole(n, ideas, rollouts_of(table, n))]
    if broken:
        sys.exit(f"2026 stays closed, these files are not whole: {broken}")
    os.makedirs(f"{ROOT}/tmp/exam2-rehearsal", exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    for name in names:
        out = f"{ROOT}/tmp/exam2-rehearsal/{name}.jsonl"
        rescore(answer_file(name), out, "--period", "reward", "--ideas", IDEAS)
        found = differences(read_jsonl(out), read_jsonl(answer_file(name)))
        if found:
            sys.exit(f"{name} does not reproduce on 2025, so 2026 stays closed: {found[:5]}")
    print(f"all {len(names)} files give back their verdicts and ICs on 2025", flush=True)
    for name in names:
        out = f"{OUT}/{name}.jsonl"
        if not os.path.exists(out):
            rescore(answer_file(name), out, "--period", "sealed", "--confirm", confirm, "--ideas", IDEAS)
            note("exam 2 scored on 2026", run=name)


def show_report(pilots: list[str]) -> None:
    models = {model: [row for name in names for row in merged(read_jsonl(answer_file(name)), read_jsonl(f"{OUT}/{name}.jsonl"))] for model, names in roster(pilots).items()}
    with open(f"{OUT}/comparisons.txt", "w") as out:

        def say(line: str) -> None:
            print(line, flush=True)
            out.write(line + "\n")

        report(models, idea_rows(), pilots, say)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["answers", "score", "report"])
    ap.add_argument("--pilots", default="", help="the runs that passed pilot 03b, with commas")
    ap.add_argument("--confirm", default="")
    args = ap.parse_args()
    os.chdir(REPO)
    pilots = [run for run in args.pilots.split(",") if run]
    if args.step == "answers":
        answers(pilots)
    elif args.step == "score":
        score(pilots, args.confirm)
    else:
        show_report(pilots)


if __name__ == "__main__":
    if not (ROOT and REPO):
        sys.exit("source setup/env.sh first")
    main()
