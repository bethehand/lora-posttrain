"""The second run of prereg/03b-pilot.md: factor-idea's task with a reward in which IC can be heard.

A pass earns FLOOR plus the rest in proportion to its percentile among passes on the same idea, from `ic_table.json`.
An episode with no answer is scored on the formula it tested best on the research window. The model sees the same
prompts and tool results as in factor-idea, and evaluation stays on factor-idea. Only the training reward differs.
"""
from __future__ import annotations

import json
import re
from bisect import bisect_right
from pathlib import Path

import verifiers.v1 as vf
from factor_idea import workers
from factor_idea.prompt import extract_factor
from factor_idea.taskset import MAX_TESTS, FactorIdeaTask, FactorIdeaTaskset

FLOOR = 0.1
TABLE_PATH = Path(__file__).with_name("ic_table.json")
REPORTED_IC = re.compile(r"incremental IC after removing known styles: ([+-]?[0-9.]+) \(t = ")


# Kept pure so it can be tested.
def rank_reward(ic: float, table: list[float]) -> float:
    return FLOOR + (1 - FLOOR) * bisect_right(table, ic) / len(table)


def best_tested(calls: list[str], reports: list[str]) -> str | None:
    """The expression whose test showed the highest research IC. A test that didn't compute shows none."""
    best, best_ic = None, None
    for call, report in zip(calls, reports):
        found = REPORTED_IC.search(report or "")
        if not found:
            continue
        try:
            expression = json.loads(call).get("expression")
        except (ValueError, AttributeError, TypeError):
            continue
        ic = float(found.group(1))
        if expression and (best_ic is None or ic > best_ic):
            best, best_ic = expression, ic
    return best


def load_table(path: Path = TABLE_PATH) -> dict[str, list[float]]:
    with open(path) as f:
        return json.load(f)


class FactorIdeaRankTask(FactorIdeaTask):
    @vf.reward(weight=1.0)
    async def factor_reward(self, trace: vf.Trace) -> float:
        own = extract_factor(trace.last_reply)
        calls = [call.arguments for message in trace.assistant_messages for call in (message.tool_calls or [])]
        reports = [str(message.content or "") for message in trace.tool_messages]
        text = own or best_tested(calls, reports)
        tests = len(trace.tool_messages)
        over_limit = tests > MAX_TESTS
        result = await workers.score(text or "", self.data.key, self.data.direction, tuple(self.data.also))
        earned = 0.0
        if result.matches_idea and result.ic is not None and not over_limit:
            earned = rank_reward(result.ic.incremental, _table()[self.data.idea_id])
        metrics = {
            "has_answer": float(own is not None),
            "used_best_test": float(own is None and text is not None),
            "parsed": float(result.parsed),
            "computes": float(result.computes),
            "matches_idea": float(result.matches_idea),
            "num_tests": float(tests),
            "over_limit": float(over_limit),
            "reward_gates": 0.0 if over_limit else result.reward("gates"),
            "reward_ic": 0.0 if over_limit else result.reward("ic"),
            "reward_rank": earned,
        }
        if result.ic is not None:
            metrics["ic"] = result.ic.incremental
            metrics["raw_ic"] = result.ic.raw
        trace.record_metrics(metrics)
        trace.info["factor"] = {
            "idea": self.data.idea_id,
            "family": self.data.family,
            "expression": text,
            "skeleton": result.skeleton,
            "reason": result.reason if text else "no factor tag",
        }
        return earned


_TABLE: dict[str, list[float]] = {}


def _table() -> dict[str, list[float]]:
    if not _TABLE:
        _TABLE.update(load_table())
    return _TABLE


class FactorIdeaRankTaskset(FactorIdeaTaskset):
    def load(self) -> list[FactorIdeaRankTask]:
        tasks = [FactorIdeaRankTask(task.data, task.config) for task in super().load()]
        missing = sorted({task.data.idea_id for task in tasks} - set(_table()))
        if missing:
            raise ValueError(f"no IC table for {missing[:5]}: this taskset is for training ideas")
        return tasks
