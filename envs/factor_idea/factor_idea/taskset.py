"""One-sentence ideas as tasks. The final reply is scored by the sandbox, on 2025.

`use_tools = true` is the task: the model may call test_factor up to MAX_TESTS times, which needs
`env.agent.max_turns = 6` in the run's config. `use_tools = false` is the one-shot control.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from factor_idea import workers
from factor_idea.judge import judge
from factor_idea.prompt import MINUTE_SYSTEM_PROMPT, MINUTE_TOOL_SYSTEM_PROMPT, SYSTEM_PROMPT, TOOL_SYSTEM_PROMPT, extract_factor
from factor_idea.tools import FactorToolset

MAX_TESTS = 5
DEFAULT_IDEAS = str(Path(__file__).resolve().parents[3] / "task" / "ideas_draft.jsonl")


class FactorIdeaData(vf.TaskData):
    idea_id: str
    idea: str
    family: str
    key: str
    direction: int
    also: list[str] = []


class FactorIdeaTaskConfig(vf.TaskConfig):
    arm: Literal["ic", "gates", "judged"] = "ic"
    """"ic": passing earns the base and IC earns the rest. "gates": passing is full marks.
    "judged": as "ic", but nothing unless an LLM judge finds the formula faithful to the idea."""
    judge_model: str = "deepseek-flash"
    judge_base_url: str = "https://api.deepseek.com/v1"
    judge_api_key_var: str = "DEEPSEEK_API_KEY"


class FactorIdeaTask(vf.Task[FactorIdeaData, vf.State, FactorIdeaTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.idea_id

    @vf.reward(weight=1.0)
    async def factor_reward(self, trace: vf.Trace) -> float:
        text = extract_factor(trace.last_reply)
        tests = len(trace.tool_messages)
        over_limit = tests > MAX_TESTS
        result = await workers.score(text or "", self.data.key, self.data.direction, tuple(self.data.also))
        metrics = {
            "has_answer": float(text is not None),
            "parsed": float(result.parsed),
            "computes": float(result.computes),
            "matches_idea": float(result.matches_idea),
            "num_tests": float(tests),
            "over_limit": float(over_limit),
            "reward_gates": 0.0 if over_limit else result.reward("gates"),
            "reward_ic": 0.0 if over_limit else result.reward("ic"),
        }
        if result.ic is not None:
            metrics["ic"] = result.ic.incremental
            metrics["raw_ic"] = result.ic.raw
        faithful = False
        if self.config.arm == "judged" and text and result.matches_idea and not over_limit:
            faithful = await judge(
                self.data.idea,
                text,
                model=self.config.judge_model,
                base_url=self.config.judge_base_url,
                api_key_var=self.config.judge_api_key_var,
            )
            metrics["faithful"] = float(faithful)
        trace.record_metrics(metrics)
        trace.info["factor"] = {
            "idea": self.data.idea_id,
            "family": self.data.family,
            "expression": text,
            "skeleton": result.skeleton,
            "reason": result.reason if text else "no factor tag",
        }
        if self.config.arm == "judged":
            return metrics["reward_ic"] if faithful else 0.0
        return 0.0 if over_limit else result.reward(self.config.arm)


class FactorIdeaConfig(vf.TasksetConfig):
    ideas: str = DEFAULT_IDEAS
    split: Literal["all", "train", "val", "heldout"] = "all"
    """For a file with a split column: ideas to train on, ideas to choose settings with, or the exam's families."""
    use_tools: bool = True
    minutes: bool = False
    """The minute pilot's prompt, for runs with FACTOR_IDEA_MINUTES set. The task itself is daily bars only."""
    tools: vf.SharedToolsetConfig = vf.SharedToolsetConfig()
    task: FactorIdeaTaskConfig = FactorIdeaTaskConfig()


class FactorIdeaTaskset(vf.Taskset[FactorIdeaTask, FactorIdeaConfig]):
    @classmethod
    def toolsets(cls, config: FactorIdeaConfig) -> list[vf.Toolset]:
        return [FactorToolset(config.tools)] if config.use_tools else []

    def load(self) -> list[FactorIdeaTask]:
        if self.config.minutes:
            system = MINUTE_TOOL_SYSTEM_PROMPT if self.config.use_tools else MINUTE_SYSTEM_PROMPT
        else:
            system = TOOL_SYSTEM_PROMPT if self.config.use_tools else SYSTEM_PROMPT
        with open(self.config.ideas) as f:
            rows = [json.loads(line) for line in f]
        if self.config.split != "all":
            rows = [row for row in rows if row["split"] == self.config.split]
        return [
            FactorIdeaTask(
                FactorIdeaData(
                    idx=i,
                    idea_id=row["id"],
                    idea=row["en"],
                    family=row["family"],
                    key=row["key"],
                    direction=row["direction"],
                    also=row.get("also", []),
                    # In the user turn, as in wiki-search: given a system message, OLMo's chat template
                    # drops its own instruction on how to write a function call.
                    prompt=f"{system}\n\nIdea: {row['en']}",
                ),
                self.config.task,
            )
            for i, row in enumerate(rows)
        ]
