"""The training reward of prereg/03b-pilot.md."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from factor_idea import sandbox as sb
from factor_idea import workers
from factor_idea.taskset import FactorIdeaConfig, FactorIdeaTaskConfig
from factor_idea_rank import taskset as ts


def report(ic):
    return f"computes: yes, 2022-07-01 to 2024-12-29, 913 days, about 141 coins a day\nrank IC: +0.0100\nincremental IC after removing known styles: {ic:+.4f} (t = 2.0)\nincremental IC by period: 2022H2 +0.0001"


def call(expression):
    return json.dumps({"expression": expression})


def test_a_pass_earns_the_floor_at_the_bottom_of_the_table_and_full_marks_at_the_top():
    table = [0.0, 0.1, 0.2, 0.3]
    assert ts.rank_reward(-1.0, table) == pytest.approx(0.1)
    assert ts.rank_reward(0.1, table) == pytest.approx(0.55)
    assert ts.rank_reward(9.0, table) == pytest.approx(1.0)


def test_the_best_test_is_the_one_with_the_highest_reported_ic():
    calls = [call("a"), call("b"), call("c")]
    assert ts.best_tested(calls, [report(0.001), report(0.005), report(-0.002)]) == "b"


def test_a_test_that_did_not_compute_or_a_broken_call_is_never_the_best():
    assert ts.best_tested([call("broken"), call("a")], ["error: bad", report(-0.01)]) == "a"
    assert ts.best_tested(["{not json", call("a")], [report(0.9), report(0.001)]) == "a"
    assert ts.best_tested([call("broken")], ["does not compute: on too many days"]) is None
    assert ts.best_tested([], []) is None


class FakeTrace:
    def __init__(self, reply, tested=()):
        self.last_reply = reply
        self.assistant_messages = [SimpleNamespace(tool_calls=[SimpleNamespace(arguments=call(e))]) for e, _ in tested]
        self.tool_messages = [SimpleNamespace(content=report(ic)) for _, ic in tested]
        self.metrics = {}
        self.info = {}

    def record_metrics(self, values):
        self.metrics.update(values)


@pytest.fixture
def task(monkeypatch, tmp_path):
    scored = []

    async def score(text, key, direction, also=()):
        scored.append(text)
        if text == "fails":
            return sb.Score(parsed=True, computes=True, matches_idea=False, reason="moves_against_the_idea")
        if not text:
            return sb.Score()
        return sb.Score(parsed=True, computes=True, matches_idea=True, ic=sb.ICResult(0.015, 0.02, 300, 3.0), skeleton="x")

    monkeypatch.setattr(workers, "score", score)
    monkeypatch.setattr(ts, "_TABLE", {"up": [0.0, 0.01, 0.02, 0.03]})
    data = SimpleNamespace(idea_id="up", family="f", key="k", direction=1, also=[])
    item = ts.FactorIdeaRankTask.__new__(ts.FactorIdeaRankTask)
    object.__setattr__(item, "data", data)
    object.__setattr__(item, "config", FactorIdeaTaskConfig())
    item.scored = scored
    return item


def grade(task, trace):
    return asyncio.run(ts.FactorIdeaRankTask.factor_reward(task, trace))


def test_an_answer_is_scored_as_it_is_and_earns_its_rank(task):
    trace = FakeTrace("<factor>mine</factor>", [("tested", 0.009)])
    assert grade(task, trace) == pytest.approx(0.1 + 0.9 * 0.5)
    assert task.scored == ["mine"]
    assert trace.metrics["has_answer"] == 1.0 and trace.metrics["used_best_test"] == 0.0
    assert trace.metrics["reward_rank"] == pytest.approx(0.55)
    assert trace.info["factor"]["expression"] == "mine"


def test_no_answer_is_scored_on_the_best_test_and_says_so(task):
    trace = FakeTrace("I ran out of turns.", [("low", 0.001), ("high", 0.009)])
    assert grade(task, trace) == pytest.approx(0.55)
    assert task.scored == ["high"]
    assert trace.metrics["has_answer"] == 0.0 and trace.metrics["used_best_test"] == 1.0
    assert trace.info["factor"]["expression"] == "high"


def test_no_answer_and_nothing_tested_earns_nothing(task):
    trace = FakeTrace("I think volume matters.")
    assert grade(task, trace) == 0.0
    assert trace.metrics["used_best_test"] == 0.0
    assert trace.info["factor"]["reason"] == "no factor tag"


def test_a_failure_earns_nothing_and_over_the_limit_nothing(task):
    assert grade(task, FakeTrace("<factor>fails</factor>")) == 0.0
    over = FakeTrace("<factor>mine</factor>", [(f"t{i}", 0.001) for i in range(6)])
    assert grade(task, over) == 0.0
    assert over.metrics["over_limit"] == 1.0 and over.metrics["reward_gates"] == 0.0


def test_the_arm_b_reward_stays_on_record_beside_it(task):
    trace = FakeTrace("<factor>mine</factor>")
    grade(task, trace)
    assert trace.metrics["reward_ic"] == pytest.approx(sb.Score(parsed=True, computes=True, matches_idea=True, ic=sb.ICResult(0.015, 0.02, 300, 3.0)).reward("ic"))


def test_the_framework_rebuilds_a_dispatched_task_as_the_rank_task(monkeypatch):
    """As serve/server.py does: the task class from Taskset.task_type(), its data from the wire, the task config."""
    source = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "task" / "ideas_v2.jsonl")
    config = FactorIdeaConfig(id="factor-idea-rank", ideas=source, split="train")
    taskset = ts.FactorIdeaRankTaskset(config)
    task_cls = type(taskset).task_type()
    assert task_cls is ts.FactorIdeaRankTask
    sent = taskset.load()[0]
    rebuilt = task_cls(task_cls.data_type().model_validate(sent.data.model_dump()), config.task)
    assert type(rebuilt) is ts.FactorIdeaRankTask
    assert rebuilt.data.idea_id == sent.data.idea_id


def test_the_taskset_refuses_an_idea_with_no_table(monkeypatch):
    monkeypatch.setattr(ts, "_TABLE", {"elsewhere": [0.0]})
    source = str(__import__("pathlib").Path(__file__).resolve().parents[3] / "task" / "ideas_v2.jsonl")
    with pytest.raises(ValueError, match="no IC table"):
        ts.FactorIdeaRankTaskset(FactorIdeaConfig(id="factor-idea-rank", ideas=source, split="val")).load()
