"""The taskset and its scoring, with the sandbox replaced by a stub."""
import asyncio

import pytest

import factor_idea
from factor_idea import sandbox as sb
from factor_idea import taskset as ts


def test_the_framework_can_find_the_taskset():
    assert factor_idea.FactorIdeaTaskset is ts.FactorIdeaTaskset
    assert factor_idea.__all__ == ["FactorIdeaTaskset"]


def tasks(**config):
    return ts.FactorIdeaTaskset(ts.FactorIdeaConfig(**config)).load()


def test_every_idea_becomes_one_task():
    loaded = tasks()
    assert len(loaded) == 31
    assert len({t.key for t in loaded}) == 31
    illiquid = next(t for t in loaded if t.key == "illiquid_up")
    assert illiquid.data.also == ["price"] and illiquid.data.direction == -1
    assert "test_factor" in illiquid.data.prompt
    assert illiquid.data.prompt.endswith("Idea: Coins whose price moves more per dollar traded, meaning they are illiquid, earn higher returns.")
    assert illiquid.data.system_prompt is None


def test_the_control_has_no_tools():
    config = ts.FactorIdeaConfig(use_tools=False)
    assert ts.FactorIdeaTaskset.toolsets(config) == []
    assert "test_factor" not in tasks(use_tools=False)[0].data.prompt
    assert len(ts.FactorIdeaTaskset.toolsets(ts.FactorIdeaConfig())) == 1


class FakeTrace:
    def __init__(self, reply, tests=0):
        self.last_reply = reply
        self.tool_messages = [object()] * tests
        self.metrics = {}
        self.info = {}

    def record_metrics(self, values):
        self.metrics.update(values)


def passing_score(*args, **kwargs):
    async def run():
        return sb.Score(parsed=True, computes=True, matches_idea=True, ic=sb.ICResult(0.015, 0.02, 300, 3.0), skeleton="x")

    return run()


def grade(trace, arm="ic"):
    task = tasks(task=ts.FactorIdeaTaskConfig(arm=arm))[0]
    return asyncio.run(task.factor_reward(trace))


def test_a_passing_answer_is_paid_by_its_arm(monkeypatch):
    monkeypatch.setattr(ts.workers, "score", passing_score)
    trace = FakeTrace("reasoning <factor>Factor('close')</factor>", tests=3)
    assert grade(trace, "ic") == pytest.approx(0.825)
    assert trace.metrics["num_tests"] == 3 and trace.metrics["has_answer"] == 1.0 and trace.metrics["ic"] == 0.015
    assert trace.info["factor"]["expression"] == "Factor('close')"
    assert grade(FakeTrace("<factor>Factor('close')</factor>"), "gates") == 1.0


def test_exactly_five_tests_are_allowed_and_six_score_zero(monkeypatch):
    monkeypatch.setattr(ts.workers, "score", passing_score)
    assert grade(FakeTrace("<factor>Factor('close')</factor>", tests=5)) > 0
    trace = FakeTrace("<factor>Factor('close')</factor>", tests=6)
    assert grade(trace) == 0.0
    assert trace.metrics["over_limit"] == 1.0


def test_a_reply_without_a_factor_scores_zero(monkeypatch):
    async def no_answer(*args, **kwargs):
        return sb.Score(reason="parse: expression is empty")

    monkeypatch.setattr(ts.workers, "score", no_answer)
    trace = FakeTrace("I think volume matters.")
    assert grade(trace) == 0.0
    assert trace.metrics["has_answer"] == 0.0
    assert trace.info["factor"]["reason"] == "no factor tag"


def test_olmo_keeps_its_function_call_instruction():
    """OLMo's template drops it when there is a system message, and then the model never calls a tool."""
    transformers = pytest.importorskip("transformers")
    import os

    model = "/mnt/nvme1/bruce/models/Olmo-3-7B-Instruct"
    if not os.path.isdir(model):
        pytest.skip("model not on this machine")
    tok = transformers.AutoTokenizer.from_pretrained(model)
    tool = {"type": "function", "function": {"name": "test_factor", "description": "d", "parameters": {"type": "object", "properties": {}}}}
    data = tasks()[0].data
    assert data.system_prompt is None
    text = tok.apply_chat_template([{"role": "user", "content": data.prompt}], tools=[tool], add_generation_prompt=True, tokenize=False)
    assert "<function_calls>" in text


def test_a_worker_does_not_outlive_its_parent():
    import os
    import subprocess
    import sys
    import time

    script = """
import asyncio, os, sys
from factor_idea import workers
print(asyncio.run(workers.report("not a factor")), flush=True)
children = [p.pid for p in workers._pool._processes.values()]
print(",".join(map(str, children)), flush=True)
os._exit(0)  # no exit hooks, as when the framework kills an env worker
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, env={**os.environ, "FACTOR_IDEA_WORKERS": "2"})
    lines = done.stdout.strip().splitlines()
    assert lines[0].startswith("error:"), done.stderr[-500:]
    children = [int(x) for x in lines[1].split(",")]
    deadline = time.monotonic() + 15
    alive = children
    while alive and time.monotonic() < deadline:
        time.sleep(0.2)
        alive = [pid for pid in children if os.path.exists(f"/proc/{pid}") and "Z" not in open(f"/proc/{pid}/stat").read().split()[2]]
    assert not alive


def test_keep_text_parsers_agree():
    """Eval and training must read a reply the same way, including the text before a call."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("vllm")
    renderers = pytest.importorskip("renderers.parsers")
    import importlib.util
    import os
    import pathlib

    model = "/mnt/nvme1/bruce/models/Olmo-3-7B-Instruct"
    if not os.path.isdir(model) or "olmo3_keep_text" not in renderers.TOOL_PARSERS:
        pytest.skip("needs the model and setup/install_olmo3_parser.sh")
    tok = transformers.AutoTokenizer.from_pretrained(model)
    path = pathlib.Path(__file__).resolve().parents[3] / "setup" / "vllm_olmo3_keep_text.py"
    spec = importlib.util.spec_from_file_location("vllm_olmo3_keep_text", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    served = module.Olmo3KeepTextToolParser(tok)
    trained = renderers.TOOL_PARSERS["olmo3_keep_text"](tok)
    call = "<function_calls>test_factor(expression=\"Factor('close').lag(1)\")</function_calls>"
    for reply, want in [
        ("I test the lag first." + call, "I test the lag first."),
        ("Two lines.\n\nThen a call: " + call, "Two lines.\n\nThen a call: "),
        (call, None),
        ("  \n" + call, None),
    ]:
        info = served.extract_tool_calls(reply, None)
        assert info.tools_called and info.content == want
        ids, parsed = trained.extract(tok(reply, add_special_tokens=False)["input_ids"])
        assert parsed[0].name == "test_factor" and parsed[0].arguments == {"expression": "Factor('close').lag(1)"}
        assert (tok.decode(ids) if ids else None) == want
    # No call: the whole reply stays text, in both.
    text = "No call here. <factor>Factor('close')</factor>"
    assert served.extract_tool_calls(text, None).content == text
    ids, parsed = trained.extract(tok(text, add_special_tokens=False)["input_ids"])
    assert tok.decode(ids) == text and not parsed
