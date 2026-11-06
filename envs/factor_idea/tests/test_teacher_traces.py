"""The teacher script and the dataset built from it: two calls in a reply, and starting from a student's mistake."""
import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts", "factor"))

openai_types = pytest.importorskip("openai.types.chat")
teacher_traces = pytest.importorskip("teacher_traces")
from make_sft_dataset import keep, off_script, wire  # noqa: E402

IDEA = {"id": "volume_up", "family": "volume_level", "key": "volume", "direction": 1}
ARGS = SimpleNamespace(model="teacher", max_tokens=100)


def reply(content, *expressions):
    calls = [
        {"id": f"call_{i}", "type": "function", "function": {"name": "test_factor", "arguments": json.dumps({"expression": e})}}
        for i, e in enumerate(expressions)
    ]
    return openai_types.ChatCompletionMessage.model_validate({"role": "assistant", "content": content, "tool_calls": calls or None})


class FakeClient:
    """Answers in order, and keeps what each request sent."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, messages, **_):
        self.requests.append(json.loads(json.dumps(messages)))
        return SimpleNamespace(choices=[SimpleNamespace(message=self.replies.pop(0))])


@pytest.fixture
def workers(monkeypatch):
    async def report(expression):
        return "error: unsupported function: lag" if "lag(" in expression else "computes: yes"

    async def score(*_):
        return SimpleNamespace(parsed=True, computes=True, matches_idea=True, reason="")

    monkeypatch.setattr(teacher_traces.workers, "report", report)
    monkeypatch.setattr(teacher_traces.workers, "score", score)


def run(client, prefix=()):
    return asyncio.run(teacher_traces.episode(client, ARGS, "system", IDEA, 0, "Idea text", 0, None, list(prefix)))


def test_two_calls_in_one_reply_stay_on_that_reply(workers):
    row = run(FakeClient(reply("Compare two.", "a", "b"), reply("Done. <factor>b</factor>")))
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["user", "assistant", "tool", "tool", "assistant"]
    assert row["messages"][1]["calls"] == ["a", "b"]
    assert row["tests"] == 2 and row["factor"] == "b" and row["final_was_tested"]


def test_two_calls_pair_with_their_results_on_the_wire(workers):
    row = run(FakeClient(reply("Compare two.", "a", "b"), reply("Done. <factor>b</factor>")))
    sent = wire(row["messages"])
    ids = [c["id"] for c in sent[1]["tool_calls"]]
    assert [sent[2]["tool_call_id"], sent[3]["tool_call_id"]] == ids and len(set(ids)) == 2


def test_an_episode_out_of_turns_ends_on_its_last_call(workers):
    row = run(FakeClient(*[reply("Again.", "a") for _ in range(teacher_traces.MAX_TURNS)]))
    assert row["stop"] == "max_turns" and row["factor"] is None and row["messages"][-1]["role"] == "tool"
    last = wire(row["messages"])[-1]
    assert last["role"] == "assistant" and last["tool_calls"]


def test_every_assistant_turn_states_whether_it_is_trained(workers):
    # The dataset loader fixes its schema from the first rows, and recovery rows come later.
    client = FakeClient(reply("Fix it.", "a"), reply("Done. <factor>a</factor>"))
    row = run(client, [{"content": "", "calls": ["lag(x)"]}])
    assert [m["train"] for m in wire(row["messages"]) if m["role"] == "assistant"] == [False, True, True]


def test_a_broken_start_is_replayed_but_not_trained(workers):
    client = FakeClient(reply("`lag` needs an expression.", "a"), reply("Done. <factor>a</factor>"))
    row = run(client, [{"content": "", "calls": ["lag(x)"]}])
    # DeepSeek's thinking mode refuses a past tool call without this field.
    assert "reasoning_content" in client.requests[0][2]
    assert client.requests[0][3]["content"].startswith("error:")
    assert row["recovered"] and row["tests"] == 2
    first, *_ = [m for m in row["messages"] if m["role"] == "assistant"]
    assert first["train"] is False and first["calls"] == ["lag(x)"]
    assert wire(row["messages"])[1]["train"] is False


def test_broken_starts_read_several_files_and_skip_repeats(tmp_path):
    def row(expression, result="error: bad"):
        call = json.dumps({"expression": expression})
        return {"idea": "volume_up", "steps": [{"role": "assistant", "content": "", "calls": [call]}, {"role": "tool", "content": result}]}

    one, two = tmp_path / "one.jsonl", tmp_path / "two.jsonl"
    one.write_text("\n".join(json.dumps(r) for r in [row("x"), row("x"), row("worked", "computes: yes")]))
    two.write_text("\n".join(json.dumps(r) for r in [row("x"), row("y"), row("z")]))
    starts = teacher_traces.broken_starts([str(one), str(two)], {"volume_up"}, 2)
    assert [turns[0]["calls"] for turns in starts["volume_up"]] == [["x"], ["y"]]


@pytest.mark.parametrize(
    "text, expected",
    [("The IC is positive, so I keep it.", False), ("As instructed, I keep the direction.", True), ("这个版本最好。", True)],
)
def test_words_from_the_brief_stay_out(text, expected):
    row = {"messages": [{"role": "assistant", "content": text, "calls": []}]}
    assert off_script(row) is expected


def test_a_students_own_words_in_an_untrained_turn_do_not_count():
    row = {"messages": [{"role": "assistant", "content": "这个", "calls": ["x"], "train": False}, {"role": "tool", "content": "error"}]}
    assert off_script(row) is False


def test_two_calls_in_one_reply_stay_out():
    row = {"messages": [{"role": "assistant", "content": "Two at once.", "calls": ["a", "b"]}]}
    assert off_script(row) is True


def test_keep_says_why():
    row = {"idea": "volume_up", "matches_idea": True, "final_was_tested": True, "messages": [{"role": "assistant", "content": "Fine.", "calls": []}]}
    assert keep(row, {"volume_up": "train"}, "train") == ""
    assert keep(row, {"volume_up": "val"}, "train") == "not in train"
    assert keep({**row, "matches_idea": False}, {"volume_up": "train"}, "train") == "failed the gates"
    assert keep({**row, "final_was_tested": False}, {"volume_up": "train"}, "train") == "handed in something untested"
    assert keep({**row, "leaks_the_manual": True}, {"volume_up": "train"}, "train") == "off script"
