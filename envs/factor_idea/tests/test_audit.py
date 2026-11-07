"""How the audit of prereg 03 draws its sample and settles H7 and H9."""
import pathlib
import random
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "factor"))

from audit import by_model, difference, read_once, same_ideas, sample_evenly, within_ten_points  # noqa: E402


def rows(**counts):
    return [{"idea": idea, "factor": f"{idea}{i}", "file": "f"} for idea, count in counts.items() for i in range(count)]


def test_the_sample_takes_one_from_each_idea_before_a_second_from_any():
    taken = sample_evenly(rows(a=5, b=5, c=1), 7, random.Random(0))
    counts = {idea: sum(r["idea"] == idea for r in taken) for idea in "abc"}
    assert counts == {"a": 3, "b": 3, "c": 1}


def test_a_model_with_too_few_gives_all_it_has():
    assert len(sample_evenly(rows(a=2, b=3), 60, random.Random(0))) == 5


def test_a_repeated_pair_is_read_once_and_counts_for_each_model():
    one = {"idea": "a", "factor": "ts_mean(close, 5)", "file": "x", "faithful": True}
    two = {"idea": "a", "factor": "ts_mean(close,5)", "file": "y", "faithful": False}
    other = {"idea": "b", "factor": "ts_mean(close,5)", "file": "y", "faithful": None}
    items = read_once({"S2": [one], "S4 arm B": [two, other]}, random.Random(0))
    assert len(items) == 2
    merged = next(i for i in items if i["idea"] == "a")
    assert [s["model"] for s in merged["from"]] == ["S2", "S4 arm B"]
    models = by_model([{**i, "n": n} for n, i in enumerate(items)], {n: True for n in range(2)})
    assert len(models["S2"]) == 1 and len(models["S4 arm B"]) == 2


def test_the_difference_of_shares_has_an_interval_round_it():
    diff, low, high = difference([True] * 40 + [False] * 10, [True] * 20 + [False] * 30)
    assert diff == pytest.approx(0.4)
    assert low < 0.4 < high and low > 0


def test_exactly_ten_points_below_is_within_the_rule_and_one_formula_more_is_not():
    earlier = [True] * 44 + [False] * 11
    assert within_ten_points([True] * 42 + [False] * 18, earlier)
    assert not within_ten_points([True] * 41 + [False] * 19, earlier)


def test_the_per_idea_difference_uses_only_ideas_both_have():
    a = [{"idea": "x", "faithful": True}, {"idea": "x", "faithful": False}, {"idea": "only", "faithful": True}]
    b = [{"idea": "x", "faithful": False}]
    diff, _, _, count = same_ideas(a, b)
    assert count == 1 and diff == pytest.approx(0.5)
