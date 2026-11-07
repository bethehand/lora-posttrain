"""The small rules behind the exploration that followed prereg 03's exam."""
import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "factor"))

from explore_after_exam import answer_research_ic, ic_share, percentile, pick_one, trials, within_slope  # noqa: E402
from pilot_compare import mean_ic_per_idea, passes_rules  # noqa: E402


def test_the_pilot_passes_only_when_all_three_rules_hold():
    assert passes_rules(0.0001, 0.0001, 0.0)
    assert not passes_rules(0.0, 0.001, 0.0)
    assert not passes_rules(0.001, 0.0, 0.0)
    assert not passes_rules(0.001, 0.001, -0.06)


def test_exactly_five_points_lower_on_val_is_within_the_rule():
    assert passes_rules(0.001, 0.001, 0.7917 - 0.8417)


def test_the_pilot_reads_ic_from_passes_with_their_own_answer():
    rows = [
        {"idea": "a", "answered": True, "passed": True, "ic": 0.02},
        {"idea": "a", "answered": True, "passed": True, "ic": 0.04},
        {"idea": "a", "answered": False, "passed": True, "ic": 0.9},
        {"idea": "a", "answered": True, "passed": False, "ic": None},
        {"idea": "b", "answered": True, "passed": False, "ic": None},
    ]
    assert mean_ic_per_idea(rows) == pytest.approx({"a": 0.03})


def reply(expression):
    return {"role": "assistant", "calls": [json.dumps({"expression": expression})]}


def report(ic):
    return {"role": "tool", "content": f"computes: yes\nrank IC: +0.01\nincremental IC after removing known styles: {ic:+.4f} (t = 2.0)\n"}


def test_a_test_that_failed_is_left_out_and_the_rest_keep_their_order():
    steps = [reply("a"), report(0.001), reply("broken"), {"role": "tool", "content": "error: bad"}, reply("c"), report(-0.002)]
    assert trials(steps) == [("a", 0.001), ("c", -0.002)]


def test_an_answer_shows_the_ic_of_its_last_test_whatever_its_spacing():
    tested = [("a + b", 0.001), ("c", 0.005), ("a+b", 0.002)]
    assert answer_research_ic("a  +  b", tested) == 0.002
    assert answer_research_ic("never tested", tested) is None
    assert answer_research_ic(None, tested) is None


def test_with_only_passes_and_failures_ic_has_none_of_the_spread():
    assert ic_share([([0.5, 0.5, 0.0, 0.0], [True, True, False, False])]) == 0.0


def test_when_all_pass_the_whole_spread_is_ic():
    assert ic_share([([0.4, 0.6, 0.5, 0.7], [True] * 4)]) == pytest.approx(1.0)


def test_the_share_adds_groups_by_their_spread():
    groups = [([1.0, 0.0], [True, False]), ([0.4, 0.6], [True, True])]
    assert ic_share(groups) == pytest.approx(0.01 / (0.25 + 0.01))


def test_a_percentile_is_the_share_of_the_table_at_or_below():
    table = np.array([0.0, 0.1, 0.2, 0.3])
    assert percentile(table, -1.0) == 0.0
    assert percentile(table, 0.1) == 0.5
    assert percentile(table, 9.0) == 1.0


def rollout(research, passed=True, faithful=True, answered=True):
    return {"research": research, "passed": passed, "faithful": faithful, "answered": answered}


def test_a_process_keeps_the_best_it_is_allowed_to_see():
    rng = np.random.default_rng(0)
    high_fails = rollout(0.03, passed=False)
    low_passes = rollout(0.01)
    unfaithful = rollout(0.02, faithful=False)
    rows = [high_fails, low_passes, unfaithful]
    assert pick_one(rows, "research", rng) is high_fails
    assert pick_one(rows, "gates", rng) is unfaithful
    assert pick_one(rows, "gates+judge", rng) is low_passes


def test_gates_and_judge_fall_back_to_a_pass_the_judge_rejected():
    only = rollout(0.02, faithful=False)
    assert pick_one([only, rollout(0.05, passed=False)], "gates+judge", np.random.default_rng(0)) is only


def test_an_untested_answer_is_kept_only_when_nothing_tested_is_left():
    rng = np.random.default_rng(0)
    untested = rollout(None)
    silent = rollout(None, passed=False, answered=False)
    assert pick_one([untested, silent], "research", rng) is untested
    assert pick_one([silent], "gates", rng) is silent


def test_a_slope_within_groups_ignores_the_gap_between_groups():
    x = np.array([-1.0, 1.0])
    pairs = {"a": (x, 2 * x), "b": (x, 2 * x), "c": (x, 2 * x)}
    mean, low, high = within_slope(pairs)
    assert mean == pytest.approx(2.0) and low == pytest.approx(2.0) and high == pytest.approx(2.0)
