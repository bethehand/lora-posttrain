"""The rules of prereg 04: who may read the ideas, how answers are merged, kept and compared."""
import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "factor"))

from second_exam import e5_passes, frozen_problems, gap, hypotheses, keep_one, merged, per_idea, pinned, roster, whole, within_correlation, within_points  # noqa: E402


def answer(idea="a", factor="close", passed=True, ic=0.01, research=0.02, **extra):
    steps = []
    if research is not None:
        steps = [
            {"role": "assistant", "calls": [json.dumps({"expression": factor})]},
            {"role": "tool", "content": f"computes: yes\nincremental IC after removing known styles: {research:+.4f} (t = 2.0)"},
        ]
    return {"idea": idea, "factor": factor, "matches_idea": passed, "ic": ic, "raw_ic": ic, "reward_ic": 0.5 if passed else 0.0, "tests": len(steps) // 2, "steps": steps, **extra}


def row(idea="a", passed=True, research=0.02, ic=0.01, answered=True, tests=2):
    """A merged row: `passed` is the gates' verdict, and the year's fields are the score."""
    return {"idea": idea, "answered": answered, "tests": tests, "passed": passed, "research": research, "faithful": None, "pass26": passed, "ic26": ic if passed else None, "raw26": ic if passed else None, "reward26": 0.5 if passed else 0.0, "pass25": passed, "ic25": ic if passed else None, "raw25": ic if passed else None, "reward25": 0.5 if passed else 0.0}


def test_two_seeds_share_a_model_and_a_passing_run_joins():
    table = roster(["factor-p1-s2-ic"])
    assert table["S4 B"] == ["s4-ic-seed1", "s4-ic-seed2"]
    assert table["p1-s2-ic"] == ["p1-s2-ic"]
    assert table["S3 B"] == ["s3p-ic"]
    assert table["DeepSeek"] == ["deepseek"]


def test_no_model_reads_the_ideas_until_the_prereg_is_frozen_with_their_hash():
    assert frozen_problems("Status: DRAFT abc", "abc", []) == ["prereg 04 is not frozen"]
    assert frozen_problems("Status: FROZEN def", "abc", []) == ["the ideas file is not the one pinned"]
    assert frozen_problems("Status: FROZEN abc", "abc", ["factor-p1-s2-ic"]) == ["factor-p1-s2-ic is not named in prereg 04"]
    assert frozen_problems("Status: FROZEN abc, with factor-p1-s2-ic", "abc", ["factor-p1-s2-ic"]) == []


def test_an_adapter_must_be_pinned_under_its_own_name():
    pins = "- s2: `abc`, from x\n- s1: `def`, from y\n"
    assert pinned(pins, "s2", "abc")
    assert not pinned(pins, "s2", "def")
    assert not pinned(pins, "s3", "abc")
    assert not pinned(pins, "s", "abc")


def test_a_later_line_for_the_same_name_replaces_an_earlier_one():
    pins = "- p2-s2-rank: `abc`, from x\n- p2-s2-rank: `def`, from y\n"
    assert pinned(pins, "p2-s2-rank", "def")
    assert not pinned(pins, "p2-s2-rank", "abc")


def test_a_file_is_whole_only_with_every_rollout_and_no_failed_episode():
    good = [{"idea": i} for i in "ab" for _ in range(2)]
    assert whole(good, {"a", "b"}, 2)
    assert not whole(good[:-1], {"a", "b"}, 2)
    assert not whole(good, {"a", "b", "c"}, 2)
    assert not whole(good + [{"idea": None}], {"a", "b"}, 2)


def test_merging_keeps_each_years_score_and_what_a_process_could_see():
    before = [answer(ic=0.03, research=0.04), answer(idea="b", factor="", passed=False, ic=None, research=None)]
    after = [answer(ic=-0.01, research=0.04), answer(idea="b", factor="", passed=False, ic=None, research=None)]
    one, none = merged(before, after)
    assert one["research"] == 0.04 and one["passed"]
    assert one["ic25"] == 0.03 and one["ic26"] == -0.01
    assert not none["answered"] and not none["passed"] and none["ic25"] is None


def test_a_formula_with_no_coins_left_passes_the_gates_but_not_the_year():
    before = [answer(passed=False, ic=0.03, reason="too_few_coins_or_constant_when_scored")]
    after = [answer(passed=True, ic=0.05)]
    row_ = merged(before, after)[0]
    assert row_["passed"] and not row_["pass25"] and row_["ic25"] is None
    assert row_["pass26"] and row_["ic26"] == 0.05


def test_a_failure_on_a_year_has_no_ic_on_that_year():
    row_ = merged([answer(ic=0.03)], [answer(passed=False, ic=0.05, reason="too_few_coins_or_constant_when_scored")])[0]
    assert row_["passed"] and not row_["pass26"] and row_["ic26"] is None and row_["ic25"] == 0.03


def test_the_two_years_must_hold_the_same_answers():
    with pytest.raises(ValueError):
        merged([answer(factor="close")], [answer(factor="open")])
    with pytest.raises(ValueError):
        merged([answer()], [])


def test_an_over_limit_answer_does_not_pass_the_gates():
    assert not merged([answer(passed=False, over_limit=True)], [answer(passed=False, over_limit=True)])[0]["passed"]


def test_ic_averages_over_passes_and_pass_rate_over_episodes():
    rows = [row(ic=0.02), row(ic=0.04), row(passed=False), row(idea="b", passed=False)]
    assert per_idea(rows, "ic26") == pytest.approx({"a": 0.03})
    assert per_idea(rows, "pass26") == pytest.approx({"a": 2 / 3, "b": 0.0})


def test_a_gap_uses_only_ideas_both_sides_have():
    assert gap({"a": 0.3, "b": 0.5}, {"a": 0.1, "c": 9.0})[0] == pytest.approx(0.2)
    assert gap({"a": 0.3, "b": 0.5}, {"a": 0.1, "b": 0.1}, only={"b"})[0] == pytest.approx(0.4)
    assert gap({"a": 0.3}, {"b": 0.1})[3] == 0


def test_one_episode_takes_each_episode_once():
    rows = [row(ic=0.01), row(ic=0.02), row(passed=False)]
    assert len(keep_one(rows, 1)) == 3


def test_keeping_goes_by_research_ic_and_never_by_the_later_score():
    best_later = row(research=0.01, ic=0.09)
    best_research = row(research=0.03, ic=-0.09)
    rows = [best_later, best_research]
    kept = keep_one(rows, 2)
    assert [r["ic26"] for r in kept] == [-0.09]
    assert kept[0]["spent"] == 4


def test_an_answer_that_fails_the_gates_or_was_not_tested_is_passed_over():
    fails = row(passed=False, research=0.09)
    untested = row(research=None, ic=0.05)
    good = row(research=0.01, ic=0.02)
    assert keep_one([fails, untested, good], 3)[0]["ic26"] == 0.02


def test_with_nothing_tested_it_keeps_an_answer_and_with_no_answer_it_fails():
    answered = row(research=None, ic=0.05)
    silent = row(passed=False, research=None, answered=False)
    assert keep_one([answered, silent], 2)[0]["ic26"] == 0.05
    assert not keep_one([silent, silent], 2)[0]["pass26"]


def test_draws_are_repeatable_and_k_of_n_draws_as_many_as_asked():
    rows = [row(research=r / 100, ic=r / 100) for r in range(8)]
    assert keep_one(rows, 4) == keep_one(rows, 4)
    kept = keep_one(rows, 4)
    assert len(kept) == 2000
    assert np.mean([r["ic26"] for r in kept]) > np.mean([r["ic26"] for r in rows])


def test_within_correlation_ignores_differences_between_groups():
    points = {"a": [[(0.0, 10.0), (1.0, 11.0)], [(5.0, -3.0), (6.0, -2.0)]], "b": [[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]]}
    r, low, high, pairs = within_correlation(points)
    assert r == pytest.approx(1.0) and low == pytest.approx(1.0) and high == pytest.approx(1.0)
    assert pairs == 7


def test_within_correlation_skips_a_group_of_one_and_sees_a_negative():
    points = {"a": [[(0.0, 1.0), (1.0, 0.0)], [(9.0, 9.0)]], "b": [[(0.0, 2.0), (2.0, 0.0)]]}
    r, _, _, pairs = within_correlation(points)
    assert r == pytest.approx(-1.0) and pairs == 4


def test_within_correlation_with_no_pairs_is_undefined_and_so_cannot_pass():
    r, low, _, pairs = within_correlation({"a": [[(0.0, 1.0)]]})
    assert np.isnan(r) and not low > 0.3 and pairs == 0


def test_within_points_counts_only_tested_passes_of_one_model_on_one_idea():
    models = {"x": [row(), row(passed=False), row(research=None)], "y": [row(idea="b")]}
    points = within_points(models, "26")
    assert [len(g) for g in points["a"]] == [1]
    assert [len(g) for g in points["b"]] == [1]
    assert within_points(models, "26", only={"b"}).keys() == {"b"}


def test_e5_needs_both_better_formulas_and_most_of_the_pass_rate():
    assert e5_passes(0.0001, 0.0)
    assert e5_passes(0.0001, -0.05)
    assert not e5_passes(0.0, 0.0)
    assert not e5_passes(0.001, -0.06)


def models_where_s2_writes_better_and_b_passes_more():
    ideas = [f"i{n}" for n in range(6)]
    s2, b = [], []
    for n, idea in enumerate(ideas):
        s2 += [row(idea, research=0.01 * (k + 1), ic=0.02 + 0.001 * n + 0.01 * k) for k in range(2)] + [row(idea, passed=False, research=None) for _ in range(2)]
        b += [row(idea, research=0.01 * (k + 1), ic=-0.01 + 0.001 * n + 0.01 * k) for k in range(4)]
    return {"S2": s2, "S4 B": b}


def test_the_hypotheses_read_the_way_the_prereg_writes_them():
    found = {e["id"]: e for e in hypotheses(models_where_s2_writes_better_and_b_passes_more(), [])}
    assert found["E1"]["passes"] and found["E1"]["ic"][0] > 0
    assert found["E2"]["passes"] and found["E2"]["rate"][0] == pytest.approx(0.5)
    assert found["E3"]["passes"] and found["E3"]["ic"][0] > 0
    assert found["E4"]["passes"] and found["E4"]["correlation"][0] == pytest.approx(1.0)
    assert "E5" not in found


def test_a_passing_run_is_tested_against_s4_b():
    models = models_where_s2_writes_better_and_b_passes_more()
    models["p1"] = list(models["S2"])
    found = [e for e in hypotheses(models, ["factor-p1"]) if e["id"] == "E5"]
    assert len(found) == 1 and found[0]["what"] == "p1 minus S4 B"
    assert found[0]["ic"][1] > 0 and found[0]["rate"][0] == pytest.approx(-0.5)
    assert not found[0]["passes"]
