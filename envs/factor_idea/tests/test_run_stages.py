"""The rules that choose rank, steps and when to stop online RL, as prereg 03 states them."""
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "factor"))

from heldout_online import mirrored, unsigned  # noqa: E402
from run_stages import earliest_within, guard_trips, half_epoch, needs_rank_128, pick_rank, reward_falls, summarize  # noqa: E402


def test_a_formula_loses_only_its_sign_and_the_brackets_round_it():
    assert unsigned("-1 * (a + b)") == "a+b"
    assert unsigned("-(a)") == "a"
    assert unsigned("(a) * (b)") == "(a)*(b)"
    assert unsigned("-1 * a.div(b)") == "a.div(b)"


def test_sixteen_held_out_ideas_are_in_mirrored_pairs_as_prereg_03_says():
    with open(REPO / "task" / "ideas_v2.jsonl") as f:
        ideas = [json.loads(line) for line in f]
    assert len(mirrored(ideas)) == 16


def test_the_earliest_checkpoint_close_to_the_best_wins():
    assert earliest_within({70: 0.20, 140: 0.46, 210: 0.50, 280: 0.40}, 0.05) == 140
    assert earliest_within({70: 0.50, 140: 0.60}, 0.05) == 140
    assert earliest_within({70: 0.60, 140: 0.60}, 0.0) == 70


def test_the_smallest_rank_close_to_the_best_wins():
    assert pick_rank({8: 0.46, 32: 0.50}) == 8
    assert pick_rank({8: 0.40, 32: 0.50}) == 32
    assert pick_rank({8: 0.40, 32: 0.50, 128: 0.54}) == 32
    assert pick_rank({8: 0.40, 32: 0.50, 128: 0.60}) == 128


def test_rank_128_runs_only_if_going_from_8_to_32_clearly_helped():
    assert needs_rank_128({8: 0.40, 32: 0.50})
    assert not needs_rank_128({8: 0.46, 32: 0.50})
    assert not needs_rank_128({8: 0.60, 32: 0.50})


def step(ent=1.0, kl=0.001):
    return {"ent": ent, "kl": kl}


def test_one_step_out_of_bounds_is_not_enough():
    assert guard_trips({1: step(kl=0.05), 2: step(), 3: step(ent=9.0), 4: step()}) is None


def test_two_steps_running_out_of_bounds_trip_the_guard():
    assert "mismatch_kl" in guard_trips({1: step(), 2: step(kl=0.02), 3: step(kl=0.02)})
    assert "entropy" in guard_trips({4: step(ent=0.1), 5: step(ent=0.2)})


def test_the_guard_only_watches_the_first_ten_steps():
    assert guard_trips({11: step(kl=0.05), 12: step(kl=0.05)}) is None
    assert guard_trips({10: step(kl=0.05), 11: step(kl=0.05)}) is None


def rewards(*values):
    return {step: {"reward": value} for step, value in enumerate(values, start=1)}


def test_five_steps_of_lost_reward_trip_the_guard():
    assert "steps 4 to 8" in reward_falls(rewards(0.6, 0.6, 0.6, 0.3, 0.3, 0.3, 0.3, 0.3))


def test_the_reward_guard_watches_the_whole_run():
    assert "steps 30 to 34" in reward_falls(rewards(*[0.6] * 30, *[0.3] * 5))


def test_a_short_bad_stretch_or_a_small_fall_does_not_trip_the_reward_guard():
    assert reward_falls(rewards(0.6, 0.6, 0.6, 0.3, 0.3, 0.6, 0.6, 0.6, 0.6)) is None
    assert reward_falls(rewards(0.6, 0.6, 0.6, *[0.45] * 10)) is None


def test_the_reward_guard_waits_for_its_starting_steps():
    assert reward_falls(rewards(0.6, 0.6)) is None
    assert reward_falls({2: {"reward": 0.6}, 3: {"reward": 0.6}, **{s: {"reward": 0.1} for s in range(4, 12)}}) is None


def test_a_half_epoch_follows_the_size_of_the_data():
    assert half_epoch(2021) == 70
    assert half_epoch(206) == 7
    assert half_epoch(1) == 1


def test_a_failed_episode_counts_against_the_stage():
    rows = [
        {"idea": "a", "matches_idea": True, "ic": 0.01, "reward_ic": 0.8, "factor": "x", "tests": 3, "faithful": True},
        {"idea": "a", "matches_idea": True, "ic": 0.03, "reward_ic": 1.0, "factor": "y", "tests": 5, "faithful": False},
        {"idea": "b", "matches_idea": False, "factor": "z", "tests": 5},
        {"idea": None},
    ]
    result = summarize(rows)
    assert result["episodes"] == 4 and result["failed"] == 1 and result["pass"] == 0.5 and result["answered"] == 0.75
    assert result["reward_ic"] == pytest.approx(0.45) and result["reward_judged"] == pytest.approx(0.2)
    assert result["ic_among_passes"] == pytest.approx(0.02) and result["tests"] == pytest.approx(13 / 4)
