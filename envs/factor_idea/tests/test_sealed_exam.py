"""The checks the sealed exam of prereg 03 makes before it opens the sealed period, and its per-idea mean."""
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "factor"))

from sealed_exam import differences, per_idea  # noqa: E402


def row(passes=True, ic=0.01, raw=0.02):
    return {"matches_idea": passes, "ic": ic, "raw_ic": raw}


def test_a_rehearsal_that_gives_back_the_file_finds_nothing():
    assert differences([row(), row(False, None, None)], [row(), row(False, None, None)]) == []


def test_a_changed_verdict_or_ic_keeps_the_sealed_period_closed():
    assert differences([row(False)], [row()]) == ["row 0: verdict"]
    assert differences([row(ic=0.011)], [row()]) == ["row 0: ic 0.011 against 0.01"]
    assert differences([row(raw=None)], [row()]) == ["row 0: raw_ic None against 0.02"]
    assert differences([row()], [row(), row()]) == ["1 rows against 2"]


def test_each_idea_counts_once_and_only_passes_count():
    rows = [
        {"idea": "a", "matches_idea": True, "ic": 0.01},
        {"idea": "a", "matches_idea": True, "ic": 0.03},
        {"idea": "a", "matches_idea": False, "ic": 0.9},
        {"idea": "b", "matches_idea": True, "ic": -0.01},
        {"idea": "c", "matches_idea": False, "ic": None},
    ]
    assert sorted(per_idea(rows, "ic")) == pytest.approx([-0.01, 0.02])
