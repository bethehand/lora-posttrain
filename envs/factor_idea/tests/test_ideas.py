"""The idea bank: well formed, its reference formulas parse, and no family straddles the split."""
import json
import pathlib
import sys

from factor_idea import dsl, sandbox as sb

ROWS = [json.loads(line) for line in open(pathlib.Path(__file__).resolve().parents[3] / "task" / "ideas_v2.jsonl")]


def test_ids_are_unique():
    assert len({r["id"] for r in ROWS}) == len(ROWS)


def test_every_row_is_complete():
    for r in ROWS:
        assert r["key"] in sb.KEYS and r["key"] in sb.KEY_GROUPS, r["id"]
        assert r["direction"] in (1, -1), r["id"]
        assert r["split"] in ("train", "val", "heldout"), r["id"]
        assert r["en"].strip() and r["zh"].strip() and r["source"].strip(), r["id"]
        assert all(g in set(sb.KEY_GROUPS.values()) for g in r.get("also", ())), r["id"]


def test_reference_formulas_parse_and_use_daily_bars():
    for r in ROWS:
        assert not dsl.reads_minutes(dsl.parse(r["reference"])), r["id"]


def test_no_family_straddles_the_split():
    held = {r["family"] for r in ROWS if r["split"] == "heldout"}
    train = {r["family"] for r in ROWS if r["split"] == "train"}
    assert held and train and not held & train


def test_every_training_family_gives_one_idea_to_validation():
    val = [r["family"] for r in ROWS if r["split"] == "val"]
    train = {r["family"] for r in ROWS if r["split"] == "train"}
    assert len(val) == len(set(val)) and set(val) == train


def test_the_excluded_factor_stays_out():
    text = json.dumps(ROWS, ensure_ascii=False)
    assert "neg_ret_disp" not in text and "dispersion_down" not in text


def test_the_split_column_is_what_the_seeds_draw():
    root = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "scripts" / "factor"))
    from draw_splits import draw

    rows = [json.loads(line) for line in open(root / "task" / "ideas_v2.jsonl")]
    assert draw(rows) == {r["id"]: r["split"] for r in rows}
