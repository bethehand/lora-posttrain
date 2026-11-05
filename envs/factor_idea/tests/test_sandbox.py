"""The checks on a synthetic market where the right answers are known."""
import numpy as np
import pytest

from factor_idea import sandbox as sb
from factor_idea.dsl import FIELDS, parse
from factor_idea.evaluate import evaluate
from factor_idea.panel import Panel
from helpers import synthetic_fields

T = 1200  # rows from 2021-12-28 through early 2025
N = 80


def make_panel(seed=5, drift=None):
    fields = synthetic_fields(T, N, seed=seed)
    if drift is not None:
        fields = drift(fields)
    dates = np.datetime64("2021-12-28") + np.arange(T)
    universe = np.ones((T, N), dtype=bool)
    universe[:30] = False
    return Panel(dates.astype("datetime64[D]"), np.arange(N).astype(str), np.arange(N).astype(str), fields, universe)


@pytest.fixture(scope="module")
def box():
    return sb.Sandbox(make_panel())


def verdict(box, text, key, direction, also=()):
    node = parse(text)
    with np.errstate(all="ignore"):
        memo = {}
        base = evaluate(node, box.gate_panel.fields, memo=memo)
        return box.check_compute(base), box.check_idea(node, key, direction, base, also=also, memo=memo)


RET = '(Factor("close").safe_div(Factor("close").lag(1)) - 1)'

GOOD = [
    ('Factor("quote_volume")', "volume", 1),
    ('Factor("quote_volume").ts_mean(5)', "volume", 1),
    ('Factor("quote_volume").ts_mean(5) / Factor("quote_volume").ts_mean(60)', "volume", 1),
    ('Factor("quote_volume").ln().sec_rank().EMA(span=5)', "volume", 1),
    ('Factor("volume") / Factor("volume").ts_mean(20)', "volume", 1),
    ('-1 * Factor("quote_volume").ts_mean(window=20, min_periods=16).ln()', "volume", -1),
    ('Factor("num_trades").ts_mean(10)', "trades", 1),
    ('(Factor("quote_volume") / Factor("num_trades")).ts_mean(10)', "trade_size", 1),
    (f"{RET}", "return", 1),
    ('Factor("close") / Factor("close").lag(20) - 1', "return", 1),
    ('-1 * (Factor("close") / Factor("close").lag(5) - 1)', "return", -1),
    ('Factor("close") / Factor("close").ts_max(60)', "return", 1),
    ('(Factor("close") / Factor("open") - 1).ts_mean(5)', "return", 1),
    (f"{RET}.ts_std(20)", "volatility", 1),
    (f"{RET}.Abs().ts_mean(10)", "volatility", 1),
    ('(Factor("high") / Factor("low")).ln().ts_mean(10)', "volatility", 1),
    ('-1 * (Factor("high") / Factor("low") - 1).ts_mean(20)', "volatility", -1),
    ('(Factor("taker_quote_vol") / Factor("quote_volume")).ts_mean(5)', "taker_buy", 1),
    ('(2 * Factor("taker_base_vol") - Factor("volume")) / Factor("volume")', "taker_buy", 1),
    ('((Factor("close") - Factor("low")) / (Factor("high") - Factor("low"))).ts_mean(5)', "close_near_high", 1),
    ('-1 * ((Factor("high") - Factor("close")) / Factor("close")).ts_mean(5)', "close_near_high", 1),
]

ILLIQUIDITY = [
    f"({RET}.Abs() + 1).ln().safe_div(Factor(\"quote_volume\")).ts_mean(6)",
    f"{RET}.Abs().safe_div(Factor(\"quote_volume\"))",
    '(Factor("high") / Factor("low") - 1).safe_div(Factor("quote_volume")).ts_mean(20)',
]

WRONG_DIRECTION = [(text, key, -d) for text, key, d in GOOD]

IGNORES_THE_KEY = [
    (f"{RET}.ts_std(20)", "volume", 1),
    ('(Factor("close") / Factor("close").lag(20) - 1)', "volume", 1),
    ('(Factor("taker_quote_vol") / Factor("quote_volume")).ts_mean(5)', "volume", 1),
    ('Factor("quote_volume").ts_mean(5)', "return", 1),
    ('Factor("quote_volume").ts_mean(5)', "volatility", 1),
    ('Factor("quote_volume").ts_mean(5)', "taker_buy", 1),
    ('Factor("num_trades").ts_mean(5)', "trade_size", 1),
]

UNIT_DEPENDENT = [
    ('Factor("close")', "return", 1),
    ('Factor("close").ln()', "return", 1),
    ('Factor("close").diff(5)', "return", 1),
    ('Factor("volume")', "volume", 1),
    ('Factor("volume").ts_mean(5)', "volume", 1),
    ('Factor("high") - Factor("low")', "volatility", 1),
    ('Factor("close").ts_std(20)', "volatility", 1),
]

CANNOT_COMPUTE = [
    'Factor("close") / Factor("close")',
    'Factor("close").ts_std(window=1, min_periods=1)',
    'Factor("close").lag(365).lag(365).lag(365).lag(365)',
    'Factor("close").sign()',
    '(Factor("close") > 0) * 1',
]


@pytest.mark.parametrize("text,key,direction", GOOD)
def test_honest_factors_pass(box, text, key, direction):
    compute, idea = verdict(box, text, key, direction)
    assert compute.ok
    assert idea.ok, (idea.reason, idea.windows, idea.unit_median)


@pytest.mark.parametrize("text", ILLIQUIDITY)
def test_an_idea_can_declare_a_second_kind_of_data(box, text):
    _, idea = verdict(box, text, "volume", -1, also=("price",))
    assert idea.ok, (idea.reason, idea.windows, idea.unrelated)


@pytest.mark.parametrize("text,key,direction", WRONG_DIRECTION)
def test_the_wrong_direction_fails(box, text, key, direction):
    _, idea = verdict(box, text, key, direction)
    assert not idea.ok
    assert idea.reason == "moves_against_the_idea"


@pytest.mark.parametrize("text,key,direction", IGNORES_THE_KEY)
def test_a_factor_that_ignores_the_key_fails(box, text, key, direction):
    _, idea = verdict(box, text, key, direction)
    assert not idea.ok
    assert idea.reason == "does_not_respond_to_the_idea"


@pytest.mark.parametrize("text,key,direction", UNIT_DEPENDENT)
def test_a_factor_that_depends_on_price_units_fails(box, text, key, direction):
    _, idea = verdict(box, text, key, direction)
    assert not idea.ok
    # A raw level across coins can be so spread out that the perturbation barely moves its rank.
    assert idea.reason in ("depends_on_price_units", "does_not_respond_to_the_idea")


@pytest.mark.parametrize("text", CANNOT_COMPUTE)
def test_degenerate_factors_fail_to_compute(box, text):
    compute, _ = verdict(box, text, "volume", 1)
    assert not compute.ok


def test_perturbations_keep_each_bar_consistent(box):
    panel = box.gate_panel
    for key in sb.KEYS:
        for w in sb.WINDOWS:
            f = box._perturbed(key, w)
            with np.errstate(invalid="ignore"):
                assert not np.nansum(f["high"] < f["low"] * (1 - 1e-9)), (key, w)
                assert not np.nansum(f["close"] > f["high"] * (1 + 1e-9)), (key, w)
                assert not np.nansum(f["close"] < f["low"] * (1 - 1e-9)), (key, w)
                assert not np.nansum(f["open"] > f["high"] * (1 + 1e-9)), (key, w)
                if key != "return":  # a rally leaves the first bar's low above its open, on purpose
                    assert not np.nansum(f["open"] < f["low"] * (1 - 1e-9)), (key, w)
                assert not np.nansum(f["taker_base_vol"] > f["volume"] * (1 + 1e-9)), (key, w)
                assert not np.nansum(f["taker_quote_vol"] > f["quote_volume"] * (1 + 1e-9)), (key, w)
            changed = set(sb.KEYS[key][0])
            for name in FIELDS:
                if name not in changed:
                    assert f[name] is panel.fields[name]


def test_perturbations_do_not_touch_the_panel(box):
    panel = box.gate_panel
    before = {k: v.copy() for k, v in panel.fields.items()}
    for key in sb.KEYS:
        box._perturbed(key, 20)
    box._unit_changed()
    for k, v in before.items():
        np.testing.assert_array_equal(panel.fields[k], v)


def test_each_perturbation_changes_only_the_planned_cells(box):
    panel = box.gate_panel
    f = box._perturbed("volume", 5)
    expected = np.zeros(panel.fields["volume"].shape, dtype=bool)
    for t, coins in box.plan:
        expected[np.ix_(np.arange(t - 4, t + 1), coins)] = True
    with np.errstate(invalid="ignore"):
        moved = f["volume"] != panel.fields["volume"]
    np.testing.assert_array_equal(moved & np.isfinite(panel.fields["volume"]), expected & np.isfinite(panel.fields["volume"]))


OMNIBUS = " + ".join(
    f"({text}).sec_rank()"
    for text in (
        'Factor("quote_volume").ts_mean(window=20, min_periods=16).ln()',
        'Factor("num_trades").ts_mean(window=20, min_periods=16).ln()',
        '(Factor("close") / Factor("close").lag(5) - 1)',
        f"{RET}.ts_std(20)",
        '(Factor("taker_quote_vol") / Factor("quote_volume")).ts_mean(5)',
        '((Factor("close") - Factor("low")) / (Factor("high") - Factor("low"))).ts_mean(5)',
    )
)


def test_one_factor_cannot_answer_every_idea(box):
    passes = {key: verdict(box, OMNIBUS, key, 1)[1].ok for key in sb.KEYS}
    assert sum(passes.values()) <= 2, passes


@pytest.mark.parametrize(
    "text,key",
    [
        (f'({RET}.ts_std(20)).sec_rank() + 0.3 * Factor("quote_volume").ts_mean(5).ln().sec_rank()', "volume"),
        ('(Factor("quote_volume").ts_mean(5).ln()).sec_rank() + 0.1 * (Factor("close") / Factor("close").lag(5)).sec_rank()', "return"),
    ],
)
def test_a_small_dose_of_the_key_fails(box, text, key):
    _, idea = verdict(box, text, key, 1)
    assert not idea.ok


def test_total_volume_is_not_trade_size(box):
    _, idea = verdict(box, 'Factor("quote_volume").ts_mean(5)', "trade_size", 1)
    assert not idea.ok
    assert idea.reason == "measures_volume_not_trade_size"


def test_trade_size_with_volume_regressed_out_is_still_trade_size(box):
    text = '(Factor("quote_volume") / Factor("num_trades")).ln().ts_mean(10).neutralize(Factor("quote_volume").ts_mean(10).ln())'
    _, idea = verdict(box, text, "trade_size", 1)
    assert idea.reason != "measures_volume_not_trade_size"


def test_a_range_formula_is_not_a_return_formula(box):
    _, idea = verdict(box, 'Factor("high") / Factor("low")', "return", 1)
    assert not idea.ok


def signal_market(fields):
    """Plant a known predictive signal: a coin's taker share today sets its return two days ahead."""
    rng = np.random.default_rng(99)
    share = fields["taker_quote_vol"] / fields["quote_volume"]
    r = rng.standard_t(4, size=share.shape) * 0.03
    planted = np.nan_to_num(share - 0.5, nan=0.0)
    r[2:] += 0.15 * planted[:-2]
    close = np.exp(np.cumsum(r, axis=0)) * 5.0
    out = dict(fields)
    out["close"] = np.where(np.isfinite(fields["close"]), close, np.nan)
    out["open"] = np.vstack([out["close"][0], out["close"][:-1]])
    out["high"] = np.fmax(out["open"], out["close"]) * 1.01
    out["low"] = np.fmin(out["open"], out["close"]) * 0.99
    return out


def test_ic_finds_a_planted_signal_and_nothing_in_noise():
    box = sb.Sandbox(make_panel(drift=signal_market))
    panel = box.dev
    first, last = str(panel.dates[100]), str(panel.dates[-3])
    with np.errstate(all="ignore"):
        signal = evaluate(parse('Factor("taker_quote_vol") / Factor("quote_volume")'), panel.fields)
        noise = evaluate(parse('Factor("num_trades") / Factor("quote_volume")'), panel.fields)
    good = box.incremental_ic(signal, panel, first, last)
    assert good.incremental > 0.2
    assert good.t_stat > 5
    bad = box.incremental_ic(noise, panel, first, last)
    assert abs(bad.incremental) < 0.03


def test_a_future_peeking_series_would_score_impossibly_high():
    """Calibration of the IC ruler: the true next-day return, if it ever leaked, would stand out."""
    box = sb.Sandbox(make_panel())
    panel = box.dev
    fwd, _ = box._ic_arrays(panel)
    result = box.incremental_ic(fwd, panel, str(panel.dates[100]), str(panel.dates[-3]))
    assert result.raw == pytest.approx(1.0)


def test_removing_styles_kills_a_style_copy():
    box = sb.Sandbox(make_panel())
    panel = box.dev
    with np.errstate(all="ignore"):
        copy = evaluate(parse(sb.STYLES["reversal_1d"]), panel.fields)
    result = box.incremental_ic(copy, panel, str(panel.dates[100]), str(panel.dates[-3]))
    assert abs(result.incremental) < 1e-9


def test_a_near_copy_of_a_style_earns_almost_nothing():
    """A tiny difference from a style must not be blown up to full size."""
    box = sb.Sandbox(make_panel(drift=signal_market))
    panel = box.dev
    first, last = str(panel.dates[100]), str(panel.dates[-3])
    with np.errstate(all="ignore"):
        style = evaluate(parse(sb.STYLES["reversal_1d"]), panel.fields)
        signal = evaluate(parse('Factor("taker_quote_vol") / Factor("quote_volume")'), panel.fields)
    alone = box.incremental_ic(signal, panel, first, last).incremental
    near_copy = box.incremental_ic(style + 1e-3 * (signal - 0.5), panel, first, last).incremental
    assert abs(near_copy) < 0.1 * alone


def test_score_and_rewards(box):
    good = box.score('Factor("quote_volume").ts_mean(5) / Factor("quote_volume").ts_mean(60)', "volume", 1)
    assert good.parsed and good.computes and good.matches_idea and good.ic is not None
    assert good.reward("gates") == 1.0
    assert 0.3 <= good.reward("ic") <= 1.0

    bad_parse = box.score('Factor("close").lag(-1)', "return", 1)
    assert not bad_parse.parsed and bad_parse.reward("gates") == 0.0 and bad_parse.reward("ic") == 0.0
    assert "positive integer" in bad_parse.reason

    flat = box.score('Factor("close") / Factor("close")', "return", 1)
    assert flat.parsed and not flat.computes and flat.reward("ic") == 0.0

    wrong = box.score('-1 * Factor("quote_volume").ts_mean(5)', "volume", 1)
    assert wrong.computes and not wrong.matches_idea and wrong.ic is None and wrong.reward("ic") == 0.0


def test_reward_scale():
    s = sb.Score(parsed=True, computes=True, matches_idea=True)
    for ic, want in [(-0.1, 0.3), (-0.03, 0.3), (0.0, 0.65), (0.015, 0.825), (0.03, 1.0), (0.2, 1.0)]:
        s.ic = sb.ICResult(ic, ic, 300, 1.0)
        assert s.reward("ic") == pytest.approx(want)
    s.ic = sb.ICResult(float("nan"), float("nan"), 0, float("nan"))
    assert s.reward("ic") == pytest.approx(0.3)
    with pytest.raises(ValueError):
        s.reward("nope")


def test_too_slow_is_a_failure_not_a_crash(box, monkeypatch):
    monkeypatch.setattr(sb, "EVAL_SECONDS", -1.0)
    result = box.score('Factor("close").ts_mean(5)', "return", 1)
    assert result.parsed and not result.computes and result.reason == "too_slow"
    assert result.reward("ic") == 0.0


def test_the_seeds_from_the_research_system_parse():
    import json
    import pathlib

    seeds = pathlib.Path(__file__).resolve().parents[3] / "task" / "seeds_research_system.jsonl"
    for line in seeds.read_text().splitlines():
        parse(json.loads(line)["formula"])


@pytest.mark.parametrize(
    "reply,want",
    [
        ('<factor>Factor("close")</factor>', 'Factor("close")'),
        ('think <factor>a</factor> then <factor>b</factor>', "b"),
        ('<factor>a\n\nSo:\n\n<factor>b </factor>', "b"),
        ('<FACTOR>a</FACTOR>', "a"),
        ('<factor>\n```python\nFactor("close")\n```\n</factor>', 'Factor("close")'),
        ('<factor>`Factor("close")`</factor>', 'Factor("close")'),
        ('no tag here', None),
        ('<factor></factor>', None),
        ('<factor>only an opening tag', None),
        ('', None),
    ],
)
def test_reading_the_answer(reply, want):
    from factor_idea.prompt import extract_factor

    assert extract_factor(reply) == want


def test_report_says_what_a_model_may_see(box):
    text = box.report('Factor("quote_volume").ts_mean(5) / Factor("quote_volume").ts_mean(60)')
    assert text.startswith("computes: yes, 2022-07-01 to 2024-12-29")
    assert "rank IC:" in text and "incremental IC after removing known styles:" in text
    assert "2022H2" in text and "2023" in text and "2024" in text
    assert "2025" not in text


def test_report_gives_the_parser_error_back(box):
    assert box.report('Factor("close").ts_ema(5)') == "error: unsupported method: ts_ema"
    assert box.report('Factor("close").lag(-1)').startswith("error:")
    assert box.report("").startswith("error:")


def test_report_says_when_a_factor_does_not_compute(box):
    assert box.report('Factor("close") / Factor("close")').startswith("does not compute")


def test_report_ignores_rows_after_2024(box):
    """The tool must not be a way to read the reward period."""
    text = 'Factor("quote_volume").ts_mean(5) / Factor("quote_volume").ts_mean(60)'
    before = box.report(text)
    start = box.dev.row("2025-01-01")
    poisoned = dict(box.dev.fields)
    for name in poisoned:
        poisoned[name] = poisoned[name].copy()
        poisoned[name][start:] = 1e15
    other = sb.Sandbox(Panel(box.dev.dates, box.dev.sec_ids, box.dev.symbols, poisoned, box.dev.universe))
    assert other.report(text) == before


def test_the_tool_prompt_uses_quotes_a_tool_call_can_hold():
    from factor_idea.prompt import SYSTEM_PROMPT, TOOL_SYSTEM_PROMPT

    assert 'Factor("' in SYSTEM_PROMPT
    assert 'Factor("' not in TOOL_SYSTEM_PROMPT
    assert "test_factor" in TOOL_SYSTEM_PROMPT and "<factor>" in TOOL_SYSTEM_PROMPT


def test_the_example_in_the_prompt_is_valid():
    parse("Factor('close').safe_div(Factor('close').lag(periods=5)) - 1")
