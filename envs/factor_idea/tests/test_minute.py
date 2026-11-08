"""The Minute(...) layer: its grammar, its operators, and that a day only sees its own minutes."""
import random

import numpy as np
import pytest

from factor_idea import dsl, ops, sandbox as sb
from factor_idea.dsl import DslError, parse
from factor_idea.evaluate import NoMinuteData, evaluate
from factor_idea.minutes import MinuteBars, PatchedMinutes, market_series, minute_returns
from factor_idea.panel import Panel
from helpers import synthetic_minutes

T, N, M = 120, 20, 144
BARS, DAILY = synthetic_minutes(T, N, M, seed=3)


def run(text, bars=BARS, fields=DAILY):
    with np.errstate(all="ignore"):
        return evaluate(parse(text), fields, minutes=bars)


def fresh(bars=BARS, **replaced):
    """A copy of the bars with an empty cache and some fields replaced."""
    return MinuteBars(bars.rows, bars.coins, bars.usable, {**bars.fields, **replaced}, bars.market, bars.shape)


Q = "Minute('quote_volume')"
R = "Minute('return')"

GOOD_SYNTAX = [
    f"{Q}.day_sum()",
    f"{Q}.day_max() / {Q}.day_sum()",
    f"{Q}.day_max().safe_div({Q}.day_sum()).ts_mean(window=10, min_periods=8)",
    f"{Q}.between(18, 24).day_sum().safe_div({Q}.day_sum())",
    f"{R}.clip(upper=0).Abs().day_center()",
    f"{R}.day_corr(Minute('market_return'))",
    f"({Q}.zscore_in_day() > 1).day_sum()",
    f"where({R} > 0, Minute('volume'), 0).day_sum()",
    f"{Q}.day_sum() / Factor('quote_volume')",
    f"({Q} * 2 + 1).ln().day_mean().sec_rank()",
]

BAD_SYNTAX = [
    (f"{Q}", "one number per coin per day"),
    (f"{Q} / 2", "one number per coin per day"),
    (f"{Q} / Factor('quote_volume')", "can't be combined"),
    (f"{Q}.ts_mean(window=5)", "needs daily values"),
    (f"{Q}.sec_rank().day_sum()", "needs daily values"),
    (f"{Q}.lag(1).day_sum()", "needs daily values"),
    (f"{Q}.pct_change().day_sum()", "needs daily values"),
    ("Factor('close').day_sum()", "not on a daily value"),
    ("Factor('close').between(0, 12)", "not on a daily value"),
    (f"{Q}.day_sum().day_sum()", "not on a daily value"),
    (f"{Q}.day_corr(Factor('close'))", "not on a daily value"),
    (f"{Q}.between(5, 5).day_sum()", "must come before"),
    (f"{Q}.between(20, 30).day_sum()", "whole hour"),
    (f"{Q}.between(1.5, 3).day_sum()", "whole hour"),
    ("Minute('fwd').day_sum()", "unknown minute field"),
    ("Minute(close).day_sum()", "in quotes"),
    ("Factor('return')", "unknown field"),
]


@pytest.mark.parametrize("text", GOOD_SYNTAX)
def test_good_syntax(text):
    assert dsl.reads_minutes(parse(text))


@pytest.mark.parametrize("text,message", BAD_SYNTAX)
def test_bad_syntax_says_why(text, message):
    with pytest.raises(DslError, match=message):
        parse(text)


def test_fields_keep_minute_and_daily_apart():
    assert dsl.fields_of(parse(f"{Q}.day_sum() / Factor('close')")) == {"minute:quote_volume", "close"}
    assert dsl.fields_of(parse(f"{R}.day_sum()")) == {"minute:close", "minute:open"}
    assert not dsl.reads_minutes(parse("Factor('close')"))
    assert dsl.skeleton(parse(f"{Q}.between(1, 2).day_sum()")) != dsl.skeleton(parse("Minute('volume').between(1, 2).day_sum()"))
    assert dsl.skeleton(parse(f"{Q}.between(1, 2).day_sum()")) == dsl.skeleton(parse(f"{Q}.between(3, 4).day_sum()"))


# Each operator against the plain definition, one coin-day at a time, with holes in the data.


def noisy(seed=1, minutes=48, coin_days=30):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(minutes, coin_days))
    x[rng.random(x.shape) < 0.2] = np.nan
    x[:, 3] = np.nan
    x[:, 4] = 7.0
    x[1:, 5] = np.nan
    x[:-1, 6] = np.nan
    return x


PLAIN = {
    "day_sum": lambda v: v.sum(),
    "day_mean": lambda v: v.mean(),
    "day_std": lambda v: v.std(ddof=1) if v.size > 1 else np.nan,
    "day_max": lambda v: v.max(),
    "day_min": lambda v: v.min(),
    "day_first": lambda v: v[0],
    "day_last": lambda v: v[-1],
}


@pytest.mark.parametrize("name", sorted(PLAIN))
def test_day_operators_against_their_definitions(name):
    x = noisy()
    out = ops.TABLE[name](x)
    for j in range(x.shape[1]):
        v = x[:, j][np.isfinite(x[:, j])]
        want = PLAIN[name](v) if v.size else np.nan
        assert np.isclose(out[j], want, equal_nan=True), (name, j, out[j], want)


def test_day_center():
    x = np.zeros((10, 3))
    x[0, 0] = 5.0
    x[9, 1] = 2.0
    x[:, 2] = 1.0
    out = ops.day_center(x)
    np.testing.assert_allclose(out, [0.05, 0.95, 0.5])
    assert np.isnan(ops.day_center(np.full((10, 1), np.nan)))[0]
    assert np.isnan(ops.day_center(np.zeros((10, 1))))[0]
    assert ops.day_center(np.array([[-9.0], [1.0]]))[0] == 0.75


def test_day_corr():
    x = noisy(1)
    y = noisy(2)
    out = ops.day_corr(x, y)
    for j in range(x.shape[1]):
        both = np.isfinite(x[:, j]) & np.isfinite(y[:, j])
        constant = both.sum() < 2 or np.ptp(x[both, j]) == 0 or np.ptp(y[both, j]) == 0
        want = np.nan if constant else np.corrcoef(x[both, j], y[both, j])[0, 1]
        assert np.isclose(out[j], want, equal_nan=True), (j, out[j], want)


def test_zscore_in_day():
    x = noisy()
    with np.errstate(all="ignore"):
        out = ops.zscore_in_day(x)
    j = 0
    v = x[:, j][np.isfinite(x[:, j])]
    np.testing.assert_allclose(out[:, j][np.isfinite(x[:, j])], (v - v.mean()) / v.std(ddof=1))
    assert np.all(out[:, 4] == 0.0)
    assert np.isnan(out[:, 3]).all()


def test_between_counts_bars_by_their_close_time():
    x = np.arange(1440.0)[:, None]
    # The bar closing at 18:01 is row 1080, the one closing at 24:00 is row 1439.
    kept = ops.between(x, start_hour=18, end_hour=24)
    assert np.flatnonzero(np.isfinite(kept[:, 0]))[[0, -1]].tolist() == [1080, 1439]
    kept = ops.between(x, start_hour=0, end_hour=1)
    assert np.flatnonzero(np.isfinite(kept[:, 0]))[[0, -1]].tolist() == [0, 59]


def test_minute_returns():
    close = np.array([[11.0], [22.0], [np.nan], [33.0]])
    open_ = np.array([[10.0], [11.0], [22.0], [20.0]])
    out = minute_returns(close, open_)
    np.testing.assert_allclose(out[:2, 0], [0.1, 1.0])
    assert np.isnan(out[2:, 0]).all()


# The minute bars agree with the daily bars they add up to.


@pytest.mark.parametrize("name,text", [
    ("quote_volume", f"{Q}.day_sum()"),
    ("volume", "Minute('volume').day_sum()"),
    ("num_trades", "Minute('num_trades').day_sum()"),
    ("taker_quote_vol", "Minute('taker_quote_vol').day_sum()"),
    ("open", "Minute('open').day_first()"),
    ("close", "Minute('close').day_last()"),
    ("high", "Minute('high').day_max()"),
    ("low", "Minute('low').day_min()"),
])
def test_minutes_add_up_to_the_daily_bar(name, text):
    got, want = run(text), DAILY[name]
    assert (np.isfinite(got) == np.isfinite(want)).all()
    np.testing.assert_allclose(got[np.isfinite(got)], want[np.isfinite(want)], rtol=1e-5)


def test_minute_returns_add_up_to_the_days_return():
    got = run(f"({R} + 1).ln().day_sum()")
    with np.errstate(all="ignore"):
        want = np.log(DAILY["close"] / DAILY["open"])
    assert np.nanmax(np.abs(got - want)) < 1e-4


def test_market_series_are_the_median_and_the_mean():
    r = 60
    first, last = BARS.starts[r], BARS.starts[r + 1]
    close, open_ = BARS.fields["close"][first:last].T.astype(float), BARS.fields["open"][first:last].T.astype(float)
    np.testing.assert_allclose(BARS.market["market_return"][r], np.median(minute_returns(close, open_), axis=1), rtol=1e-5, atol=1e-7)
    quote = BARS.fields["quote_volume"][first:last].T.astype(float)
    np.testing.assert_allclose(BARS.market["market_volume_share"][r], (quote / quote.sum(axis=0)).mean(axis=1), rtol=1e-5)


def test_a_coin_day_with_too_few_bars_has_no_value():
    usable = BARS.usable.copy()
    i = int(BARS.starts[50])
    usable[i] = False
    bars = MinuteBars(BARS.rows, BARS.coins, usable, BARS.fields, BARS.market, BARS.shape)
    out = run(f"{Q}.day_sum()", bars)
    assert np.isnan(out[BARS.rows[i], BARS.coins[i]])
    assert np.isfinite(out[BARS.rows[i + 1], BARS.coins[i + 1]])


def test_cache_extends_without_changing_values():
    bars = fresh()
    node = parse(f"{Q}.day_max()")
    short = np.array(bars.day_values(node, 40))
    full = bars.day_values(node, T)
    assert short.shape[0] == 40
    np.testing.assert_array_equal(short, full[:40])
    np.testing.assert_array_equal(full, fresh().day_values(node, T))


def test_operators_in_one_formula_are_computed_in_one_pass():
    bars = fresh()
    a, b = parse(f"{Q}.day_max()"), parse(f"{Q}.day_sum()")
    reads = []
    original = bars._reader

    def counting(part, change, day):
        read = original(part, change, day)

        def counted(name):
            reads.append(name)
            return read(name)

        return counted

    bars._reader = counting
    bars.prepare([a, b], T)
    chunks = len(range(0, int(bars.starts[T]), 256))
    assert len(reads) <= 2 * chunks  # at worst each operator asks once per chunk; the memo answers the rest
    np.testing.assert_array_equal(bars.day_values(a, T), fresh().day_values(a, T))
    np.testing.assert_array_equal(bars.day_values(b, T), fresh().day_values(b, T))


def test_no_minute_data_is_a_clear_error():
    with pytest.raises(NoMinuteData):
        evaluate(parse(f"{Q}.day_sum()"), DAILY)


# No value at row t may depend on anything but row t's own bars, once the daily stage has been applied.


EVERY_MINUTE_OPERATOR = [
    *[f"{Q}.{name}()" for name in ("day_sum", "day_mean", "day_std", "day_max", "day_min", "day_first", "day_last", "day_center")],
    f"{Q}.day_corr(Minute('volume'))",
    f"{R}.day_corr(Minute('market_return'))",
    f"{Q}.day_corr(Minute('market_volume_share'))",
    f"{Q}.between(6, 18).day_sum()",
    f"({Q}.zscore_in_day() > 1).day_sum()",
    f"{R}.day_sum()",
    f"{R}.Abs().day_mean()",
    f"{R}.clip(upper=0).Abs().day_center()",
    f"where({R} > 0, {Q}, 0).day_sum()",
    f"({Q} + 1).ln().pow(2).sign().day_sum()",
    f"({Q} - Minute('volume')).max(Minute('volume')).min({Q}).fillna(0).safe_div(2).day_mean()",
    f"(({Q} >= 1) * ({Q} <= 1e9) * ({Q} < 1e10) * ({Q} != 0) * ({Q} == {Q})).log().day_sum()",
    f"-{Q}.day_sum() / 2",
]


def test_every_method_in_the_minute_grammar_is_covered():
    joined = " ".join(EVERY_MINUTE_OPERATOR)
    for method in dsl._MINUTE_METHODS:
        assert f".{method}(" in joined, method


def poisoned_bars(rows_to_poison, seed=0):
    rng = np.random.default_rng(seed)
    where_ = np.isin(BARS.rows, rows_to_poison)
    fields = {}
    for name, x in BARS.fields.items():
        y = np.array(x)
        y[where_] = rng.choice([1e15, -1e15, 0.0, np.nan, 1e-15], size=(int(where_.sum()), x.shape[1])).astype(np.float32)
        fields[name] = y
    market = {name: np.array(x) for name, x in BARS.market.items()}
    for name in market:
        market[name][rows_to_poison] = np.nan
    return MinuteBars(BARS.rows, BARS.coins, BARS.usable, fields, market, BARS.shape)


@pytest.mark.parametrize("text", EVERY_MINUTE_OPERATOR)
def test_a_day_sees_only_its_own_minutes(text):
    full = run(text)
    target = 70
    others = np.setdiff1d(np.arange(T), [target])
    with np.errstate(all="ignore"):
        out = run(text, poisoned_bars(others))
    # Everything outside that day is wild, and that day's numbers don't move.
    np.testing.assert_array_equal(full[target], out[target])


@pytest.mark.parametrize("text", EVERY_MINUTE_OPERATOR)
def test_truncation(text):
    cut = 80
    full = run(text)
    short = fresh()
    fields = {name: a[: cut + 1] for name, a in DAILY.items()}
    np.testing.assert_array_equal(full[: cut + 1], run(text, short, fields))


def test_later_rows_are_not_even_read():
    node = parse(f"{Q}.day_sum()")
    bars = fresh()
    cut = 40
    last_allowed = int(bars.starts[cut])
    seen = []
    original = bars._reader

    def spying(part, change, day):
        seen.append(part)
        return original(part, change, day)

    bars._reader = spying
    bars.day_values(node, cut)
    assert seen
    for part in seen:
        assert (part.stop if isinstance(part, slice) else int(np.max(part)) + 1) <= last_allowed


def random_minute_expression(rng: random.Random, depth: int) -> str:
    names = ["open", "high", "low", "close", "volume", "quote_volume", "num_trades", "taker_base_vol", "taker_quote_vol", "return"]
    if depth == 0 or rng.random() < 0.2:
        return f"Minute('{rng.choice(names)}')"
    a = random_minute_expression(rng, depth - 1)
    kind = rng.randrange(5)
    if kind == 0:
        return f"({a}) {rng.choice(['+', '-', '*', '/'])} ({random_minute_expression(rng, depth - 1)})"
    if kind == 1:
        return f"({a}).{rng.choice(['Abs', 'ln', 'sign', 'zscore_in_day'])}()"
    if kind == 2:
        start = rng.randrange(0, 23)
        return f"({a}).between({start}, {rng.randrange(start + 1, 25)})"
    if kind == 3:
        return f"where(({a}) > 0, {a}, 0)"
    return f"({a}).clip(lower=-1, upper=1)"


@pytest.mark.parametrize("seed", range(40))
def test_random_minute_expressions(seed):
    rng = random.Random(seed)
    inner = random_minute_expression(rng, rng.randint(1, 4))
    agg = rng.choice(["day_sum", "day_mean", "day_std", "day_max", "day_min", "day_first", "day_last", "day_center"])
    text = f"({inner}).{agg}()"
    if rng.random() < 0.4:
        text += ".ts_mean(window=5, min_periods=1)"
    cut = 70
    full = run(text)
    with np.errstate(all="ignore"):
        out = run(text, poisoned_bars(np.arange(cut + 1, T), seed), {k: np.where(np.arange(T)[:, None] > cut, np.nan, v) for k, v in DAILY.items()})
    np.testing.assert_array_equal(full[: cut + 1], out[: cut + 1])


def test_the_harness_catches_a_day_that_peeks(monkeypatch):
    """A test that can't fail proves nothing: let a day read the next day's bars and watch it fail."""
    honest = BARS._reader

    def peeking(self, part, change, day):
        read = honest(part, change, day)
        if isinstance(part, slice):
            def read_later(name):
                x = read(name).copy()
                source = self.fields[name]
                ahead = min(part.stop + 40, source.shape[0])
                later = np.asarray(source[ahead - x.shape[1] : ahead]).T.astype(np.float64)
                x[0] = later[0]
                return x
            return read_later
        return read

    monkeypatch.setattr(MinuteBars, "_reader", peeking)
    text = f"{Q}.day_sum()"
    full = run(text, fresh())
    others = np.setdiff1d(np.arange(T), [70])
    with np.errstate(all="ignore"):
        out = run(text, poisoned_bars(others))
    assert not np.array_equal(full[70], out[70], equal_nan=True)


# The gate, on a market where the right answers are known.


GATE_T = 1000
GATE_N = 40


@pytest.fixture(scope="module")
def gate():
    bars, daily = synthetic_minutes(GATE_T, GATE_N, M, seed=9)
    dates = (np.datetime64("2021-12-28") + np.arange(GATE_T)).astype("datetime64[D]")
    universe = np.ones((GATE_T, GATE_N), dtype=bool)
    universe[:30] = False
    panel = Panel(dates, np.arange(GATE_N).astype(str), np.arange(GATE_N).astype(str), daily, universe)
    return sb.Sandbox(panel, minutes=bars)


def judged(box, text, key, direction, also=()):
    node = parse(text)
    memo = {}
    with np.errstate(all="ignore"):
        base = evaluate(node, box.gate_panel.fields, memo=memo, minutes=box.minutes)
        return box.check_compute(base), box.check_idea(node, key, direction, base, also=also, memo=memo)


DOWN = f"{R}.clip(upper=0).Abs().day_center()"
UP = f"{R}.clip(lower=0).day_center()"
HONEST = [
    (f"{Q}.day_center()", "volume_late", 1),
    (f"{Q}.day_center().ts_mean(window=10, min_periods=8)", "volume_late", 1),
    (f"{Q}.between(18, 24).day_sum().safe_div({Q}.day_sum())", "volume_late", 1),
    (f"-1 * {Q}.between(0, 6).day_sum().safe_div({Q}.day_sum())", "volume_late", 1),
    (f"{Q}.day_max().safe_div({Q}.day_sum())", "volume_spike", 1),
    (f"{Q}.day_max().safe_div({Q}.day_sum()).ts_mean(window=10, min_periods=8)", "volume_spike", 1),
    (f"{Q}.day_std().safe_div({Q}.day_mean())", "volume_spike", 1),
    (f"where({R} > 0, {Q}, 0).day_sum().safe_div({Q}.day_sum())", "volume_on_up", 1),
    (DOWN, "down_late", 1),
    (f"{DOWN}.ts_mean(window=10, min_periods=8)", "down_late", 1),
    (UP, "up_late", 1),
    (f"{UP}.EMA(span=5, min_periods=4)", "up_late", 1),
    (f"{R}.day_corr(Minute('market_return'))", "market_sync", 1),
    (f"{R}.day_corr(Minute('market_return')).ts_mean(window=10, min_periods=8)", "market_sync", 1),
    # Older keys, answered from minutes.
    (f"{Q}.day_sum().ts_mean(window=5, min_periods=4)", "volume", 1),
    ("Minute('num_trades').day_sum().ts_mean(window=5, min_periods=4)", "trades", 1),
    (f"{Q}.day_sum().safe_div(Minute('num_trades').day_sum()).ts_mean(window=10, min_periods=8)", "trade_size", 1),
    ("Minute('close').day_last().safe_div(Minute('open').day_first()) - 1", "return", 1),
    (f"{R}.day_sum()", "return", 1),
    (f"{R}.day_std().ts_mean(window=10, min_periods=8)", "volatility", 1),
    ("Minute('taker_quote_vol').day_sum().safe_div(Minute('quote_volume').day_sum())", "taker_buy", 1),
]


@pytest.mark.parametrize("text,key,direction", HONEST)
def test_honest_minute_factors_pass(gate, text, key, direction):
    compute, idea = judged(gate, text, key, direction)
    assert compute.ok, compute
    assert idea.ok, (idea.reason, idea.windows, idea.unit_median, idea.unrelated)


@pytest.mark.parametrize("text,key,direction", HONEST)
def test_the_wrong_direction_fails(gate, text, key, direction):
    _, idea = judged(gate, text, key, -direction)
    assert not idea.ok
    assert idea.reason == "moves_against_the_idea"


@pytest.mark.parametrize("key", ["volume_late", "volume_spike", "volume_on_up", "down_late", "up_late", "market_sync"])
@pytest.mark.parametrize("text", [
    "Factor('quote_volume').ts_mean(window=5, min_periods=4)",
    "(Factor('close') / Factor('open') - 1).ts_mean(window=5, min_periods=4)",
    "(Factor('high') / Factor('low')).ln()",
    "Factor('taker_quote_vol') / Factor('quote_volume')",
])
def test_a_daily_formula_cant_answer_an_idea_about_the_inside_of_a_day(gate, text, key):
    _, idea = judged(gate, text, key, 1)
    assert not idea.ok
    assert idea.reason == "does_not_respond_to_the_idea"
    assert all(v["median"] == 0.0 for v in idea.windows.values())


@pytest.mark.parametrize("text,key", [
    (f"{Q}.day_center()", "down_late"),
    (f"{Q}.day_center()", "market_sync"),
    (f"{Q}.day_max().safe_div({Q}.day_sum())", "volume_on_up"),
    (DOWN, "volume_late"),
    (DOWN, "up_late"),
    (UP, "down_late"),
    (f"{Q}.day_sum().ln()", "volume_late"),
    (f"{Q}.day_sum().ln()", "volume_spike"),
    (f"{R}.day_sum()", "volume_late"),
    (f"{R}.day_std()", "volume_spike"),
])
def test_a_formula_about_something_else_fails(gate, text, key):
    _, idea = judged(gate, text, key, 1)
    assert not idea.ok, (idea.windows,)


@pytest.mark.parametrize("text,key", [
    ("Minute('close').day_last()", "return"),
    ("Minute('volume').day_sum()", "volume"),
    ("Minute('volume').day_max()", "volume_spike"),
])
def test_unit_dependent_minute_factors_fail(gate, text, key):
    compute, idea = judged(gate, text, key, 1)
    assert compute.ok
    assert not idea.ok


def test_total_volume_is_not_a_share(gate):
    """Dollars late in the day mostly measure dollars: more of them all day moves it more."""
    _, idea = judged(gate, f"{Q}.between(18, 24).day_sum()", "volume_late", 1)
    assert not idea.ok


def test_the_largest_minute_in_dollars_measures_volume(gate):
    _, idea = judged(gate, f"{Q}.day_max().ts_mean(window=5, min_periods=4)", "volume_spike", 1)
    assert idea.reason == "measures_volume_not_volume_spike"


@pytest.mark.parametrize("text,key", [
    (f"({Q}.day_max().safe_div({Q}.day_sum())).sec_rank() + 0.3 * ({Q}.day_center()).sec_rank()", "volume_late"),
    (f"({Q}.day_center()).sec_rank() + 0.3 * ({Q}.day_max().safe_div({Q}.day_sum())).sec_rank()", "volume_spike"),
    (f"({UP}).sec_rank() + 0.3 * ({DOWN}).sec_rank()", "down_late"),
    (f"({DOWN}).sec_rank() + 0.3 * ({UP}).sec_rank()", "up_late"),
])
def test_a_dose_of_the_right_thing_in_a_rivals_formula_fails(gate, text, key):
    _, idea = judged(gate, text, key, 1)
    assert not idea.ok
    if idea.reason != "does_not_respond_to_the_idea":
        assert idea.reason == "driven_by_unrelated_data", idea


@pytest.mark.parametrize("text", [
    f"({Q} / {Q}).day_sum()",
    f"{Q}.day_sum() - {Q}.day_sum()",
    f"{Q}.day_max() * 0",
])
def test_degenerate_minute_factors_dont_compute(gate, text):
    compute, _ = judged(gate, text, "volume", 1)
    assert not compute.ok


# Patching a few coin-days must give exactly what perturbing all the bars and starting over gives.


def _perturb_everything(box, key, window, name_list):
    """Apply a key's minute change to the whole store the slow way."""
    bars = box.minutes
    fields = {name: np.array(x) for name, x in bars.fields.items()}
    patched = box._perturbed_minutes(parse(f"{Q}.day_sum()"), key, window, box._perturbed(key, window), {})
    index = patched.index
    seen: dict = {}

    def original(name):
        if name not in seen:
            seen[name] = np.asarray(bars.market[name][bars.rows[index]] if name in bars.market else bars.fields[name][index]).T.astype(np.float64)
        return seen[name]

    day = dict(patched.day)
    for name in name_list:
        fields[name][index] = patched.change(name, original, day).T.astype(np.float32)
    return MinuteBars(bars.rows, bars.coins, bars.usable, fields, bars.market, bars.shape)


@pytest.mark.parametrize("key", sorted(sb.MINUTE_KEYS))
@pytest.mark.parametrize("text", [f"{Q}.day_max().safe_div({Q}.day_sum())", DOWN, f"{R}.day_corr(Minute('market_return'))", "Minute('close').day_last().safe_div(Minute('open').day_first())"])
def test_patching_equals_starting_over(gate, key, text):
    node = parse(text)
    window = 5
    with np.errstate(all="ignore"):
        memo = {}
        base = evaluate(node, gate.gate_panel.fields, memo=memo, minutes=gate.minutes)
        fields = gate._perturbed(key, window)
        quick = evaluate(node, fields, memo={}, minutes=gate._perturbed_minutes(node, key, window, fields, memo))
        slow_bars = _perturb_everything(gate, key, window, list(dsl.FIELDS))
        slow = evaluate(node, fields, memo={}, minutes=slow_bars)
    both = np.isfinite(quick) | np.isfinite(slow)
    np.testing.assert_allclose(quick[both], slow[both], rtol=2e-4, atol=1e-5)
    assert base.shape == quick.shape


# A perturbation is one event: the bars it leaves must add up to the daily fields it leaves.


@pytest.mark.parametrize("key,window", [(k, w) for k in ("volume", "trades", "trade_size", "return", "volatility", "close_near_high") for w in (1, 5)])
@pytest.mark.parametrize("name,text", [
    ("quote_volume", f"{Q}.day_sum()"),
    ("num_trades", "Minute('num_trades').day_sum()"),
    ("open", "Minute('open').day_first()"),
    ("close", "Minute('close').day_last()"),
    ("high", "Minute('high').day_max()"),
    ("low", "Minute('low').day_min()"),
])
def test_daily_and_minute_perturbations_are_one_event(gate, key, window, name, text):
    node = parse(text)
    with np.errstate(all="ignore"):
        memo = {}
        evaluate(node, gate.gate_panel.fields, memo=memo, minutes=gate.minutes)
        fields = gate._perturbed(key, window)
        got = evaluate(node, fields, memo={}, minutes=gate._perturbed_minutes(node, key, window, fields, memo))
    want = fields[name]
    ok = np.isfinite(got) & np.isfinite(want)
    assert ok.sum() > 1000
    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-4)


@pytest.mark.parametrize("key", ["volume_late", "volume_spike", "volume_on_up", "down_late", "up_late", "market_sync"])
@pytest.mark.parametrize("name,text", [
    ("quote_volume", f"{Q}.day_sum()"),
    ("num_trades", "Minute('num_trades').day_sum()"),
    ("open", "Minute('open').day_first()"),
    ("close", "Minute('close').day_last()"),
])
def test_keys_about_the_inside_of_a_day_leave_the_daily_bar_alone(gate, key, name, text):
    node = parse(text)
    with np.errstate(all="ignore"):
        memo = {}
        evaluate(node, gate.gate_panel.fields, memo=memo, minutes=gate.minutes)
        fields = gate._perturbed(key, 20)
        got = evaluate(node, fields, memo={}, minutes=gate._perturbed_minutes(node, key, 20, fields, memo))
    want = gate.gate_panel.fields[name]
    assert fields[name] is want
    ok = np.isfinite(got) & np.isfinite(want)
    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-3)


@pytest.mark.parametrize("key", ["volume_late", "volume_on_up"])
@pytest.mark.parametrize("name", ["quote_volume", "volume", "num_trades", "taker_quote_vol"])
def test_volume_keys_only_rearrange_a_days_minutes(gate, key, name):
    patched = gate._perturbed_minutes(parse(f"{Q}.day_sum()"), key, 5, gate._perturbed(key, 5), {})
    index = patched.index[:200]
    bars = gate.minutes

    def original(field):
        return np.asarray(bars.fields[field][index]).T.astype(np.float64)

    day = {k: v[:200] for k, v in patched.day.items()}
    after = patched.change(name, original, day)
    np.testing.assert_array_equal(np.sort(after, axis=0), np.sort(original(name), axis=0))
    assert (after != original(name)).any()


@pytest.mark.parametrize("key", ["down_late", "up_late", "market_sync"])
def test_price_keys_only_rearrange_a_days_returns(gate, key):
    patched = gate._perturbed_minutes(parse(f"{R}.day_sum()"), key, 5, gate._perturbed(key, 5), {})
    index = patched.index[:200]
    bars = gate.minutes

    def original(field):
        source = bars.market[field][bars.rows[index]] if field in bars.market else bars.fields[field][index]
        return np.asarray(source).T.astype(np.float64)

    day = {k: v[:200] for k, v in patched.day.items()}
    new = {name: patched.change(name, original, day) for name in ("open", "high", "low", "close")}
    moves = lambda read: sb._log_returns(read)
    before = moves(original)
    after = moves(lambda name: new[name])
    np.testing.assert_allclose(np.sort(after, axis=0), np.sort(before, axis=0), atol=1e-9)
    assert (np.abs(after - before) > 1e-9).any()
    np.testing.assert_allclose(new["open"][0], original("open")[0])
    np.testing.assert_allclose(new["close"][-1], original("close")[-1])
    assert (new["high"] >= np.maximum(new["open"], new["close"]) * (1 - 1e-9)).all()
    assert (new["low"] <= np.minimum(new["open"], new["close"]) * (1 + 1e-9)).all()


@pytest.mark.parametrize("key", ["volume_late", "volume_on_up", "down_late", "up_late", "market_sync"])
@pytest.mark.parametrize("text", [
    f"{Q}.day_sum()", f"{Q}.day_max().safe_div({Q}.day_sum())", f"{Q}.day_std().safe_div({Q}.day_mean())",
    f"{R}.day_std()", f"{R}.Abs().day_sum()", f"{R}.day_max()", f"{R}.day_min()", f"{R}.day_sum()",
])
def test_a_formula_that_ignores_the_order_of_the_day_cant_respond(gate, key, text):
    """The numbers in the day are the same, so anything that doesn't look at when they fall stays put."""
    _, idea = judged(gate, text, key, 1)
    assert not idea.ok
    assert all(abs(v["median"]) < 0.02 for v in idea.windows.values()), idea.windows


def test_report_and_score_read_minutes(gate):
    text = f"{Q}.day_center()"
    report = gate.report(text)
    assert report.startswith("computes: yes"), report
    score = gate.score(text, "volume_late", 1, ic_range=("2022-07-01", "2024-01-01"))
    assert score.parsed and score.computes and score.matches_idea, score
    assert score.ic is not None and score.ic.days > 300


def test_without_bars_report_and_score_say_so():
    box = sb.Sandbox.__new__(sb.Sandbox)
    box.minutes = None
    panel_dates = (np.datetime64("2021-12-28") + np.arange(T)).astype("datetime64[D]")
    box.gate_panel = Panel(panel_dates, np.arange(N).astype(str), np.arange(N).astype(str), DAILY, np.ones((T, N), dtype=bool))
    box.dev = box.gate_panel
    assert "minute data is not available" in box.report(f"{Q}.day_sum()")
    score = box.score(f"{Q}.day_sum()", "volume", 1)
    assert not score.computes and "minute data is not available" in score.reason
