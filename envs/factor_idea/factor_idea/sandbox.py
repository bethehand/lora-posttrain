"""What the sandbox checks about a factor, and what it pays.

Three steps, in order:

1. It computes: the expression parses and gives enough coins and distinct values on most days.
2. It matches the idea: raising the idea's key data for a few coins raises (or, for a
   bearish idea, lowers) those coins' cross-sectional rank, unrelated data doesn't move
   the rank much more than the key does, and changing a coin's price unit changes
   nothing. Measured on rows up to 2024-12-31. A perturbation is the same event in the daily
   and the minute data, so a formula is judged the same whichever it reads. Some keys only
   rearrange a day's minutes: they leave every daily bar alone, so only a formula that
   looks inside the day can respond to them.
3. Its score: what its rank adds to explaining next-day return ranks beyond liquidity,
   volatility, reversal and momentum (a semi-partial rank IC), in a period the model never sees.

No step reads a row after the factor's date: forward returns are the only thing that
looks ahead, they enter only step 3, and they never feed into a factor value.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from factor_idea import dsl
from factor_idea.dsl import DslError, Node
from factor_idea.evaluate import EvalTimeout, NoMinuteData, evaluate
from factor_idea.minutes import MinuteBars, PatchedMinutes
from factor_idea.ops import rank_row
from factor_idea.panel import Panel

GATE_FIRST, GATE_LAST = "2022-07-01", "2024-12-31"
SELECTION_IC = ("2022-07-01", "2024-12-29")
RL_IC = ("2025-01-01", "2025-12-29")
SEALED_IC = ("2026-01-01", "2026-10-16")
REPORT_PERIODS = (("2022H2", "2022-07-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-29"))

EVAL_SECONDS = 180.0
MIN_COINS = 20
MIN_COVERAGE = 0.5
MIN_DISTINCT = 5
MIN_GOOD_DAYS = 0.9

SCALE = 3.0
JUMP = 1.5  # tripling a price is a far rarer event than tripling a day's volume; +50% is comparable
FALL = 0.5  # the same size of daily move the other way, as a simple return
WINDOWS = (1, 5, 20, 60)
AS_OF_DATES = 12
COIN_SHARE = 0.10
MIN_COINS_PER_DATE = 5
RISE_MEDIAN = 0.10
RISE_SHARE = 0.6
UNIT_FACTOR = 10.0
UNIT_MEDIAN = 0.02
SPIKE = 0.3  # share of a day's volume moved into its busiest minute
SYNC_BLOCK = 5  # minutes

# The middle 90% of honest formulas fall within about +/-0.02 in a year (see ic_persistence.py).
# The scale spans negative IC too: where an idea's IC is negative whatever the formula, clipping
# at 0 would give every answer the same reward and RL nothing to learn from.
IC_RANGE = 0.03
BASE_REWARD = 0.3

STYLES = {
    "liquidity": 'Factor("quote_volume").ts_mean(window=30, min_periods=20).ln()',
    "volatility": '(Factor("close").safe_div(Factor("close").lag(periods=1)) - 1).ts_std(window=20, min_periods=15)',
    "reversal_1d": 'Factor("close").safe_div(Factor("close").lag(periods=1)) - 1',
    "reversal_5d": 'Factor("close").safe_div(Factor("close").lag(periods=5)) - 1',
    "momentum_30d": 'Factor("close").safe_div(Factor("close").lag(periods=30)) - 1',
}

_PRICE = ("open", "high", "low", "close")
_VOLUME = ("volume", "quote_volume", "taker_base_vol", "taker_quote_vol")


def _scale_volume(f, rows, coins):
    for name in (*_VOLUME, "num_trades"):
        f[name][np.ix_(rows, coins)] *= SCALE


def _scale_trades(f, rows, coins):
    f["num_trades"][np.ix_(rows, coins)] *= SCALE


def _scale_trade_size(f, rows, coins):
    for name in _VOLUME:
        f[name][np.ix_(rows, coins)] *= SCALE


def _shift_prices(f, rows, coins, factor):
    """Every price after the open of the window's first day is ``factor`` times what it was.

    The first bar's low moves with its high, so the day's range is unchanged and a
    range formula doesn't pass for a return formula. After a rally that bar's low sits above its open.
    """
    first, later = rows[:1], rows[1:]
    for name in ("close", "high", "low"):
        f[name][np.ix_(first, coins)] *= factor
    for name in _PRICE:
        f[name][np.ix_(later, coins)] *= factor


def _jump_price(f, rows, coins):
    _shift_prices(f, rows, coins, JUMP)


def _fall_price(f, rows, coins):
    _shift_prices(f, rows, coins, FALL)


def _amplify_moves(f, rows, coins):
    """Every price's log distance from the previous close gets SCALE times longer, inside the window."""
    original = {name: f[name][np.ix_(rows, coins)].copy() for name in _PRICE}
    previous_close = f["close"][rows[0] - 1, coins].copy()
    path = previous_close.copy()
    for i, _ in enumerate(rows):
        for name in _PRICE:
            f[name][rows[i], coins] = path * (original[name][i] / previous_close) ** SCALE
        path = f["close"][rows[i], coins].copy()
        previous_close = original["close"][i]


def _raise_taker_buy_share(f, rows, coins):
    """The odds that a traded unit was an aggressive buy get SCALE times higher."""
    for part, whole in (("taker_base_vol", "volume"), ("taker_quote_vol", "quote_volume")):
        total = f[whole][np.ix_(rows, coins)]
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.nan_to_num(f[part][np.ix_(rows, coins)] / total, nan=0.0)
        f[part][np.ix_(rows, coins)] = SCALE * share / (1.0 + (SCALE - 1.0) * share) * total


def _close_toward_high(f, rows, coins):
    """Same open and close: the upper shadow gets shorter and the lower shadow longer."""
    o, c = f["open"][np.ix_(rows, coins)], f["close"][np.ix_(rows, coins)]
    h, l = f["high"][np.ix_(rows, coins)], f["low"][np.ix_(rows, coins)]
    top, bottom = np.maximum(o, c), np.minimum(o, c)
    f["high"][np.ix_(rows, coins)] = top + (h - top) / SCALE
    f["low"][np.ix_(rows, coins)] = np.maximum(bottom - SCALE * (bottom - l), 0.01 * bottom)


def _solid_body(f, rows, coins):
    """Same open and close: both shadows get shorter, so the body is more of the day's range."""
    o, c = f["open"][np.ix_(rows, coins)], f["close"][np.ix_(rows, coins)]
    h, l = f["high"][np.ix_(rows, coins)], f["low"][np.ix_(rows, coins)]
    top, bottom = np.maximum(o, c), np.minimum(o, c)
    f["high"][np.ix_(rows, coins)] = top + (h - top) / SCALE
    f["low"][np.ix_(rows, coins)] = bottom - (bottom - l) / SCALE


def _volume_on_up_days(f, rows, coins):
    """The window's days keep their prices and trade the same volumes, but bigger volumes land on up days."""
    moves = f["close"][np.ix_(rows, coins)] / f["close"][np.ix_(rows - 1, coins)] - 1.0
    order = _bigger_on_up(np.nan_to_num(f["quote_volume"][np.ix_(rows, coins)], nan=0.0), np.nan_to_num(moves, nan=0.0))
    for name in _FLOW:
        f[name][np.ix_(rows, coins)] = np.take_along_axis(f[name][np.ix_(rows, coins)], order, axis=0)


def _no_daily_change(f, rows, coins):
    """The daily bar stays as it is: this key only rearranges minutes inside the day."""


# The key data an idea can be about, each with the daily fields its perturbation changes.
KEYS = {
    "volume": ((*_VOLUME, "num_trades"), _scale_volume),
    "trades": (("num_trades",), _scale_trades),
    "trade_size": (_VOLUME, _scale_trade_size),
    "return": (_PRICE, _jump_price),
    "volatility": (_PRICE, _amplify_moves),
    "taker_buy": (("taker_base_vol", "taker_quote_vol"), _raise_taker_buy_share),
    "close_near_high": (("high", "low"), _close_toward_high),
    "solid_body": (("high", "low"), _solid_body),
    "volume_on_up_days": ((*_VOLUME, "num_trades"), _volume_on_up_days),
    "volume_late": ((), _no_daily_change),
    "volume_spike": ((), _no_daily_change),
    "volume_on_up": ((), _no_daily_change),
    "down_late": ((), _no_daily_change),
    "up_late": ((), _no_daily_change),
    "market_sync": ((), _no_daily_change),
}


# What a rally's mirror image does to a formula. It is no idea's key: see Sandbox._likes_a_fall.
SHOCKS = {**KEYS, "fall": (_PRICE, _fall_price)}


# The same perturbations on minute bars. Each takes a field's name, a reader of the original bars
# (minutes x coin-days) and what the coin-days need to know, and gives that field's new bars.
_FLOW = (*_VOLUME, "num_trades")


def _scale_minutes(names: tuple[str, ...]):
    def change(name, original, day):
        return original(name) * SCALE if name in names else original(name)

    return change


def _jump_minutes(name, original, day):
    x = original(name)
    if name not in _PRICE:
        return x
    out = x * JUMP
    if name == "open":
        out[0, day["first"]] = x[0, day["first"]]
    return out


def _amplify_minutes(name, original, day):
    x = original(name)
    return day["new_previous_close"] * (x / day["previous_close"]) ** SCALE if name in _PRICE else x


def _taker_minutes(name, original, day):
    x = original(name)
    whole = {"taker_base_vol": "volume", "taker_quote_vol": "quote_volume"}.get(name)
    if whole is None:
        return x
    total = original(whole)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.nan_to_num(x / total, nan=0.0)
    return SCALE * share / (1.0 + (SCALE - 1.0) * share) * total


def _close_toward_high_minutes(name, original, day):
    x = original(name)
    if name not in _PRICE:
        return x
    top, bottom = day["top"], day["bottom"]
    above = top + (x - top) / SCALE
    below = np.maximum(bottom - SCALE * (bottom - x), 0.01 * bottom)
    return np.where(x > top, above, np.where(x < bottom, below, x))


# Keys about the inside of a day. All but the spike only rearrange a day's minutes, so the numbers
# in the day stay the same and any formula that doesn't care about their order can't respond.


def _trade_places(minutes: int, first: np.ndarray, second: np.ndarray, swap: np.ndarray) -> np.ndarray:
    """A new order for each coin-day's minutes: first[i] and second[i] trade places where swap[i]."""
    days = first.shape[1]
    order = np.broadcast_to(np.arange(minutes)[:, None], (minutes, days)).copy()
    column = np.broadcast_to(np.arange(days), first.shape)
    order[first[swap], column[swap]] = second[swap]
    order[second[swap], column[swap]] = first[swap]
    return order


def _bigger_later(size: np.ndarray, picked: np.ndarray) -> np.ndarray:
    """Among the picked minutes, the i-th and the i-th from last trade places if the earlier is bigger."""
    minutes = size.shape[0]
    by_time = np.argsort(~picked, axis=0, kind="stable")
    count = picked.sum(axis=0)
    i = np.arange(minutes)[:, None]
    partner = np.take_along_axis(by_time, np.where(i < count, count - 1 - i, i), axis=0)
    wanted = (i < count - 1 - i) & (np.take_along_axis(size, by_time, axis=0) > np.take_along_axis(size, partner, axis=0))
    return _trade_places(minutes, by_time, partner, wanted)


def _log_returns(original) -> np.ndarray:
    close = np.log(original("close"))
    out = np.empty_like(close)
    out[0] = close[0] - np.log(original("open")[0])
    out[1:] = close[1:] - close[:-1]
    return out


def _bigger_on_up(size: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """The i-th up minute and the i-th down minute trade places if the down minute is the bigger."""
    minutes = size.shape[0]
    by_kind = np.argsort(np.where(moves > 0, 0, np.where(moves < 0, 1, 2)), axis=0, kind="stable")
    ups, downs = (moves > 0).sum(axis=0), (moves < 0).sum(axis=0)
    i = np.arange(minutes)[:, None]
    down = np.take_along_axis(by_kind, np.minimum(i + ups, minutes - 1), axis=0)
    wanted = (i < np.minimum(ups, downs)) & (np.take_along_axis(size, by_kind, axis=0) < np.take_along_axis(size, down, axis=0))
    return _trade_places(minutes, by_kind, down, wanted)


def _in_step_with_market(moves: np.ndarray, market: np.ndarray) -> np.ndarray:
    """Inside each short block of minutes, the coin's moves go in the same order as the market's."""
    minutes, days = moves.shape
    blocks = minutes // SYNC_BLOCK
    whole = blocks * SYNC_BLOCK
    shape = (blocks, SYNC_BLOCK, days)
    own, theirs = moves[:whole].reshape(shape), market[:whole].reshape(shape)
    place = np.argsort(np.argsort(theirs, axis=1), axis=1)
    chosen = np.take_along_axis(np.argsort(own, axis=1), place, axis=1)
    complete = np.isfinite(own).all(axis=1, keepdims=True) & np.isfinite(theirs).all(axis=1, keepdims=True)
    chosen = np.where(complete, chosen, np.arange(SYNC_BLOCK)[None, :, None])
    order = np.broadcast_to(np.arange(minutes)[:, None], (minutes, days)).copy()
    order[:whole] = (chosen + (np.arange(blocks) * SYNC_BLOCK)[:, None, None]).reshape(whole, days)
    return order


def _flow_in_new_order(name, original, day, find_order):
    x = original(name)
    if name not in _FLOW:
        return x
    if "order" not in day:
        day["order"] = find_order(original)
    return np.take_along_axis(x, day["order"], axis=0)


def _prices_from_new_moves(name, original, day, find_order):
    """Prices rebuilt from the day's minute returns in a new order: the day opens and closes where it did."""
    x = original(name)
    if name not in _PRICE:
        return x
    if "after" not in day:
        moves = _log_returns(original)
        moved = np.take_along_axis(moves, find_order(moves, original), axis=0)
        day["after"] = np.exp(np.nancumsum(moved - moves, axis=0))
        day["before"] = np.vstack([np.ones((1, x.shape[1])), day["after"][:-1]])
    after, before = day["after"], day["before"]
    return x * {"open": before, "close": after, "high": np.fmax(before, after), "low": np.fmin(before, after)}[name]


def _volume_late_minutes(name, original, day):
    def order(read):
        size = read("quote_volume")
        return _bigger_later(np.nan_to_num(size, nan=0.0), np.isfinite(size))

    return _flow_in_new_order(name, original, day, order)


def _volume_on_up_minutes(name, original, day):
    def order(read):
        return _bigger_on_up(np.nan_to_num(read("quote_volume"), nan=0.0), _log_returns(read))

    return _flow_in_new_order(name, original, day, order)


def _volume_spike_minutes(name, original, day):
    x = original(name)
    if name not in _FLOW:
        return x
    busiest = np.argmax(np.nan_to_num(original("quote_volume"), nan=-1.0), axis=0)
    out = (1.0 - SPIKE) * x
    out[busiest, np.arange(x.shape[1])] += SPIKE * np.nansum(x, axis=0)
    return out


def _down_late_minutes(name, original, day):
    return _prices_from_new_moves(name, original, day, lambda moves, _: _bigger_later(np.abs(np.nan_to_num(moves, nan=0.0)), moves < 0))


def _up_late_minutes(name, original, day):
    return _prices_from_new_moves(name, original, day, lambda moves, _: _bigger_later(np.nan_to_num(moves, nan=0.0), moves > 0))


def _market_sync_minutes(name, original, day):
    return _prices_from_new_moves(name, original, day, lambda moves, read: _in_step_with_market(moves, read("market_return")))


def _unit_minutes(name, original, day):
    x = original(name)
    if name in _PRICE:
        return x * UNIT_FACTOR
    return x / UNIT_FACTOR if name in ("volume", "taker_base_vol") else x


# key -> the minute fields its perturbation changes, and how.
MINUTE_KEYS = {
    "volume": (_FLOW, _scale_minutes(_FLOW)),
    "trades": (("num_trades",), _scale_minutes(("num_trades",))),
    "trade_size": (_VOLUME, _scale_minutes(_VOLUME)),
    "return": (_PRICE, _jump_minutes),
    "volatility": (_PRICE, _amplify_minutes),
    "taker_buy": (("taker_base_vol", "taker_quote_vol"), _taker_minutes),
    "close_near_high": (_PRICE, _close_toward_high_minutes),
    "volume_late": (_FLOW, _volume_late_minutes),
    "volume_spike": (_FLOW, _volume_spike_minutes),
    "volume_on_up": (_FLOW, _volume_on_up_minutes),
    "down_late": (_PRICE, _down_late_minutes),
    "up_late": (_PRICE, _up_late_minutes),
    "market_sync": (_PRICE, _market_sync_minutes),
}


def _changed_by(key: str) -> frozenset[str]:
    """Everything a key's perturbation changes, in the names dsl.fields_of() uses."""
    minutes = MINUTE_KEYS[key][0] if key in MINUTE_KEYS else ()
    return frozenset(SHOCKS[key][0]) | frozenset(dsl.minute_name(n) for n in minutes)


# Keys in one group legitimately move each other's formulas: a price jump is also a big move,
# and more volume is also more dollars per trade. Only other groups count as unrelated data.
KEY_GROUPS = {
    "volume": "flow",
    "trades": "flow",
    "trade_size": "flow",
    "return": "price",
    "volatility": "price",
    "close_near_high": "price",
    "taker_buy": "taker",
    "solid_body": "price",
    "volume_on_up_days": "flow",
    "volume_late": "flow",
    "volume_spike": "flow",
    "volume_on_up": "flow",
    "down_late": "price",
    "up_late": "price",
    "market_sync": "price",
}


# Average trade size is dollars per trade: scaling dollars and trades together must leave it
# alone. A formula that moves anyway is measuring total volume under the wrong name.
# The same goes for what a day's minutes do with its volume: a share or a timing doesn't change when
# the whole day's volume does, and a total that merely happens to respond is measuring volume.
MUST_IGNORE = {"trade_size": "volume", "volume_on_up_days": "volume", "volume_late": "volume", "volume_spike": "volume", "volume_on_up": "volume"}
MUST_IGNORE_SHARE = 0.5

# Keys of one group that are still different questions. A formula that moves more for a rival than
# for its own key is about the rival, even if a dose of the right thing is mixed in.
RIVALS = {
    "solid_body": ("volatility", "close_near_high"),
    "volume_late": ("volume_spike", "volume_on_up"),
    "volume_spike": ("volume_late", "volume_on_up"),
    "volume_on_up": ("volume_late", "volume_spike"),
    "down_late": ("up_late", "close_near_high"),
    "up_late": ("down_late", "close_near_high"),
}


def _pct_rank(values: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Percentile ranks among universe members with a value; NaN elsewhere."""
    mask = members & np.isfinite(values)
    out = np.full(values.shape, np.nan)
    if mask.any():
        out[mask] = rank_row(values[mask]) / mask.sum()
    return out


@dataclass
class ComputeCheck:
    ok: bool
    reason: str = ""
    good_day_share: float = 0.0


@dataclass
class IdeaCheck:
    ok: bool
    reason: str = ""
    windows: dict = field(default_factory=dict)
    unit_median: float = 0.0
    unrelated: dict = field(default_factory=dict)


@dataclass
class ICResult:
    incremental: float
    raw: float
    days: int
    t_stat: float


@dataclass
class Score:
    parsed: bool = False
    computes: bool = False
    matches_idea: bool = False
    reason: str = ""
    ic: ICResult | None = None
    skeleton: str = ""
    seconds: float = 0.0

    def reward(self, arm: str) -> float:
        """Arm "gates": passing is full marks. Arm "ic": passing earns the base, and IC earns the rest."""
        if not (self.parsed and self.computes and self.matches_idea):
            return 0.0
        if arm == "gates":
            return 1.0
        if arm == "ic":
            value = self.ic.incremental if self.ic and np.isfinite(self.ic.incremental) else -IC_RANGE
            return BASE_REWARD + (1.0 - BASE_REWARD) * float(np.clip((value + IC_RANGE) / (2 * IC_RANGE), 0.0, 1.0))
        raise ValueError(f"unknown arm {arm!r}")


class Sandbox:
    def __init__(self, dev: Panel, seed: int = 20261105, minutes: MinuteBars | None = None):
        self.dev = dev
        self.gate_panel = dev.upto(GATE_LAST)
        self.minutes = minutes
        self._ic_inputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._plan_perturbations(np.random.default_rng(seed))

    def _plan_perturbations(self, rng: np.random.Generator) -> None:
        panel = self.gate_panel
        first, last = panel.row(GATE_FIRST), len(panel.dates) - 1
        self.as_of = np.linspace(first, last, AS_OF_DATES).astype(int)
        close = panel.fields["close"]
        used = np.zeros(close.shape[1], dtype=bool)
        self.plan: list[tuple[int, np.ndarray]] = []
        history = max(WINDOWS) + 1
        for t in self.as_of:
            members = panel.universe[t]
            seen = np.isfinite(close[t - history : t + 1]).all(axis=0)
            eligible = np.flatnonzero(members & seen)
            count = max(MIN_COINS_PER_DATE, int(np.ceil(COIN_SHARE * members.sum())))
            fresh = eligible[~used[eligible]]
            pool = fresh if fresh.size >= count else eligible
            coins = np.sort(rng.choice(pool, size=min(count, pool.size), replace=False))
            used[coins] = True
            self.plan.append((int(t), coins))
        self.unit_coins = np.sort(rng.choice(close.shape[1], size=int(COIN_SHARE * close.shape[1]), replace=False))

    def values(self, node: Node, panel: Panel, *, deadline: float | None = None, memo: dict | None = None) -> np.ndarray:
        return evaluate(node, panel.fields, deadline=deadline, memo=memo, minutes=self.minutes)

    def check_compute(self, values: np.ndarray) -> ComputeCheck:
        panel = self.gate_panel
        rows = np.arange(panel.row(GATE_FIRST), len(panel.dates), 5)
        good = 0
        for r in rows:
            members = panel.universe[r]
            valid = members & np.isfinite(values[r])
            n = int(valid.sum())
            if n >= max(MIN_COINS, MIN_COVERAGE * members.sum()) and np.unique(values[r][valid]).size >= MIN_DISTINCT:
                good += 1
        share = good / rows.size
        if share < MIN_GOOD_DAYS:
            return ComputeCheck(False, "too_few_coins_or_constant", share)
        return ComputeCheck(True, "", share)

    def _perturbed(self, key: str, window: int) -> dict[str, np.ndarray]:
        changed, apply = SHOCKS[key]
        panel = self.gate_panel
        fields = dict(panel.fields)
        for name in changed:
            fields[name] = fields[name].copy()
        for t, coins in self.plan:
            apply(fields, np.arange(t - window + 1, t + 1), coins)
        return fields

    def _perturbed_minutes(self, node: Node, key: str, window: int, fields: dict[str, np.ndarray], memo: dict | None) -> PatchedMinutes | None:
        """The minute bars of the same event ``_perturbed`` puts in the daily fields."""
        if self.minutes is None or not dsl.reads_minutes(node) or key not in MINUTE_KEYS:
            return None
        rows, coins, first = [], [], []
        for t, chosen in self.plan:
            span = np.arange(t - window + 1, t + 1)
            rows.append(np.repeat(span, chosen.size))
            coins.append(np.tile(chosen, span.size))
            first.append(np.repeat(span == span[0], chosen.size))
        rows, coins, first = np.concatenate(rows), np.concatenate(coins), np.concatenate(first)
        before = self.gate_panel.fields
        day = {
            "first": first,
            "previous_close": before["close"][rows - 1, coins],
            "new_previous_close": fields["close"][rows - 1, coins],
            "top": np.fmax(before["open"], before["close"])[rows, coins],
            "bottom": np.fmin(before["open"], before["close"])[rows, coins],
        }
        return PatchedMinutes(self.minutes, rows, coins, MINUTE_KEYS[key][1], day, known=memo)

    def _unit_minutes(self, node: Node, memo: dict | None) -> PatchedMinutes | None:
        if self.minutes is None or not dsl.reads_minutes(node):
            return None
        total = len(self.gate_panel.dates)
        rows = np.repeat(np.arange(total), self.unit_coins.size)
        coins = np.tile(self.unit_coins, total)
        return PatchedMinutes(self.minutes, rows, coins, _unit_minutes, {}, known=memo)

    def _unit_changed(self) -> dict[str, np.ndarray]:
        fields = dict(self.gate_panel.fields)
        for name in _PRICE:
            fields[name] = fields[name].copy()
            fields[name][:, self.unit_coins] *= UNIT_FACTOR
        for name in ("volume", "taker_base_vol"):
            fields[name] = fields[name].copy()
            fields[name][:, self.unit_coins] /= UNIT_FACTOR
        return fields

    def _response(self, node: Node, key: str, base: np.ndarray, memo: dict | None, deadline) -> dict[int, dict]:
        """How the rank of the perturbed coins moves when ``key`` is raised, for each window."""
        panel = self.gate_panel
        changed = _changed_by(key)
        reusable = {n: v for n, v in (memo or {}).items() if not (dsl.fields_of(n) & changed)}
        windows = {}
        for w in WINDOWS:
            fields = self._perturbed(key, w)
            minutes = self._perturbed_minutes(node, key, w, fields, memo)
            after = evaluate(node, fields, deadline=deadline, memo=dict(reusable), minutes=minutes)
            moves = []
            for t, coins in self.plan:
                before_rank = _pct_rank(base[t], panel.universe[t])[coins]
                after_rank = _pct_rank(after[t], panel.universe[t])[coins]
                moves.append(after_rank - before_rank)
            moves = np.concatenate(moves)
            moves = moves[np.isfinite(moves)]
            if moves.size == 0:
                windows[w] = {"median": float("nan"), "up": 0.0, "down": 0.0, "n": 0}
                continue
            windows[w] = {
                "median": float(np.median(moves)),
                "up": float((moves > 0).mean()),
                "down": float((moves < 0).mean()),
                "n": int(moves.size),
            }
        return windows

    def check_idea(
        self, node: Node, key: str, direction: int, base: np.ndarray, *, also: tuple[str, ...] = (), deadline=None, memo=None
    ) -> IdeaCheck:
        """``direction`` is +1 if the idea says more of the key data is bullish, -1 if bearish.

        ``also`` names key groups the idea itself is about besides its key's: an illiquidity
        idea, price move per dollar traded, is about prices as much as about volume.
        """
        raw = self._response(node, key, base, memo, deadline)
        windows = {
            w: {
                "median": direction * v["median"],
                "same": v["up"] if direction > 0 else v["down"],
                "opposite": v["down"] if direction > 0 else v["up"],
                "n": v["n"],
            }
            for w, v in raw.items()
        }
        own = max((abs(v["median"]) for v in windows.values() if v["n"]), default=0.0)
        rises = any(v["median"] >= RISE_MEDIAN and v["same"] >= RISE_SHARE for v in windows.values() if v["n"])
        falls = any(v["median"] <= -RISE_MEDIAN and v["opposite"] >= RISE_SHARE for v in windows.values() if v["n"])
        if falls:
            return IdeaCheck(False, "moves_against_the_idea", windows)
        if not rises:
            return IdeaCheck(False, "does_not_respond_to_the_idea", windows)
        if key == "return" and self._likes_a_fall(node, direction, base, memo, deadline):
            return IdeaCheck(False, "measures_move_size_not_direction", windows)
        ignored = MUST_IGNORE.get(key)
        if ignored is not None:
            response = self._response(node, ignored, base, memo, deadline)
            # Only a move the same way as the key counts: a trade-size formula that has had total
            # volume regressed out falls when volume rises, and that is not measuring volume.
            moved = max((direction * v["median"] for v in response.values() if v["n"]), default=0.0)
            if moved > MUST_IGNORE_SHARE * own:
                return IdeaCheck(False, f"measures_{ignored}_not_{key}", windows)
        unrelated = {}
        reads = dsl.fields_of(node)
        for other in KEYS:
            allowed = KEY_GROUPS[other] == KEY_GROUPS[key] or KEY_GROUPS[other] in also
            if (allowed and other not in RIVALS.get(key, ())) or not (reads & _changed_by(other)):
                continue
            response = self._response(node, other, base, memo, deadline)
            unrelated[other] = max((abs(v["median"]) for v in response.values() if v["n"]), default=0.0)
        if unrelated and max(unrelated.values()) > own:
            return IdeaCheck(False, "driven_by_unrelated_data", windows, unrelated=unrelated)
        unit_median = self._unit_response(node, base, deadline=deadline, memo=memo)
        if not unit_median <= UNIT_MEDIAN:
            return IdeaCheck(False, "depends_on_price_units", windows, unit_median, unrelated)
        return IdeaCheck(True, "", windows, unit_median, unrelated)

    def _likes_a_fall(self, node: Node, direction: int, base: np.ndarray, memo: dict | None, deadline) -> bool:
        """A rally and a fall both make a coin more volatile. A formula that moves the way the idea
        wants after either one is about the size of moves, not about prices rising."""
        response = self._response(node, "fall", base, memo, deadline)
        return any(
            direction * v["median"] >= RISE_MEDIAN and (v["up"] if direction > 0 else v["down"]) >= RISE_SHARE
            for v in response.values()
            if v["n"]
        )

    def _unit_response(self, node: Node, base: np.ndarray, *, deadline=None, memo=None) -> float:
        panel = self.gate_panel
        after = evaluate(node, self._unit_changed(), deadline=deadline, minutes=self._unit_minutes(node, memo))
        moves = []
        for t in self.as_of:
            before_rank = _pct_rank(base[t], panel.universe[t])[self.unit_coins]
            after_rank = _pct_rank(after[t], panel.universe[t])[self.unit_coins]
            moves.append(np.abs(after_rank - before_rank))
        moves = np.concatenate(moves)
        moves = moves[np.isfinite(moves)]
        return float(np.median(moves)) if moves.size else float("nan")

    def _ic_arrays(self, panel: Panel) -> tuple[np.ndarray, np.ndarray]:
        """Forward returns and style exposures for a panel. fwd[t] = close[t+2] / close[t+1] - 1."""
        cached = self._ic_inputs.get(id(panel))
        if cached is not None:
            return cached
        close, quote = panel.fields["close"], panel.fields["quote_volume"]
        tradable = np.where(quote > 0, close, np.nan)
        fwd = np.full(close.shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            fwd[:-2] = tradable[2:] / tradable[1:-1] - 1.0
        styles = np.stack([np.asarray(evaluate(dsl.parse(text), panel.fields)) for text in STYLES.values()])
        self._ic_inputs[id(panel)] = (fwd, styles)
        return fwd, styles

    def incremental_ic(self, values: np.ndarray, panel: Panel, first: str, last: str) -> ICResult:
        fwd, styles = self._ic_arrays(panel)
        raw, incremental = [], []
        for r in range(panel.row(first), panel.row(last, side="right")):
            mask = panel.universe[r] & np.isfinite(values[r]) & np.isfinite(fwd[r]) & np.isfinite(styles[:, r]).all(axis=0)
            n = int(mask.sum())
            if n < MIN_COINS:
                continue
            f, y = rank_row(values[r][mask]), rank_row(fwd[r][mask])
            design = np.column_stack([np.ones(n)] + [rank_row(s[r][mask]) for s in styles])
            beta, *_ = np.linalg.lstsq(design, f, rcond=None)
            residual = f - design @ beta
            raw.append(_corr(f, y))
            # Semi-partial correlation: what the factor adds, on the factor's own scale. A near
            # copy of a style leaves a tiny residual and scores near 0; re-ranking that residual
            # would blow rounding noise and small differences up to full size.
            incremental.append(_corr(residual, y) * np.sqrt(residual.var() / f.var()) if f.var() > 0 else float("nan"))
        raw_a, inc_a = np.asarray(raw, dtype=np.float64), np.asarray(incremental, dtype=np.float64)
        inc_a = inc_a[np.isfinite(inc_a)]
        if inc_a.size == 0:
            return ICResult(float("nan"), float("nan"), 0, float("nan"))
        spread = inc_a.std(ddof=1) if inc_a.size > 1 else float("nan")
        t_stat = float(inc_a.mean() / (spread / np.sqrt(inc_a.size))) if spread and spread > 0 else float("nan")
        return ICResult(float(inc_a.mean()), float(np.nanmean(raw_a)), int(inc_a.size), t_stat)

    def report(self, text: str) -> str:
        """What test_factor tells a model: whether the expression computes, and its IC on the
        history it is allowed to see. Rows after 2024 are not on this panel, and the idea check
        stays hidden, so a model can't tune against the verifier."""
        started = time.monotonic()
        try:
            node = dsl.parse(text)
        except DslError as exc:
            return f"error: {exc}"
        panel = self.gate_panel
        try:
            values = evaluate(node, panel.fields, deadline=started + EVAL_SECONDS, minutes=self.minutes)
        except EvalTimeout:
            return "error: this expression took too long to compute"
        except NoMinuteData as exc:
            return f"error: {exc}"
        check = self.check_compute(values)
        if not check.ok:
            return (
                "does not compute: on too many days it has too few coins with a value "
                "or nearly the same value for every coin"
            )
        first, last = SELECTION_IC
        whole = self.incremental_ic(values, panel, first, last)
        rows = range(panel.row(first), panel.row(last, side="right"))
        coins = int(np.median([(panel.universe[r] & np.isfinite(values[r])).sum() for r in rows]))
        periods = ", ".join(
            f"{name} {self.incremental_ic(values, panel, a, b).incremental:+.4f}" for name, a, b in REPORT_PERIODS
        )
        return (
            f"computes: yes, {first} to {last}, {whole.days} days, about {coins} coins a day\n"
            f"rank IC: {whole.raw:+.4f}\n"
            f"incremental IC after removing known styles: {whole.incremental:+.4f} (t = {whole.t_stat:.1f})\n"
            f"incremental IC by period: {periods}"
        )

    def score(
        self,
        text: str,
        key: str,
        direction: int,
        *,
        also: tuple[str, ...] = (),
        ic_range: tuple[str, str] = RL_IC,
        ic_panel: Panel | None = None,
        ic_minutes: MinuteBars | None = None,
    ) -> Score:
        """Run all three steps on one expression. Unusable input scores 0; it never raises."""
        started = time.monotonic()
        deadline = started + EVAL_SECONDS
        result = Score()
        try:
            node = dsl.parse(text)
        except DslError as exc:
            result.reason = f"parse: {exc}"
            result.seconds = time.monotonic() - started
            return result
        result.parsed = True
        result.skeleton = dsl.skeleton(node)
        try:
            memo: dict = {}
            base = evaluate(node, self.gate_panel.fields, deadline=deadline, memo=memo, minutes=self.minutes)
            check = self.check_compute(base)
            if not check.ok:
                result.reason = check.reason
                return result
            result.computes = True
            idea = self.check_idea(node, key, direction, base, also=also, deadline=deadline, memo=memo)
            if not idea.ok:
                result.reason = idea.reason
                return result
            panel = ic_panel if ic_panel is not None else self.dev
            values = evaluate(node, panel.fields, deadline=deadline, minutes=ic_minutes if ic_panel is not None else self.minutes)
            result.ic = self.incremental_ic(values, panel, *ic_range)
            # A formula that goes blank after the gate rows would otherwise be paid on the few days left.
            if result.ic.days < MIN_GOOD_DAYS * (panel.row(ic_range[1], side="right") - panel.row(ic_range[0])):
                result.reason = "too_few_coins_or_constant_when_scored"
                return result
            result.matches_idea = True
        except EvalTimeout:
            result.reason = "too_slow"
        except NoMinuteData as exc:
            result.reason = str(exc)
        finally:
            result.seconds = time.monotonic() - started
        return result


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")
