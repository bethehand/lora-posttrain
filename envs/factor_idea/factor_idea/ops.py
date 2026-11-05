"""Operators over daily panels: T x N float64 arrays where NaN means no value. The elementwise
ones and those at the end also run inside a day, on arrays of minutes x coin-days.

Semantics follow the research platform's operators on valid inputs. The one
deliberate difference: an invalid result (x / 0, log of a non-positive number)
is always NaN here, never +/-inf, so it can't poison a later rolling window.

Every value at row t is computed from rows <= t. Rolling sums are built from
power-of-two blocks instead of differences of a cumulative sum: a sum never
subtracts, so one huge early value can't leave rounding residue in later rows.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_CHUNK = 64


def clean(x: np.ndarray) -> np.ndarray:
    """Map +/-inf to NaN."""
    out = np.array(x, dtype=np.float64)
    out[np.isinf(out)] = np.nan
    return out


# Elementwise


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b


def safe_div(a, b):
    return np.where(np.equal(b, 0), np.nan, np.divide(a, b))


def power(a, b=None, *, exp=None):
    return np.power(a, exp if b is None else b)


def neg(a):
    return -a


def abs_(a):
    return np.abs(a)


def log(a):
    return np.where(a > 0, np.log(a), np.nan)


def sign(a):
    return np.sign(a)


def clip(a, *, lower=None, upper=None):
    return np.clip(a, -np.inf if lower is None else lower, np.inf if upper is None else upper)


def max_(a, b):
    return np.fmax(a, b)


def min_(a, b):
    return np.fmin(a, b)


def fillna(a, b):
    return np.where(np.isnan(a), b, a)


def where(cond, a, b):
    return np.where(np.nan_to_num(cond, nan=0.0) != 0.0, a, b)


def _compare(fn: Callable) -> Callable:
    def compare(a, b):
        return np.where(np.isnan(a) | np.isnan(b), np.nan, fn(a, b).astype(np.float64))

    return compare


# Time series: rolling windows that skip NaN


def _trailing_sum(x: np.ndarray, w: int) -> np.ndarray:
    """Sum of each row's last w rows; rows before the frame count as 0."""
    n = x.shape[0]
    total = np.zeros_like(x)
    block = x.copy()
    size, done = 1, 0
    while size <= w:
        if w & size:
            if done < n:
                total[done:] += block[: n - done]
            done += size
        block[size:] += block[:-size].copy()
        size *= 2
    return total


def _present(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x).astype(np.float64)


def _count(x: np.ndarray, w: int) -> np.ndarray:
    return _trailing_sum(_present(x), w)


def _mask_short(out: np.ndarray, count: np.ndarray, w: int, min_periods: int | None) -> np.ndarray:
    return np.where(count >= (w if min_periods is None else min_periods), out, np.nan)


def ts_sum(x, *, window, min_periods=None):
    return _mask_short(_trailing_sum(np.nan_to_num(x, nan=0.0), window), _count(x, window), window, min_periods)


def ts_mean(x, *, window, min_periods=None):
    n = _count(x, window)
    return _mask_short(_trailing_sum(np.nan_to_num(x, nan=0.0), window) / n, n, window, min_periods)


def _mean_var(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = _count(x, w)
    filled = np.nan_to_num(x, nan=0.0)
    s1 = _trailing_sum(filled, w)
    s2 = _trailing_sum(filled * filled, w)
    mean = s1 / n
    var = (s2 - s1 * mean) / (n - 1)
    var = np.where(n < 2, np.nan, var)
    # Below this size the difference above is rounding error: the window is constant.
    var = np.where(var <= 1e-15 * mean * mean, 0.0, var)
    return mean, var, n


def ts_var(x, *, window, min_periods=None):
    _, var, n = _mean_var(x, window)
    return _mask_short(var, n, window, min_periods)


def ts_std(x, *, window, min_periods=None):
    _, var, n = _mean_var(x, window)
    return _mask_short(np.sqrt(var), n, window, min_periods)


def ts_zscore(x, *, window, min_periods=None):
    mean, var, n = _mean_var(x, window)
    std = np.sqrt(var)
    out = np.where(std == 0, np.where(np.isnan(x), np.nan, 0.0), (x - mean) / std)
    return _mask_short(out, n, window, min_periods)


def _extreme(x: np.ndarray, w: int, fn: Callable) -> np.ndarray:
    block = x.copy()
    size = 1
    while size * 2 <= w:
        block[size:] = fn(block[size:], block[:-size])
        size *= 2
    rest = w - size
    if rest == 0:
        return block
    out = block.copy()
    out[rest:] = fn(block[rest:], block[:-rest])
    return out


def ts_max(x, *, window, min_periods=None):
    return _mask_short(_extreme(x, window, np.fmax), _count(x, window), window, min_periods)


def ts_min(x, *, window, min_periods=None):
    return _mask_short(_extreme(x, window, np.fmin), _count(x, window), window, min_periods)


def _per_window(x: np.ndarray, w: int, fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """fn(windows of shape T x n x w, validity of the same shape) -> T x n, in chunks of columns."""
    t, n = x.shape
    padded = np.vstack([np.full((w - 1, n), np.nan), x]) if w > 1 else x
    view = sliding_window_view(padded, w, axis=0)
    out = np.empty((t, n), dtype=np.float64)
    for start in range(0, n, _CHUNK):
        v = view[:, start : start + _CHUNK]
        out[:, start : start + _CHUNK] = fn(v, np.isfinite(v))
    return out


def _moment_ratio(x, window, min_periods, fn):
    def compute(v, valid):
        n = valid.sum(axis=-1)
        filled = np.where(valid, v, 0.0)
        mean = filled.sum(axis=-1) / n
        d = np.where(valid, v - mean[..., None], 0.0)
        m2 = (d * d).sum(axis=-1) / n
        return fn(d, m2, n)

    return _mask_short(_per_window(x, window, compute), _count(x, window), window, min_periods)


def ts_skew(x, *, window, min_periods=None):
    return _moment_ratio(x, window, min_periods, lambda d, m2, n: (d**3).sum(axis=-1) / n / np.power(m2, 1.5))


def ts_kurt(x, *, window, min_periods=None):
    return _moment_ratio(x, window, min_periods, lambda d, m2, n: (d**4).sum(axis=-1) / n / (m2 * m2) - 3.0)


def _order_statistic(x, window, min_periods, pick):
    def compute(v, valid):
        n = valid.sum(axis=-1)
        ordered = np.sort(v, axis=-1)  # NaN sorts last
        return pick(ordered, n)

    return _mask_short(_per_window(x, window, compute), _count(x, window), window, min_periods)


def _take(ordered: np.ndarray, index: np.ndarray) -> np.ndarray:
    return np.take_along_axis(ordered, np.clip(index, 0, None)[..., None].astype(np.int64), axis=-1)[..., 0]


def ts_median(x, *, window, min_periods=None):
    return _order_statistic(
        x, window, min_periods, lambda s, n: 0.5 * (_take(s, (n - 1) // 2) + _take(s, n // 2))
    )


def ts_quantile(x, *, window, q, min_periods=None):
    return _order_statistic(x, window, min_periods, lambda s, n: _take(s, np.floor((n - 1) * q + 0.5)))


def ts_rank(x, *, window, min_periods=None):
    """Average rank (1-based) of the latest value among its window's values."""

    def compute(v, valid):
        latest = v[..., -1:]
        less = (v < latest).sum(axis=-1)
        equal = (v == latest).sum(axis=-1)
        return np.where(valid[..., -1], less + (equal + 1) / 2.0, np.nan)

    return _mask_short(_per_window(x, window, compute), _count(x, window), window, min_periods)


def _arg_extreme(x, window, min_periods, fill, arg):
    t = x.shape[0]
    # Rows before the frame are not part of the window, so positions count from its real start.
    offset = np.maximum(window - 1 - np.arange(t), 0)[:, None].astype(np.float64)

    def compute(v, valid):
        return arg(np.where(valid, v, fill), axis=-1).astype(np.float64)

    return _mask_short(_per_window(x, window, compute) - offset, _count(x, window), window, min_periods)


def ts_argmax(x, *, window, min_periods=None):
    return _arg_extreme(x, window, min_periods, -np.inf, np.argmax)


def ts_argmin(x, *, window, min_periods=None):
    return _arg_extreme(x, window, min_periods, np.inf, np.argmin)


def ts_decay_linear(x, *, window, min_periods=None):
    """Weights 1..window with the latest heaviest, renormalised over the values present."""
    weights = np.arange(1.0, window + 1.0)

    def compute(v, valid):
        return (np.where(valid, v, 0.0) * weights).sum(axis=-1) / (valid * weights).sum(axis=-1)

    return _mask_short(_per_window(x, window, compute), _count(x, window), window, min_periods)


def ts_product(x, *, window, min_periods=None):
    def compute(v, valid):
        return np.prod(np.where(valid, v, 1.0), axis=-1)

    return _mask_short(_per_window(x, window, compute), _count(x, window), window, min_periods)


def _ema_decay(span: int) -> float:
    return 1.0 - 2.0 / (float(span) + 1.0)


def _decayed_sum(x: np.ndarray, decay: float, w: int) -> np.ndarray:
    """Sum of x[t - i] * decay**i over i < w; rows before the frame count as 0."""
    n = x.shape[0]
    total = np.zeros_like(x)
    block = x.copy()
    size, done = 1, 0
    while size <= w:
        if w & size:
            if done < n:
                total[done:] += (decay**done) * block[: n - done]
            done += size
        block[size:] += (decay**size) * block[:-size].copy()
        size *= 2
    return total


def ts_ewm_mean(x, *, span, min_periods=None, window=None):
    """Weighted mean of the last ``window`` rows' values with weight decay**age."""
    w = 10 * span if window is None else window
    decay = _ema_decay(span)
    mp = 1 if min_periods is None else min_periods
    out = _decayed_sum(np.nan_to_num(x, nan=0.0), decay, w) / _decayed_sum(_present(x), decay, w)
    return np.where(np.isfinite(x) & (_count(x, w) >= mp), out, np.nan)


def _compose_trailing(a: np.ndarray, c: np.ndarray, length: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each cell, compose the maps m -> a*m + c of its last ``length`` rows into one (A, C)."""
    t, n = a.shape
    block_a, block_c = a.copy(), c.copy()
    total_a, total_c = np.ones((t, n)), np.zeros((t, n))
    end = np.broadcast_to(np.arange(t)[:, None], (t, n)).copy()
    longest = int(length.max(initial=0))
    size = 1
    while size <= longest:
        use = (length & size) != 0
        # The next block holds older rows than those composed so far, so it applies first.
        total_c = np.where(use, total_c + total_a * np.take_along_axis(block_c, end, axis=0), total_c)
        total_a = np.where(use, total_a * np.take_along_axis(block_a, end, axis=0), total_a)
        end = np.where(use, end - size, end)
        block_c[size:] += block_a[size:] * block_c[:-size]
        block_a[size:] *= block_a[:-size].copy()
        size *= 2
    return total_a, total_c


def ema(x, *, span=10, min_periods=None, window=None):
    """pandas ewm(span, adjust=False) restarted at the first value of each row's last ``window`` rows."""
    w = 10 * span if window is None else window
    mp = span if min_periods is None else min_periods
    t, n = x.shape
    observed = np.isfinite(x)
    values = np.nan_to_num(x, nan=0.0)
    decay = _ema_decay(span)
    alpha = 1.0 - decay
    row = np.broadcast_to(np.arange(t)[:, None], (t, n))
    last_seen = np.maximum.accumulate(np.where(observed, row, -1), axis=0)
    previous = np.vstack([np.full((1, n), -1), last_seen[:-1]])
    gap = np.where(previous < 0, 1, row - previous)
    a = np.full((t, n), decay)
    c = alpha * values
    after_gap = observed & (gap > 1)
    carry = np.power(decay, gap[after_gap])
    a[after_gap] = carry / (carry + alpha)
    c[after_gap] = alpha / (carry + alpha) * values[after_gap]
    a[~observed] = 1.0
    c[~observed] = 0.0
    next_seen = np.minimum.accumulate(np.where(observed, row, t)[::-1], axis=0)[::-1]
    first = np.take_along_axis(next_seen, np.maximum(row - w + 1, 0), axis=0)
    total_a, total_c = _compose_trailing(a, c, np.where(observed, row - first, 0))
    out = total_a * np.take_along_axis(values, np.minimum(first, t - 1), axis=0) + total_c
    out = np.where(observed & (_count(x, w) >= mp), out, np.nan)
    out[: min(mp, t)] = np.nan  # the old research stack dropped these rows
    return out


def lag(x, *, periods):
    out = np.full_like(x, np.nan)
    if periods < x.shape[0]:
        out[periods:] = x[:-periods]
    return out


def diff(x, *, periods=1):
    return x - lag(x, periods=periods)


def pct_change(x, *, periods=1):
    previous = lag(x, periods=periods)
    return (x - previous) / previous


def _pair_sums(x, y, w):
    both = np.isfinite(x) & np.isfinite(y)
    fx, fy = np.where(both, x, 0.0), np.where(both, y, 0.0)
    n = _trailing_sum(both.astype(np.float64), w)
    sx, sy = _trailing_sum(fx, w), _trailing_sum(fy, w)
    return n, sx, sy, _trailing_sum(fx * fy, w), fx, fy


def ts_cov(x, y, *, window, min_periods=None):
    n, sx, sy, sxy, _, _ = _pair_sums(x, y, window)
    out = np.where(n >= 2, (sxy - sx * sy / n) / (n - 1), np.nan)
    return _mask_short(out, n, window, min_periods)


def ts_corr(x, y, *, window, min_periods=None):
    n, sx, sy, sxy, fx, fy = _pair_sums(x, y, window)
    sxx, syy = _trailing_sum(fx * fx, window), _trailing_sum(fy * fy, window)
    vx, vy = sxx - sx * sx / n, syy - sy * sy / n
    constant = (vx <= 1e-15 * sx * sx / n) | (vy <= 1e-15 * sy * sy / n)
    out = np.where(constant | (n < 2), np.nan, np.clip((sxy - sx * sy / n) / np.sqrt(vx * vy), -1.0, 1.0))
    return _mask_short(out, n, window, min_periods)


# Cross section: each row on its own, over every coin with a value that day


def rank_row(row: np.ndarray) -> np.ndarray:
    """Average 1-based ranks of a row's finite values; NaN stays NaN."""
    out = np.full_like(row, np.nan)
    mask = np.isfinite(row)
    values = row[mask]
    if values.size == 0:
        return out
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    first_of_group = np.r_[True, ordered[1:] != ordered[:-1]]
    group = np.cumsum(first_of_group) - 1
    sizes = np.bincount(group)
    average = np.cumsum(sizes) - (sizes - 1) / 2.0
    ranks = np.empty_like(values)
    ranks[order] = average[group]
    out[mask] = ranks
    return out


def rank(x, *, pct=True):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = rank_row(x[i])
    if pct:
        out = out / np.isfinite(x).sum(axis=1, keepdims=True)
    return out


def _row_stats(x: np.ndarray, ddof: int) -> tuple[np.ndarray, np.ndarray]:
    n = np.isfinite(x).sum(axis=1, keepdims=True).astype(np.float64)
    mean = np.nansum(x, axis=1, keepdims=True) / n
    ss = np.nansum((x - mean) ** 2, axis=1, keepdims=True)
    std = np.sqrt(np.where(n > ddof, ss / (n - ddof), np.nan))
    return mean, std


def zscore(x):
    mean, std = _row_stats(x, 1)
    degenerate = np.isnan(std) | (std == 0)
    return np.where(degenerate, np.where(np.isnan(x), np.nan, 0.0), (x - mean) / std)


def demean(x):
    mean, _ = _row_stats(x, 0)
    return x - mean


def sec_pct(x):
    return x / np.nansum(x, axis=1, keepdims=True)


def _winsorize(x, limit, ddof):
    mean, std = _row_stats(x, ddof)
    degenerate = np.isnan(std) | (std == 0)
    return np.where(degenerate, x, np.clip(x, mean - limit * std, mean + limit * std))


def winsorize(x, *, limit=3.0):
    return _winsorize(x, limit, 1)


def sec_winsorize(x, *, limit=3.0):
    return _winsorize(x, limit, 0)


def scale(x, *, a=1.0):
    denom = np.nansum(np.abs(x), axis=1, keepdims=True)
    return np.where(denom == 0, np.where(np.isnan(x), np.nan, 0.0), x * a / denom)


def neutralize(y, *xs):
    """Per row: residual of y regressed on xs with an intercept. Needs more points than parameters."""
    out = np.full_like(y, np.nan)
    k = len(xs)
    for i in range(y.shape[0]):
        mask = np.isfinite(y[i])
        for x in xs:
            mask &= np.isfinite(x[i])
        count = int(mask.sum())
        if count < k + 2:
            continue
        design = np.column_stack([np.ones(count)] + [x[i][mask] for x in xs])
        try:
            beta, *_ = np.linalg.lstsq(design, y[i][mask], rcond=None)
        except np.linalg.LinAlgError:
            continue
        out[i, mask] = y[i][mask] - design @ beta
    return out


# Inside one day: rows are that day's minutes, columns are coin-days


def between(x, *, start_hour, end_hour):
    """Only the bars that close after start_hour and by end_hour, UTC. A day's first row closes at 00:01."""
    per_hour = x.shape[0] // 24
    out = np.full_like(x, np.nan)
    out[start_hour * per_hour : end_hour * per_hour] = x[start_hour * per_hour : end_hour * per_hour]
    return out


def _no_nan(x: np.ndarray) -> np.ndarray:
    # Values here are finite or NaN, so this skips the inf checks that make np.nan_to_num slow.
    return np.where(np.isnan(x), 0.0, x)


def _day_mean_var(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.isfinite(x).sum(axis=0).astype(np.float64)
    filled = _no_nan(x)
    s1 = filled.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / n
        var = ((filled * filled).sum(axis=0) - s1 * mean) / (n - 1)
    var = np.where(n < 2, np.nan, var)
    # Below this size the difference above is rounding error: the day is constant.
    return mean, np.where(var <= 1e-15 * mean * mean, 0.0, var), n


def zscore_in_day(x):
    mean, var, _ = _day_mean_var(x)
    std = np.sqrt(var)
    return np.where(std == 0, np.where(np.isnan(x), np.nan, 0.0), (x - mean) / std)


def day_sum(x):
    return np.where(np.isfinite(x).any(axis=0), _no_nan(x).sum(axis=0), np.nan)


def day_mean(x):
    return _day_mean_var(x)[0]


def day_std(x):
    return np.sqrt(_day_mean_var(x)[1])


def day_max(x):
    return np.fmax.reduce(x, axis=0)


def day_min(x):
    return np.fmin.reduce(x, axis=0)


def _day_pick(x: np.ndarray, position: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x).any(axis=0), x[position, np.arange(x.shape[1])], np.nan)


def day_first(x):
    return _day_pick(x, np.argmax(np.isfinite(x), axis=0))


def day_last(x):
    return _day_pick(x, x.shape[0] - 1 - np.argmax(np.isfinite(x[::-1]), axis=0))


def day_center(x):
    """When in the day a positive amount falls: 0 is the day's start and 1 its end. Negative values count as nothing."""
    amount = np.maximum(_no_nan(x), 0.0)
    position = (np.arange(x.shape[0]) + 0.5) / x.shape[0]
    total = amount.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (amount * position[:, None]).sum(axis=0) / total, np.nan)


def day_corr(x, y):
    both = np.isfinite(x) & np.isfinite(y)
    n = both.sum(axis=0).astype(np.float64)
    fx, fy = np.where(both, x, 0.0), np.where(both, y, 0.0)
    sx, sy = fx.sum(axis=0), fy.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        vx, vy = (fx * fx).sum(axis=0) - sx * sx / n, (fy * fy).sum(axis=0) - sy * sy / n
        constant = (vx <= 1e-15 * sx * sx / n) | (vy <= 1e-15 * sy * sy / n)
        out = np.clip(((fx * fy).sum(axis=0) - sx * sy / n) / np.sqrt(vx * vy), -1.0, 1.0)
    return np.where(constant | (n < 2), np.nan, out)


TABLE: dict[str, Callable] = {
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": div,
    "safe_div": safe_div,
    "pow": power,
    "neg": neg,
    "abs": abs_,
    "log": log,
    "ln": log,
    "sign": sign,
    "clip": clip,
    "max": max_,
    "min": min_,
    "fillna": fillna,
    "where": where,
    "gt": _compare(np.greater),
    "ge": _compare(np.greater_equal),
    "lt": _compare(np.less),
    "le": _compare(np.less_equal),
    "eq": _compare(np.equal),
    "ne": _compare(np.not_equal),
    "ts_sum": ts_sum,
    "ts_mean": ts_mean,
    "ts_std": ts_std,
    "ts_var": ts_var,
    "ts_skew": ts_skew,
    "ts_kurt": ts_kurt,
    "ts_max": ts_max,
    "ts_min": ts_min,
    "ts_median": ts_median,
    "ts_quantile": ts_quantile,
    "ts_rank": ts_rank,
    "ts_argmax": ts_argmax,
    "ts_argmin": ts_argmin,
    "ts_decay_linear": ts_decay_linear,
    "ts_product": ts_product,
    "ts_zscore": ts_zscore,
    "ts_ewm_mean": ts_ewm_mean,
    "ema": ema,
    "lag": lag,
    "diff": diff,
    "pct_change": pct_change,
    "ts_corr": ts_corr,
    "ts_cov": ts_cov,
    "rank": rank,
    "zscore": zscore,
    "demean": demean,
    "sec_pct": sec_pct,
    "winsorize": winsorize,
    "sec_winsorize": sec_winsorize,
    "scale": scale,
    "neutralize": neutralize,
    "between": between,
    "zscore_in_day": zscore_in_day,
    "day_sum": day_sum,
    "day_mean": day_mean,
    "day_std": day_std,
    "day_max": day_max,
    "day_min": day_min,
    "day_first": day_first,
    "day_last": day_last,
    "day_center": day_center,
    "day_corr": day_corr,
}
