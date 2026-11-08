"""Check each operator against a slow, obviously correct loop written separately from it."""
import math

import numpy as np
import pytest

from factor_idea import ops
from factor_idea.dsl import parse
from factor_idea.evaluate import evaluate
from helpers import synthetic_fields

T, N = 90, 12
FIELDS = synthetic_fields(T, N, seed=3)
X = FIELDS["close"].copy()
Y = FIELDS["volume"].copy()
# Holes in the middle of a series, and values the rolling code finds awkward
rng = np.random.default_rng(1)
X[rng.random(X.shape) < 0.08] = np.nan
Y[rng.random(Y.shape) < 0.08] = np.nan
X[30, 3] = 1e12
X[31, 4] = 1e-12
DATA = dict(FIELDS, close=X, volume=Y)


def _windows(x, w):
    for t in range(x.shape[0]):
        for j in range(x.shape[1]):
            yield t, j, x[max(0, t - w + 1) : t + 1, j]


def _oracle_rolling(x, w, mp, fn):
    out = np.full(x.shape, np.nan)
    need = w if mp is None else mp
    for t, j, win in _windows(x, w):
        vals = win[np.isfinite(win)]
        if vals.size >= need:
            out[t, j] = fn(vals)
    return out


def _run(text):
    with np.errstate(all="ignore"):
        return evaluate(parse(text), DATA)


def _same(a, b, tol=1e-7):
    a, b = np.asarray(a), np.asarray(b)
    assert (np.isnan(a) == np.isnan(b)).all(), f"NaN pattern differs at {np.argwhere(np.isnan(a) != np.isnan(b))[:5].tolist()}"
    m = np.isfinite(a)
    denom = np.maximum(np.abs(b[m]), 1.0)
    assert (np.abs(a[m] - b[m]) <= tol * denom).all(), float((np.abs(a[m] - b[m]) / denom).max())


def _var(v):
    return float(np.var(v, ddof=1)) if v.size > 1 else float("nan")


def _skew(v):
    d = v - v.mean()
    m2 = (d**2).mean()
    return float((d**3).mean() / m2**1.5) if m2 > 0 else float("nan")


def _kurt(v):
    d = v - v.mean()
    m2 = (d**2).mean()
    return float((d**4).mean() / m2**2 - 3.0) if m2 > 0 else float("nan")


ROLLING = [
    ("ts_mean", lambda v: float(v.mean())),
    ("ts_sum", lambda v: float(v.sum())),
    ("ts_var", _var),
    ("ts_std", lambda v: math.sqrt(_var(v)) if v.size > 1 else float("nan")),
    ("ts_max", lambda v: float(v.max())),
    ("ts_min", lambda v: float(v.min())),
    ("ts_median", lambda v: float(np.median(v))),
    ("ts_product", lambda v: float(np.prod(v))),
    ("ts_skew", _skew),
    ("ts_kurt", _kurt),
]


@pytest.mark.parametrize("name,fn", ROLLING)
@pytest.mark.parametrize("w,mp", [(1, None), (2, 1), (5, None), (7, 3), (20, 16), (33, 1), (200, 10)])
def test_rolling(name, fn, w, mp):
    if w == 1 and name in ("ts_var", "ts_std", "ts_skew", "ts_kurt"):
        return
    got = _run(f'Factor("close").{name}(window={w}, min_periods={mp})')
    want = ops.clean(_oracle_rolling(X, w, mp, fn))
    _same(got, want, tol=1e-5 if name in ("ts_kurt", "ts_skew", "ts_var", "ts_std") else 1e-7)


@pytest.mark.parametrize("w,mp,q", [(5, None, 0.5), (9, 3, 0.25), (20, 10, 0.9), (20, 10, 0.0), (20, 10, 1.0)])
def test_quantile_nearest(w, mp, q):
    def fn(v):
        ordered = np.sort(v)
        return float(ordered[int(math.floor((v.size - 1) * q + 0.5))])

    _same(_run(f'Factor("close").ts_quantile(window={w}, q={q}, min_periods={mp})'), _oracle_rolling(X, w, mp, fn))


@pytest.mark.parametrize("w,mp", [(5, None), (9, 3), (40, 5)])
def test_ts_rank(w, mp):
    out = np.full(X.shape, np.nan)
    need = w if mp is None else mp
    for t, j, win in _windows(X, w):
        vals = win[np.isfinite(win)]
        if np.isfinite(X[t, j]) and vals.size >= need:
            out[t, j] = (vals < X[t, j]).sum() + ((vals == X[t, j]).sum() + 1) / 2
    _same(_run(f'Factor("close").ts_rank(window={w}, min_periods={mp})'), out)


@pytest.mark.parametrize("name,pick", [("ts_argmax", np.nanargmax), ("ts_argmin", np.nanargmin)])
@pytest.mark.parametrize("w,mp", [(5, None), (9, 3), (40, 5)])
def test_arg_extreme(name, pick, w, mp):
    out = np.full(X.shape, np.nan)
    need = w if mp is None else mp
    for t, j, win in _windows(X, w):
        if np.isfinite(win).sum() >= need:
            out[t, j] = pick(win)
    _same(_run(f'Factor("close").{name}(window={w}, min_periods={mp})'), out)


@pytest.mark.parametrize("w,mp", [(5, None), (9, 3), (40, 5)])
def test_decay_linear(w, mp):
    out = np.full(X.shape, np.nan)
    need = w if mp is None else mp
    for t in range(T):
        for j in range(N):
            num = den = cnt = 0
            for age in range(w):
                if t - age >= 0 and np.isfinite(X[t - age, j]):
                    weight = w - age
                    num += weight * X[t - age, j]
                    den += weight
                    cnt += 1
            if cnt >= need:
                out[t, j] = num / den
    _same(_run(f'Factor("close").ts_decay_linear(window={w}, min_periods={mp})'), out)


@pytest.mark.parametrize("w,mp", [(5, None), (20, 10), (45, 3)])
def test_zscore_ts(w, mp):
    out = np.full(X.shape, np.nan)
    need = w if mp is None else mp
    for t, j, win in _windows(X, w):
        vals = win[np.isfinite(win)]
        if vals.size >= max(need, 2) and np.isfinite(X[t, j]):
            std = vals.std(ddof=1)
            out[t, j] = 0.0 if std == 0 else (X[t, j] - vals.mean()) / std
    _same(_run(f'Factor("close").ts_zscore(window={w}, min_periods={mp})'), out)


@pytest.mark.parametrize("w,mp", [(5, None), (20, 10), (45, 3)])
def test_corr_and_cov(w, mp):
    corr = np.full(X.shape, np.nan)
    cov = np.full(X.shape, np.nan)
    need = w if mp is None else mp
    for t in range(T):
        for j in range(N):
            a = X[max(0, t - w + 1) : t + 1, j]
            b = Y[max(0, t - w + 1) : t + 1, j]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= max(need, 2):
                cov[t, j] = np.cov(a[m], b[m], ddof=1)[0, 1]
                if a[m].std() > 0 and b[m].std() > 0:
                    corr[t, j] = np.corrcoef(a[m], b[m])[0, 1]
    _same(_run(f'Factor("close").ts_corr(Factor("volume"), {w}, {mp})'), corr, tol=1e-5)
    _same(_run(f'Factor("close").ts_cov(Factor("volume"), {w}, {mp})'), cov, tol=1e-5)


@pytest.mark.parametrize("span,mp,window", [(3, None, None), (5, 3, 12), (10, 8, 100), (2, 1, 3)])
def test_ewm_adjusted(span, mp, window):
    w = 10 * span if window is None else window
    need = 1 if mp is None else mp
    decay = 1 - 2 / (span + 1)
    out = np.full(X.shape, np.nan)
    for t in range(T):
        for j in range(N):
            if not np.isfinite(X[t, j]):
                continue
            num = den = cnt = 0
            for age in range(w):
                if t - age >= 0 and np.isfinite(X[t - age, j]):
                    num += decay**age * X[t - age, j]
                    den += decay**age
                    cnt += 1
            if cnt >= need:
                out[t, j] = num / den
    extra = f", window={window}" if window else ""
    _same(_run(f'Factor("close").ts_ewm_mean(span={span}, min_periods={mp}{extra})'), out)


@pytest.mark.parametrize("span,mp,window", [(3, None, None), (5, 3, 12), (10, 8, 100), (2, 1, 3)])
def test_ema_legacy(span, mp, window):
    """pandas.ewm(adjust=False) with gaps, restarted at the window's first value."""
    w = 10 * span if window is None else window
    need = span if mp is None else mp
    decay = 1 - 2 / (span + 1)
    alpha = 1 - decay
    out = np.full(X.shape, np.nan)
    for t in range(need, T):
        for j in range(N):
            if not np.isfinite(X[t, j]):
                continue
            rows = [r for r in range(max(0, t - w + 1), t + 1) if np.isfinite(X[r, j])]
            if len(rows) < need:
                continue
            m = X[rows[0], j]
            for prev, r in zip(rows, rows[1:]):
                carry = decay ** (r - prev)
                m = (carry * m + alpha * X[r, j]) / (carry + alpha)
            out[t, j] = m
    extra = f", window={window}" if window else ""
    _same(_run(f'Factor("close").EMA(span={span}, min_periods={mp}{extra})'), out)


def test_lag_diff_pct():
    prev = np.vstack([np.full((2, N), np.nan), X[:-2]])
    _same(_run('Factor("close").lag(2)'), prev)
    _same(_run('Factor("close").diff(2)'), X - prev)
    _same(_run('Factor("close").pct_change(2)'), ops.clean((X - prev) / prev))
    _same(_run('Factor("close").lag(365)'), np.full(X.shape, np.nan))


def _rank_oracle(row):
    out = np.full(row.shape, np.nan)
    vals = row[np.isfinite(row)]
    for j in range(row.size):
        if np.isfinite(row[j]):
            out[j] = (vals < row[j]).sum() + ((vals == row[j]).sum() + 1) / 2
    return out


def test_cross_section():
    tied = np.round(np.nan_to_num(X, nan=0.0) % 3.0)
    tied = np.where(np.isnan(X), np.nan, tied)
    data = dict(DATA, open=tied)
    with np.errstate(all="ignore"):
        got = evaluate(parse('Factor("open").rank(pct=False)'), data)
        pct = evaluate(parse('Factor("open").rank()'), data)
    want = np.vstack([_rank_oracle(r) for r in tied])
    _same(got, want)
    _same(pct, want / np.isfinite(tied).sum(axis=1, keepdims=True))


def _row_oracle(x, fn):
    out = np.full(x.shape, np.nan)
    for i, row in enumerate(x):
        m = np.isfinite(row)
        if m.any():
            out[i, m] = fn(row[m])
    return out


def test_zscore_demean_pct_scale_winsorize():
    _same(_run('Factor("close").zscore()'), _row_oracle(X, lambda v: (v - v.mean()) / v.std(ddof=1) if v.size > 1 and v.std() > 0 else np.zeros_like(v)))
    _same(_run('Factor("close").demean()'), _row_oracle(X, lambda v: v - v.mean()))
    _same(_run('Factor("close").sec_pct()'), _row_oracle(X, lambda v: v / v.sum()))
    _same(_run('Factor("close").scale(a=2)'), _row_oracle(X, lambda v: v * 2 / np.abs(v).sum()))
    _same(
        _run('Factor("close").winsorize(limit=1.5)'),
        _row_oracle(X, lambda v: np.clip(v, v.mean() - 1.5 * v.std(ddof=1), v.mean() + 1.5 * v.std(ddof=1)) if v.size > 1 else v),
    )
    _same(
        _run('Factor("close").sec_winsorize(limit=1.5)'),
        _row_oracle(X, lambda v: np.clip(v, v.mean() - 1.5 * v.std(ddof=0), v.mean() + 1.5 * v.std(ddof=0))),
    )


def test_neutralize_residual_is_orthogonal():
    y, z = np.log(FIELDS["close"]), np.log(FIELDS["volume"])
    with np.errstate(all="ignore"):
        res = evaluate(parse('Factor("close").ln().neutralize(Factor("volume").ln())'), FIELDS)
    for i in range(T):
        m = np.isfinite(res[i])
        if m.sum() > 3:
            assert abs(res[i][m].sum()) < 1e-8
            assert abs((res[i][m] * z[i][m]).sum()) < 1e-6
    # too few points: nothing is fitted
    small = {k: v[:, :3] for k, v in FIELDS.items()}
    with np.errstate(all="ignore"):
        got = evaluate(parse('Factor("close").neutralize(Factor("volume"), Factor("open"))'), small)
    assert np.isnan(got).all()


def test_elementwise():
    a = np.array([[1.0, -2.0, 0.0, np.nan, 4.0]])
    b = np.array([[2.0, 0.0, 0.0, 1.0, np.nan]])
    data = {"open": a, "close": b}

    def run(text):
        with np.errstate(all="ignore"):
            return evaluate(parse(text), data)

    _same(run('Factor("open") / Factor("close")'), [[0.5, np.nan, np.nan, np.nan, np.nan]])
    _same(run('Factor("open").safe_div(Factor("close"))'), [[0.5, np.nan, np.nan, np.nan, np.nan]])
    _same(run('Factor("open").ln()'), [[0.0, np.nan, np.nan, np.nan, math.log(4)]])
    _same(run('Factor("open").log()'), [[0.0, np.nan, np.nan, np.nan, math.log(4)]])
    _same(run('Factor("open").Abs()'), [[1, 2, 0, np.nan, 4]])
    _same(run('Factor("open").sign()'), [[1, -1, 0, np.nan, 1]])
    _same(run('Factor("open") ** 0.5'), [[1, np.nan, 0, np.nan, 2]])
    _same(run('Factor("open").max(Factor("close"))'), [[2, 0, 0, 1, 4]])
    _same(run('Factor("open").min(Factor("close"))'), [[1, -2, 0, 1, 4]])
    _same(run('Factor("open").fillna(Factor("close"))'), [[1, -2, 0, 1, 4]])
    _same(run('Factor("open").clip(lower=0, upper=2)'), [[1, 0, 0, np.nan, 2]])
    _same(run('Factor("open") > Factor("close")'), [[0, 0, 0, np.nan, np.nan]])
    _same(run('Factor("open") == 0'), [[0, 0, 1, np.nan, 0]])
    _same(run('where(Factor("open") > 0, 1, -1)'), [[1, -1, -1, -1, 1]])
    _same(run('-Factor("open")'), [[-1, 2, -0.0, np.nan, -4]])


def test_a_huge_early_value_leaves_no_residue_in_a_later_sum():
    x = np.ones((64, 1))
    x[0, 0] = 1e30
    got = ops.ts_sum(x, window=8)
    assert (got[8:, 0] == 8.0).all()


def test_evaluate_does_not_touch_its_inputs():
    copies = {k: v.copy() for k, v in FIELDS.items()}
    text = '(Factor("close").ts_mean(5) + Factor("volume").ts_max(3)).rank().EMA(span=3).ts_rank(4).zscore()'
    with np.errstate(all="ignore"):
        evaluate(parse(text), FIELDS)
    for k, v in copies.items():
        np.testing.assert_array_equal(FIELDS[k], v)
