"""No value at row t may depend on any row after t.

Two tests from the research platform's own discipline, applied to every operator and to
random expressions built from them:
- poison: overwrite every row after t with wild values; rows up to t must not change;
- truncation: compute on the frame cut at t; rows up to t must equal the full frame's.
"""
import random

import numpy as np
import pytest

from factor_idea import ops
from factor_idea.dsl import FIELDS, parse
from factor_idea.evaluate import evaluate
from helpers import synthetic_fields

T, N = 160, 24
DATA = synthetic_fields(T, N, seed=11)
CUTS = (40, 80, 120, 159)

F = 'Factor("close")'
G = 'Factor("volume")'
EVERY_OPERATOR = [
    f"{F} + {G}", f"{F} - {G}", f"{F} * {G}", f"{F} / {G}", f"{F} ** 0.5", f"-{F}", f"{F} > {G}", f"{F} >= {G}",
    f"{F} < {G}", f"{F} <= {G}", f"{F} == {G}", f"{F} != {G}",
    f"{F}.Abs()", f"{F}.log()", f"{F}.ln()", f"{F}.sign()", f"{F}.pow(2)", f"{F}.clip(lower=1, upper=100)",
    f"{F}.max({G})", f"{F}.min({G})", f"{F}.fillna({G})", f"{F}.safe_div({G})", f"({F} > 1).where({F}, {G})",
    *[f"{F}.{name}(window=7, min_periods=3)" for name in (
        "ts_mean", "ts_std", "ts_var", "ts_skew", "ts_kurt", "ts_sum", "ts_max", "ts_min", "ts_median", "ts_rank",
        "ts_argmax", "ts_argmin", "ts_decay_linear", "ts_product", "ts_zscore",
    )],
    f"{F}.ts_quantile(window=7, q=0.3, min_periods=3)",
    f"{F}.ts_ewm_mean(span=4, min_periods=2)", f"{F}.ts_ewm_mean(span=4, window=9)",
    f"{F}.EMA(span=4, min_periods=2)", f"{F}.EMA(span=4, window=9)",
    f"{F}.lag(3)", f"{F}.delay(3)", f"{F}.diff(3)", f"{F}.pct_change(3)", f"{F}.ts_delta(3)",
    f"{F}.ts_corr({G}, 7, 3)", f"{F}.ts_cov({G}, 7, 3)",
    f"{F}.sec_rank()", f"{F}.rank(pct=False)", f"{F}.percentile()", f"{F}.zscore()", f"{F}.demean()", f"{F}.sec_demean()",
    f"{F}.sec_pct()", f"{F}.winsorize(limit=2)", f"{F}.sec_winsorize(limit=2)", f"{F}.scale(a=2)",
    f"{F}.neutralize({G})", f"{F}.orthogonalize({G})",
]


def poisoned(data, cut, seed):
    rng = np.random.default_rng(seed)
    out = {}
    for name, a in data.items():
        b = a.copy()
        wild = rng.choice([1e15, -1e15, 0.0, np.nan, 1e-15], size=b[cut + 1 :].shape)
        b[cut + 1 :] = wild
        out[name] = b
    return out


def truncated(data, cut):
    return {name: a[: cut + 1] for name, a in data.items()}


def assert_prefix_equal(a, b, cut):
    np.testing.assert_array_equal(a[: cut + 1], b[: cut + 1])


def run(text, data):
    with np.errstate(all="ignore"):
        return evaluate(parse(text), data)


@pytest.mark.parametrize("text", EVERY_OPERATOR)
@pytest.mark.parametrize("cut", CUTS)
def test_every_operator(text, cut):
    full = run(text, DATA)
    assert_prefix_equal(full, run(text, poisoned(DATA, cut, cut)), cut)
    assert_prefix_equal(full, run(text, truncated(DATA, cut)), cut)


def test_every_method_in_the_grammar_is_covered():
    from factor_idea.dsl import _METHODS

    joined = " ".join(EVERY_OPERATOR)
    for method in _METHODS:
        assert f".{method}(" in joined, method


def random_expression(rng: random.Random, depth: int) -> str:
    if depth == 0 or rng.random() < 0.15:
        return f'Factor("{rng.choice(FIELDS)}")'
    a = random_expression(rng, depth - 1)
    kind = rng.randrange(6)
    w = rng.choice([1, 2, 3, 5, 8, 13, 21, 40])
    mp = rng.choice([None, 1, max(1, w // 2)])
    if kind == 0:
        return f"({a}) {rng.choice(['+', '-', '*', '/'])} ({random_expression(rng, depth - 1)})"
    if kind == 1:
        name = rng.choice(["ts_mean", "ts_std", "ts_sum", "ts_max", "ts_min", "ts_median", "ts_rank", "ts_argmax", "ts_argmin", "ts_decay_linear", "ts_zscore", "ts_skew", "ts_kurt", "ts_var", "ts_product"])
        return f"({a}).{name}(window={w}, min_periods={mp})"
    if kind == 2:
        return f"({a}).{rng.choice(['lag', 'diff', 'pct_change'])}({w})"
    if kind == 3:
        return f"({a}).{rng.choice(['sec_rank', 'zscore', 'demean', 'sec_pct', 'scale', 'winsorize', 'sec_winsorize', 'percentile'])}()"
    if kind == 4:
        return f"({a}).{rng.choice(['Abs', 'ln', 'sign'])}()"
    choice = rng.randrange(4)
    b = random_expression(rng, depth - 1)
    if choice == 0:
        return f"({a}).ts_corr({b}, {max(w, 2)})"
    if choice == 1:
        return f"({a}).EMA(span={max(w, 2)}, window={rng.choice([None, max(w, 2) * 2])})"
    if choice == 2:
        return f"({a}).ts_ewm_mean(span={max(w, 2)})"
    return f"({a}).neutralize({b})"


@pytest.mark.parametrize("seed", range(150))
def test_random_expressions(seed):
    rng = random.Random(seed)
    text = random_expression(rng, rng.randint(2, 5))
    cut = rng.choice(CUTS[:-1])
    full = run(text, DATA)
    assert_prefix_equal(full, run(text, poisoned(DATA, cut, seed)), cut)
    assert_prefix_equal(full, run(text, truncated(DATA, cut)), cut)


def test_the_harness_catches_an_operator_that_peeks(monkeypatch):
    """A test that can't fail proves nothing: plant a look-ahead operator and watch both tests fail."""

    def peek(x, *, periods):
        out = np.full_like(x, np.nan)
        out[:-periods] = x[periods:]
        return out

    monkeypatch.setitem(ops.TABLE, "lag", peek)
    text = f"{F}.lag(3)"
    cut = 80
    full = run(text, DATA)
    with pytest.raises(AssertionError):
        assert_prefix_equal(full, run(text, poisoned(DATA, cut, 0)), cut)
    with pytest.raises(AssertionError):
        assert_prefix_equal(full, run(text, truncated(DATA, cut)), cut)


def test_universe_and_forward_returns_stay_outside_factor_values():
    """Forward returns are built from rows after t, so they must never be reachable from the grammar."""
    from factor_idea import dsl

    assert "fwd" not in dsl.FIELDS
    with pytest.raises(dsl.DslError):
        parse('Factor("fwd")')
