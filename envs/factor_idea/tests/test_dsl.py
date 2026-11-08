import pytest

from factor_idea import dsl
from factor_idea.dsl import Const, DslError, Field, Op, parse

# Real research expressions, with the data they don't have here replaced by fields it has
ACCEPTED = [
    'Factor("close")',
    '(Factor("close").safe_div(Factor("close").lag(periods=1)) - 1)',
    '((Factor("close").safe_div(Factor("close").lag(periods=1)) - 1).Abs() + 1).ln().safe_div(Factor("quote_volume")).ts_mean(min_periods=5, window=6)',
    '(-1 * ((Factor("close").safe_div(Factor("close").lag(periods=1)) - 1).demean() ** 2).ts_median(min_periods=12, window=15))',
    "((Factor('close') / Factor('close').delay(1) - 1).Abs() + 1).ln() / Factor('quote_volume')",
    "Factor('close').ts_corr(Factor('volume'), 30) * -1",
    '(Factor("volume") ** 2.5).ts_mean(min_periods=8, window=10).EMA(min_periods=8, span=10, window=100)',
    'Factor("close").EMA(5)',
    'Factor("close").EMA(n=5)',
    'Factor("close").ts_ewm_mean(span=5, min_periods=3, window=40)',
    'Factor("close").ts_mean(5).neutralize(Factor("quote_volume").ln(), Factor("volume").ln())',
    'neutralize(Factor("close"), Factor("volume"))',
    'where(Factor("close") >= Factor("open"), Factor("volume"), 0)',
    '(Factor("close") >= 1).where(Factor("volume"), Factor("quote_volume"))',
    'safe_div(Factor("close"), Factor("volume"))',
    'safe_div(1, Factor("volume"))',
    'Factor("close").sec_rank(pct=False)',
    'Factor("close").rank()',
    'Factor("close").clip(lower=-1, upper=1)',
    'Factor("close").clip(0)',
    'Factor("close").winsorize(limit=2.5).sec_winsorize().scale(a=2)',
    'Factor("close").ts_quantile(window=20, q=0.5, min_periods=10)',
    'Factor("close").fillna(0).max(Factor("open")).min(1.5)',
    '2 ** Factor("close").sign()',
    '-Factor("close") + +Factor("open")',
    'Factor("close").pow(0.5)',
    'Factor("close").ts_mean(window=5, min_periods=None)',
]

REJECTED = [
    ("", "empty"),
    ("1 + 2", "at least one Factor"),
    ('Factor("close").lag(-1)', "positive integer"),
    ('Factor("close").lag(periods=-1)', "positive integer"),
    ('Factor("close").lag(0)', "positive integer"),
    ('Factor("close").delay(-2)', "positive integer"),
    ('Factor("close").diff(-1)', "positive integer"),
    ('Factor("close").ts_mean(window=-5)', "positive integer"),
    ('Factor("close").ts_mean(window=0)', "positive integer"),
    ('Factor("close").ts_mean(window=5.5)', "positive integer"),
    ('Factor("close").ts_mean(window=True)', "positive integer"),
    ('Factor("close").ts_mean()', "requires window"),
    ('Factor("close").ts_mean(5, 6)', "must not exceed"),
    ('Factor("close").ts_mean(window=366)', "too long"),
    ('Factor("close").lag(366)', "at most"),
    ('Factor("close").EMA(121)', "at most"),
    ('Factor("close").EMA(5, 51)', "must not exceed"),
    ('Factor("close").ts_mean(5, min_periods=4, window=5)', "twice"),
    ('Factor("close").ts_quantile(window=5, q=2)', "between 0 and 1"),
    ('Factor("close").clip()', "lower or upper"),
    ('Factor("close").winsorize(limit=0)', "positive"),
    ('Factor("nope")', "unknown field"),
    ('Factor("log_mcap")', "unknown field"),
    ('Factor("close", "1d")', "exactly one"),
    ("Factor(close)", "in quotes"),
    ('Alpha("x")', "not available"),
    ('Factor("close").sec_rank(index_pool="usdt_perp_max")', "pool"),
    ('Factor("close").neutralize(Factor("open"), pool="p")', "pool"),
    ('Factor("close").index_pool(pool="p")', "unsupported method"),
    ('Factor("close").shift(-1)', "unsupported method"),
    ('Factor("close").lead(1)', "unsupported method"),
    ('Factor("close").__class__', "unsupported syntax"),
    ('Factor("close").__class__.__mro__', "unsupported syntax"),
    ('__import__("os").system("true")', "unsupported function"),
    ('open("/etc/passwd")', "unsupported function"),
    ('Factor("close")[0]', "unsupported syntax"),
    ('[Factor("close") for _ in range(9)]', "unsupported syntax"),
    ('(lambda: Factor("close"))()', "unsupported function call"),
    ('Factor("close") if 1 else Factor("open")', "unsupported syntax"),
    ('Factor("close") and Factor("open")', "unsupported syntax"),
    ('Factor("close") % 2', "unsupported binary"),
    ('Factor("close") // 2', "unsupported binary"),
    ('1 < Factor("close") < 2', "chained"),
    ('"close"', "unsupported literal"),
    ('Factor("close") + "a"', "unsupported literal"),
    ('Factor("close") + True', "unsupported literal"),
    ('Factor("close") + None', "unsupported literal"),
    ('Factor("close") + 1 / 0', "invalid constant"),
    ('Factor("close") + 10.0 ** 10 ** 10', "invalid constant"),
    ('Factor("close") = 1', "syntax"),
    ('import os', "syntax"),
    ('Factor("close") ' + "+ 1 " * 2000, "too long"),
    ("(" * 300 + 'Factor("close")' + ")" * 300, "syntax"),
    ('Factor("close")' + ".Abs()" * 150, "too deeply nested"),
]


@pytest.mark.parametrize("text", ACCEPTED)
def test_accepts(text):
    assert isinstance(parse(text), (Field, Op))


@pytest.mark.parametrize("text,message", REJECTED)
def test_rejects(text, message):
    with pytest.raises(DslError, match=message):
        parse(text)


def test_constants_fold():
    assert parse('-1 * Factor("close")') == Op("mul", (Const(-1.0), Field("close")))
    assert parse('(2 + 3) * Factor("close")') == Op("mul", (Const(5.0), Field("close")))


def test_keyword_order_does_not_change_the_tree():
    a = parse('Factor("close").ts_mean(min_periods=5, window=6)')
    b = parse('Factor("close").ts_mean(6, 5)')
    assert a == b
    assert parse('(Factor("close") > 1).where(false_value=Factor("low"), true_value=Factor("high"))') == parse(
        'where(Factor("close") > 1, Factor("high"), Factor("low"))'
    )


def test_aliases_share_one_operator():
    assert parse('Factor("close").delay(3)') == parse('Factor("close").lag(periods=3)')
    assert parse('Factor("close").EMA(n=7)') == parse('Factor("close").EMA(span=7)')
    assert parse('Factor("close").sec_rank()') == parse('Factor("close").rank(pct=True)')
    assert parse('Factor("close").percentile()').name == "rank"


def test_skeleton_ignores_numbers():
    a = dsl.skeleton(parse('Factor("volume").ts_mean(5) / Factor("volume").ts_mean(20)'))
    b = dsl.skeleton(parse('Factor("volume").ts_mean(window=10) / Factor("volume").ts_mean(window=60)'))
    assert a == b
    assert a != dsl.skeleton(parse('Factor("volume").ts_mean(5) / Factor("volume").ts_std(20)'))


def test_fields_of():
    assert dsl.fields_of(parse('Factor("close").ts_corr(Factor("volume"), 5) + 1')) == {"close", "volume"}


def test_a_number_too_large_for_a_float_is_a_parse_error():
    with pytest.raises(DslError, match="too large"):
        parse('Factor("close") * 1' + "0" * 400)


def test_min_periods_cannot_outlast_the_gates_history():
    """A formula blank through 2024 would pass the gates on its other branch and switch on when scored."""
    with pytest.raises(DslError, match="min_periods must be at most"):
        parse('Factor("close").EMA(span=120, min_periods=1101)')
    parse('Factor("close").EMA(span=120, min_periods=96)')
