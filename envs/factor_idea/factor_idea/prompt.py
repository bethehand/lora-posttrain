"""What the model is told, and how its answer is read back."""
from __future__ import annotations

import re

SYSTEM_PROMPT = """\
You write cross-sectional factors for Binance USDT perpetual futures, using daily bars.

A factor is one expression that gives every coin a number each day. A higher number means the coin is expected to do better over the following days. You will be given a one-sentence idea; turn it into a factor.

Data, one value per coin per day:
- Factor("open"), Factor("high"), Factor("low"), Factor("close"): prices
- Factor("volume"): amount traded, in units of the coin
- Factor("quote_volume"): amount traded, in USDT
- Factor("num_trades"): number of trades
- Factor("taker_base_vol"), Factor("taker_quote_vol"): the part of volume and of quote_volume that came from aggressive buyers

Operators:
- Arithmetic and comparisons: + - * / ** > >= < <= == != (a comparison gives 1 or 0)
- Each value on its own: .Abs() .ln() .sign() .pow(x) .clip(lower=a, upper=b) .max(other) .min(other) .fillna(other) .safe_div(other), and where(condition, a, b)
- One coin over its past n days, n a positive integer: .ts_mean(window=n, min_periods=m) .ts_std .ts_var .ts_sum .ts_max .ts_min .ts_median .ts_skew .ts_kurt .ts_rank (the latest value's rank in its window, from 1 to n) .ts_zscore .ts_argmax .ts_argmin .ts_decay_linear .ts_product (all take window and min_periods), .ts_quantile(window=n, q=x), .ts_corr(other, window=n), .ts_cov(other, window=n)
- Exponential averages: .EMA(span=n, min_periods=m) .ts_ewm_mean(span=n, min_periods=m)
- Earlier values: .lag(periods=n) .diff(periods=n) .pct_change(periods=n)
- All coins on one day: .sec_rank() .zscore() .demean() .sec_pct() .winsorize(limit=x) .scale() .neutralize(other)

Example: Factor("close").safe_div(Factor("close").lag(periods=5)) - 1 is each coin's return over the past 5 days.

Every operator only reads the past. Windows and periods must be positive, at most 365.

The factor must be driven by the data the idea talks about and must point the way the idea says: if the idea says something is bearish, the factor must go down when that thing goes up. Values must be comparable across coins, so a coin's price unit must not matter.

Finish your reply with the factor expression between <factor> and </factor>."""

# The text between a closing tag and the nearest opening tag before it. A model that restates
# itself writes "<factor>a ... <factor>a</factor>", and its answer is the second a.
_TOOL_SECTION = """\
Before you answer you can try candidates with the test_factor tool. Give it one expression. It tells you whether the expression computes and, on data from July 2022 to December 2024, its rank IC with the next day's return and its incremental IC, which is what remains after removing liquidity, volatility, short-term reversal and momentum. Higher is better, and an incremental IC of 0.01 to 0.03 is already good.

You may call test_factor at most 5 times, one call per turn. Testing many variants on the same history overfits it, so stop once you have something sound. test_factor does not check that the factor matches the idea; that is up to you. Keep the direction the idea says even if the IC comes out negative: the idea may simply be wrong on this history.

Write Factor('close') with single quotes inside the tool call. When you are done, reply without calling a tool and finish with the factor between <factor> and </factor>. A reply with no factor scores 0."""

_MINUTE_SECTION = """\
A factor can also look inside a coin's day, at its 1,440 one-minute bars, in UTC. Minute("close") is like Factor("close") with a value for every minute. All the fields above are available, plus Minute("return") (each minute's price change), Minute("market_return") (the median coin's return that minute) and Minute("market_volume_share") (the average coin's share of its day's quote_volume traded that minute).
- Minute by minute: arithmetic and comparisons, .Abs() .ln() .sign() .pow(x) .clip(lower=a, upper=b) .max(other) .min(other) .fillna(other) .safe_div(other), where(condition, a, b), .zscore_in_day(), and .between(start_hour, end_hour), which keeps only that part of the day
- Turning a day of minutes into one number: .day_sum() .day_mean() .day_std() .day_max() .day_min() .day_first() .day_last() .day_corr(other), and .day_center(), which says when in the day a positive amount falls, from 0 at the start to 1 at the end, ignoring negative values

A minute value can't be mixed with a daily value or use any other operator until a day_ operator has turned it into one number per day. After that it is a daily value like any other.

Example: Minute("quote_volume").between(18, 24).day_sum().safe_div(Minute("quote_volume").day_sum()).ts_mean(window=10, min_periods=8) is the share of a coin's dollar volume traded in the last six hours of the UTC day, averaged over 10 days."""

MINUTE_SYSTEM_PROMPT = SYSTEM_PROMPT.replace("using daily bars.", "using daily and one-minute bars.").replace(
    "Every operator only reads the past.", f"{_MINUTE_SECTION}\n\nEvery operator only reads the past."
)


def _with_tool(prompt: str) -> str:
    return (
        prompt.replace('Factor("', "Factor('")
        .replace('Minute("', "Minute('")
        .replace('")', "')")
        .replace("Finish your reply with the factor expression between <factor> and </factor>.", _TOOL_SECTION)
    )


TOOL_SYSTEM_PROMPT = _with_tool(SYSTEM_PROMPT)
MINUTE_TOOL_SYSTEM_PROMPT = _with_tool(MINUTE_SYSTEM_PROMPT)

_TAG = re.compile(r"<factor>((?:(?!<factor>).)*?)</factor>", re.DOTALL | re.IGNORECASE)


def extract_factor(reply: str) -> str | None:
    """The last tagged expression in a reply, without code fences; None if there isn't one."""
    found = _TAG.findall(reply or "")
    if not found:
        return None
    text = found[-1].strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    return text.strip("`").strip() or None
