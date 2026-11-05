#!/usr/bin/env python3
"""Sit the verifier down for an exam: factors whose right verdict is known in advance.

The verifier can only be trusted as a reward if it passes honest factors and fails cheats.
Honest factors are written by hand, and some come from the research system. Cheats are
honest factors with the sign flipped, honest factors for a different key, unit-dependent
formulas, degenerate formulas, and formulas that barely touch the key. Uses rows up to
2024-12-31 only, through Sandbox.gate_panel.
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs", "factor_idea"))

from factor_idea import dsl, sandbox as sb  # noqa: E402
from factor_idea.evaluate import evaluate  # noqa: E402
from factor_idea.minutes import load_minutes  # noqa: E402
from factor_idea.panel import load_dev  # noqa: E402

C, O, H, L = 'Factor("close")', 'Factor("open")', 'Factor("high")', 'Factor("low")'
Q, V, N = 'Factor("quote_volume")', 'Factor("volume")', 'Factor("num_trades")'
TQ, TB = 'Factor("taker_quote_vol")', 'Factor("taker_base_vol")'
RET = f"({C}.safe_div({C}.lag(periods=1)) - 1)"

# Key -> honest formulas where more of the key data means a higher value.
UP = {
    "volume": [
        f"{Q}.ts_mean(window=5, min_periods=4)",
        f"{Q}.ts_mean(window=20, min_periods=16).ln()",
        f"{Q}.ln().sec_rank()",
        f"{Q}.safe_div({Q}.ts_mean(window=20, min_periods=16))",
        f"{Q}.safe_div({Q}.ts_mean(window=60, min_periods=48))",
        f"{Q}.ts_mean(window=5, min_periods=4).safe_div({Q}.ts_mean(window=20, min_periods=16))",
        f"{Q}.ts_mean(window=5, min_periods=4).safe_div({Q}.ts_mean(window=60, min_periods=48))",
        f"{Q}.ts_zscore(window=20, min_periods=16)",
        f"{Q}.ln().ts_zscore(window=60, min_periods=48)",
        f"{Q}.ts_rank(window=20, min_periods=16)",
        f"{Q}.EMA(span=5, min_periods=4).safe_div({Q}.EMA(span=20, min_periods=16))",
        f"{Q}.pct_change(periods=5)",
        f"{Q}.ln().diff(periods=5).ts_mean(window=3)",
        f"{V}.ts_mean(window=5, min_periods=4).safe_div({V}.ts_mean(window=60, min_periods=48))",
        f"{V}.safe_div({V}.ts_mean(window=20, min_periods=16)).EMA(span=5, min_periods=4)",
        f"{Q}.ts_sum(window=7, min_periods=6).ln().zscore()",
        f"{Q}.ts_mean(window=10, min_periods=8).ln().sec_rank().ts_mean(window=5, min_periods=4)",
    ],
    "trades": [
        f"{N}.ts_mean(window=5, min_periods=4)",
        f"{N}.ts_mean(window=20, min_periods=16).ln()",
        f"{N}.safe_div({N}.ts_mean(window=20, min_periods=16))",
        f"{N}.ts_mean(window=5, min_periods=4).safe_div({N}.ts_mean(window=60, min_periods=48))",
        f"{N}.ln().sec_rank()",
        f"{N}.ts_zscore(window=20, min_periods=16)",
    ],
    "trade_size": [
        f"{Q}.safe_div({N})",
        f"{Q}.safe_div({N}).ts_mean(window=10, min_periods=8)",
        f"{Q}.safe_div({N}).ln().sec_rank()",
        f"{Q}.safe_div({N}).safe_div({Q}.safe_div({N}).ts_mean(window=20, min_periods=16))",
        f"{Q}.ts_sum(window=5, min_periods=4).safe_div({N}.ts_sum(window=5, min_periods=4))",
    ],
    "return": [
        RET,
        f"{C}.safe_div({C}.lag(periods=3)) - 1",
        f"{C}.safe_div({C}.lag(periods=5)) - 1",
        f"{C}.safe_div({C}.lag(periods=20)) - 1",
        f"{C}.pct_change(periods=30)",
        f"{C}.safe_div({C}.lag(periods=10)).ln()",
        f"{C}.safe_div({C}.ts_mean(window=20, min_periods=16)) - 1",
        f"{C}.safe_div({C}.ts_max(window=60, min_periods=48))",
        f"{C}.safe_div({C}.EMA(span=20, min_periods=16)) - 1",
        f"{C}.EMA(span=5, min_periods=4).safe_div({C}.EMA(span=20, min_periods=16)) - 1",
        f"({C}.safe_div({O}) - 1).ts_mean(window=5, min_periods=4)",
        f"{RET}.ts_mean(window=10, min_periods=8)",
        f"{RET}.ts_sum(window=5, min_periods=4).sec_rank()",
        f"{RET}.EMA(span=5, min_periods=4)",
        f"{C}.ts_rank(window=20, min_periods=16)",
    ],
    "volatility": [
        f"{RET}.ts_std(window=5, min_periods=4)",
        f"{RET}.ts_std(window=20, min_periods=16)",
        f"{RET}.ts_std(window=60, min_periods=48)",
        f"{RET}.Abs().ts_mean(window=10, min_periods=8)",
        f"({RET} ** 2).ts_mean(window=20, min_periods=16)",
        f"({H}.safe_div({L}) - 1).ts_mean(window=10, min_periods=8)",
        f"{H}.safe_div({L}).ln().EMA(span=10, min_periods=8)",
        f"({H} - {L}).safe_div({C}).ts_mean(window=5, min_periods=4)",
        f"{RET}.Abs().ts_max(window=20, min_periods=16)",
        f"{H}.ts_max(window=20, min_periods=16).safe_div({L}.ts_min(window=20, min_periods=16)).ln()",
    ],
    "taker_buy": [
        f"{TQ}.safe_div({Q})",
        f"{TQ}.safe_div({Q}).ts_mean(window=5, min_periods=4)",
        f"{TB}.safe_div({V}).ts_mean(window=10, min_periods=8)",
        f"(2 * {TQ} - {Q}).safe_div({Q}).EMA(span=5, min_periods=4)",
        f"{TQ}.ts_sum(window=20, min_periods=16).safe_div({Q}.ts_sum(window=20, min_periods=16))",
        f"{TQ}.safe_div({Q}).ts_zscore(window=20, min_periods=16)",
        f"{TQ}.safe_div({Q} - {TQ}).ts_mean(window=5, min_periods=4)",
    ],
    "close_near_high": [
        f"({C} - {L}).safe_div({H} - {L})",
        f"({C} - {L}).safe_div({H} - {L}).ts_mean(window=5, min_periods=4)",
        f"{C}.safe_div({H})",
        f"{C}.safe_div({H}).ts_mean(window=10, min_periods=8)",
        f"-1 * ({H} - {C}).safe_div({C}).ts_mean(window=5, min_periods=4)",
        f"({C} - {L}).safe_div({C}).ts_mean(window=5, min_periods=4)",
        f"({C} - {L}).safe_div({H} - {L}).EMA(span=10, min_periods=8)",
    ],
}

# From the research system, computable from daily klines. (name, formula, key, direction)
RESEARCH_SYSTEM = [
    ("s10:ret_c2c", RET, "return", 1),
    ("s10:ret_abs_log_ret_per_quote_vol_mean6_raw", f"({RET}.Abs() + 1).ln().safe_div({Q}).ts_mean(min_periods=5, window=6)", "volume", -1),
    ("s10:neg_ret_disp_med15", f"-1 * ({RET}.demean() ** 2).ts_median(min_periods=12, window=15)", "volatility", -1),
    ("notebook:amihud_sqrt_mean6", f"((({RET}.Abs() + 1) ** 0.5 - 1) / {Q}).ts_mean(6)", "volume", -1),
    ("notebook:amihud_ln_abs_mean6", f"-1 * ({RET}.Abs().ln() / {Q}).ts_mean(6)", "volume", -1),
    ("notebook:amihud_mean10", f"(({RET}.Abs() + 1).ln() / {Q}).ts_mean(10)", "volume", -1),
    ("notebook:amihud_mean5", f"(({RET}.Abs() + 1).ln() / {Q}).ts_mean(5)", "volume", -1),
    ("notebook:amihud_1d", f"{RET}.Abs() / {Q}", "volume", -1),
    ("notebook:ret_7d", f"({C} - {C}.delay(7)) / {C}.delay(7)", "return", 1),
    ("notebook:ret_5d", f"{C} / {C}.delay(5) - 1", "return", 1),
]

UNIT_DEPENDENT = [
    (C, "return", 1),
    (f"{C}.ln()", "return", 1),
    (f"{C}.diff(periods=5)", "return", 1),
    (f"{C}.sec_rank()", "return", 1),
    (f"{C}.EMA(span=5, min_periods=4)", "return", 1),
    (V, "volume", 1),
    (f"{V}.ts_mean(window=5, min_periods=4)", "volume", 1),
    (f"{V}.ln().sec_rank()", "volume", 1),
    (f"{H} - {L}", "volatility", 1),
    (f"{C}.ts_std(window=20, min_periods=16)", "volatility", 1),
    (f"{H} - {C}", "close_near_high", -1),
    (f"{TB}", "taker_buy", 1),
    (f"{TB}.ts_mean(window=5, min_periods=4)", "taker_buy", 1),
]

DEGENERATE = [
    (f"{C} / {C}", "return", 1),
    (f"{Q} - {Q}", "volume", 1),
    (f"{C}.ts_std(window=1, min_periods=1)", "volatility", 1),
    (f"{C}.lag(365).lag(365).lag(365).lag(365).lag(365)", "return", 1),
    (f"{RET}.sign()", "return", 1),
    (f"({C} > {O}) * 1", "return", 1),
    (f"{Q}.Abs() * 0", "volume", 1),
]


def dosed(key: str, other: str, dose: float) -> str:
    """Mostly some other factor, with a dose of the key data."""
    good, unrelated = UP[key][1], UP[other][1]
    return f"({unrelated}).sec_rank() + {dose} * ({good}).sec_rank()"


OMNIBUS = " + ".join(f"({UP[k][1]}).sec_rank()" for k in UP)

# Decided before looking at the verdicts: right answers a careful reader would give.
HARD = [
    ("range posing as return", f"{H}.safe_div({L})", "return", 1, False),
    ("dollars of taker buying, not a share", f"{TQ}.ts_mean(window=5)", "taker_buy", 1, False),
    ("taker dollars over coin volume, unit-dependent", f"{TQ}.safe_div({V})", "taker_buy", 1, False),
    ("key regressed away", f"{RET}.neutralize({Q}.ln())", "volume", 1, False),
    ("coin units from dollars, unit-dependent", f"{Q}.safe_div({C})", "volume", 1, False),
    ("volume volatility for a price-volatility idea", f"{Q}.ts_std(window=20, min_periods=16)", "volatility", 1, False),
    ("trade count for a trade-size idea", f"{N}.ts_mean(window=5)", "trade_size", -1, False),
    ("open-close body for a close-position idea", f"{C}.safe_div({O})", "close_near_high", 1, False),
    ("dollar volume for a trade-size idea", f"{Q}.EMA(span=14, min_periods=1)", "trade_size", 1, False),
    ("rank to a power", f"({Q}.ln().sec_rank()) ** 3", "volume", 1, True),
    ("clipped surge ratio", f"{Q}.safe_div({Q}.ts_mean(window=20, min_periods=16)).clip(upper=2)", "volume", 1, True),
    ("surge after removing liquidity", f"{Q}.ln().neutralize({Q}.ts_mean(window=30, min_periods=20).ln())", "volume", 1, True),
    ("six-month return", f"{C}.safe_div({C}.lag(180)) - 1", "return", 1, True),
    ("skip-a-week momentum", f"{C}.lag(10).safe_div({C}.lag(30)) - 1", "return", 1, True),
]

# Not cheats and not honest: reported so the gate's blind spots are on the record.
GREY = [
    ("absolute surge, also high when volume collapses", f"{Q}.ts_zscore(window=20, min_periods=16).Abs()", "volume", 1),
    ("volume a month ago", f"{Q}.lag(30)", "volume", 1),
    ("volume volatility for a volume idea", f"{Q}.ts_std(window=20, min_periods=16)", "volume", 1),
    ("one-year average volume", f"{Q}.ts_mean(window=365, min_periods=200).ln()", "volume", 1),
    ("day of the highest volume", f"{Q}.ts_argmax(window=20, min_periods=16)", "volume", 1),
    ("volume only on up days", f"where({RET} > 0, {Q}.ln(), 0)", "volume", 1),
    ("price up on high volume", f"{RET} * {Q}.safe_div({Q}.ts_mean(window=20, min_periods=16))", "volume", 1),
    ("price-volume correlation", f"{C}.ts_corr({V}, 30)", "volume", 1),
]


# The same exam for formulas that look inside the day. Key -> honest formulas where more of the key
# means a higher value.
MQ, MV, MN = "Minute('quote_volume')", "Minute('volume')", "Minute('num_trades')"
MR, MC, MO, MH, ML = "Minute('return')", "Minute('close')", "Minute('open')", "Minute('high')", "Minute('low')"
MTQ, MM = "Minute('taker_quote_vol')", "Minute('market_return')"
MDOWN = f"{MR}.clip(upper=0).Abs().day_center()"
MUP = f"{MR}.clip(lower=0).day_center()"
MSYNC = f"{MR}.day_corr({MM})"
MINUTE_UP = {
    "volume_late": [
        f"{MQ}.day_center()",
        f"{MQ}.day_center().ts_mean(window=10, min_periods=8)",
        f"{MQ}.day_center().EMA(span=5, min_periods=4)",
        f"{MQ}.day_center().sec_rank()",
        f"{MV}.day_center()",
        f"{MN}.day_center().ts_mean(window=5, min_periods=4)",
        f"{MQ}.between(18, 24).day_sum().safe_div({MQ}.day_sum())",
        f"{MQ}.between(18, 24).day_sum().safe_div({MQ}.day_sum()).ts_mean(window=10, min_periods=8)",
        f"{MQ}.between(12, 24).day_sum().safe_div({MQ}.day_sum())",
        f"{MQ}.between(18, 24).day_sum().safe_div({MQ}.between(0, 6).day_sum())",
        f"-1 * {MQ}.between(0, 6).day_sum().safe_div({MQ}.day_sum())",
        f"-1 * {MQ}.between(3, 11).day_sum().safe_div({MQ}.day_sum()).ts_mean(window=5, min_periods=4)",
    ],
    "volume_spike": [
        f"{MQ}.day_max().safe_div({MQ}.day_sum())",
        f"{MQ}.day_max().safe_div({MQ}.day_sum()).ts_mean(window=10, min_periods=8)",
        f"{MQ}.day_max().safe_div({MQ}.day_sum()).EMA(span=5, min_periods=4)",
        f"{MQ}.day_max().safe_div({MQ}.day_mean()).ln()",
        f"{MQ}.day_std().safe_div({MQ}.day_mean())",
        f"{MQ}.day_std().safe_div({MQ}.day_mean()).ts_mean(window=5, min_periods=4)",
        f"{MV}.day_max().safe_div({MV}.day_sum())",
        f"({MQ} ** 2).day_sum().safe_div({MQ}.day_sum() ** 2)",
        f"{MN}.day_max().safe_div({MN}.day_sum()).ts_mean(window=5, min_periods=4)",
    ],
    "volume_on_up": [
        f"where({MR} > 0, {MQ}, 0).day_sum().safe_div({MQ}.day_sum())",
        f"where({MR} > 0, {MQ}, 0).day_sum().safe_div({MQ}.day_sum()).ts_mean(window=10, min_periods=8)",
        f"where({MR} > 0, {MV}, 0).day_sum().safe_div(where({MR} != 0, {MV}, 0).day_sum())",
        f"where({MR} > 0, {MV}, 0).day_sum().safe_div(where({MR} != 0, {MV}, 0).day_sum()).ts_mean(window=10, min_periods=8).EMA(span=10, min_periods=8)",
        f"where({MR} > 0, {MQ}, 0).day_sum().safe_div(where({MR} < 0, {MQ}, 0).day_sum())",
        f"-1 * where({MR} < 0, {MQ}, 0).day_sum().safe_div({MQ}.day_sum())",
        f"{MQ}.day_corr({MR})",
    ],
    "down_late": [
        MDOWN,
        f"{MDOWN}.ts_mean(window=10, min_periods=8)",
        f"{MDOWN}.EMA(span=5, min_periods=4)",
        f"{MDOWN}.sec_rank()",
        f"{MR}.clip(upper=0).Abs().between(12, 24).day_sum().safe_div({MR}.clip(upper=0).Abs().day_sum())",
        f"where({MR} < 0, {MR}.Abs(), 0).day_center()",
        f"-1 * {MR}.clip(upper=0).Abs().between(0, 12).day_sum().safe_div({MR}.clip(upper=0).Abs().day_sum())",
    ],
    "up_late": [
        MUP,
        f"{MUP}.ts_mean(window=10, min_periods=8)",
        f"{MUP}.EMA(span=5, min_periods=4)",
        f"{MUP}.sec_rank()",
        f"{MR}.clip(lower=0).between(12, 24).day_sum().safe_div({MR}.clip(lower=0).day_sum())",
        f"where({MR} > 0, {MR}, 0).day_center()",
        f"-1 * {MR}.clip(lower=0).between(0, 12).day_sum().safe_div({MR}.clip(lower=0).day_sum())",
    ],
    "market_sync": [
        MSYNC,
        f"{MSYNC}.ts_mean(window=10, min_periods=8)",
        f"{MSYNC}.EMA(span=5, min_periods=4)",
        f"{MSYNC}.sec_rank()",
        f"({MR} + 1).ln().day_corr(({MM} + 1).ln())",
        f"{MR}.between(0, 12).day_corr({MM})",
        f"({MR} * {MM}).day_mean().safe_div({MR}.day_std())",
    ],
}

# The older keys, answered from minutes instead of from the daily bar.
MINUTE_OLD_KEYS = {
    "volume": [f"{MQ}.day_sum().ts_mean(window=5, min_periods=4)", f"{MQ}.day_sum().ln().sec_rank()", f"{MQ}.day_sum().safe_div({MQ}.day_sum().ts_mean(window=20, min_periods=16))", f"{MQ}.day_mean().ts_mean(window=20, min_periods=16).ln()"],
    "trades": [f"{MN}.day_sum().ts_mean(window=5, min_periods=4)", f"{MN}.day_sum().safe_div({MN}.day_sum().ts_mean(window=20, min_periods=16))"],
    "trade_size": [f"{MQ}.day_sum().safe_div({MN}.day_sum())", f"{MQ}.day_sum().safe_div({MN}.day_sum()).ts_mean(window=10, min_periods=8)"],
    "return": [f"{MC}.day_last().safe_div({MO}.day_first()) - 1", f"{MR}.day_sum()", f"{MR}.day_sum().ts_sum(window=5, min_periods=4)", f"({MR} + 1).ln().day_sum().ts_mean(window=10, min_periods=8)"],
    "volatility": [f"{MR}.day_std()", f"{MR}.day_std().ts_mean(window=10, min_periods=8)", f"({MR} ** 2).day_sum().ts_mean(window=20, min_periods=16)", f"{MR}.Abs().day_sum().ts_mean(window=5, min_periods=4)", f"{MH}.day_max().safe_div({ML}.day_min()).ln().ts_mean(window=10, min_periods=8)"],
    "taker_buy": [f"{MTQ}.day_sum().safe_div({MQ}.day_sum())", f"{MTQ}.day_sum().safe_div({MQ}.day_sum()).ts_mean(window=5, min_periods=4)", f"(2 * {MTQ} - {MQ}).day_sum().safe_div({MQ}.day_sum())", f"{MTQ}.safe_div({MQ}).day_mean().ts_mean(window=5, min_periods=4)"],
    "close_near_high": [f"{MC}.day_last().safe_div({MH}.day_max())", f"({MC}.day_last() - {ML}.day_min()).safe_div({MH}.day_max() - {ML}.day_min()).ts_mean(window=5, min_periods=4)"],
}

# From the research system: the Bases its live components read, written in this language.
MINUTE_RESEARCH_SYSTEM = [
    ("s10:dir_volume_share", f"where({MR} > 0, {MV}, 0).day_sum().safe_div(where({MR} != 0, {MV}, 0).day_sum())", "volume_on_up", 1),
    ("s10:dir_vol_share_pow2p5_mean10_ema10", f"(where({MR} > 0, {MV}, 0).day_sum().safe_div(where({MR} != 0, {MV}, 0).day_sum()) ** 2.5).ts_mean(min_periods=8, window=10).EMA(min_periods=8, span=10)", "volume_on_up", 1),
    ("base:quote_vol_concentration", f"{MQ}.day_max().safe_div({MQ}.day_sum())", "volume_spike", 1),
    ("base:quote_vol_time_center_1d", f"{MQ}.day_center()", "volume_late", 1),
    ("base:us_quote_vol_share_1d", f"{MQ}.between(18, 24).day_sum().safe_div({MQ}.day_sum())", "volume_late", 1),
    ("base:down_ret_time_center", MDOWN, "down_late", 1),
    ("base:up_ret_time_center", MUP, "up_late", 1),
    ("base:ret_mkt_cor", MSYNC, "market_sync", 1),
]

MINUTE_UNIT_DEPENDENT = [
    (f"{MC}.day_last()", "return", 1),
    (f"{MC}.day_last().ln()", "return", 1),
    (f"{MV}.day_sum()", "volume", 1),
    (f"{MV}.day_sum().ts_mean(window=5, min_periods=4)", "volume", 1),
    (f"{MH}.day_max() - {ML}.day_min()", "volatility", 1),
    (f"{MV}.day_max()", "volume_spike", 1),
    (f"{MC}.day_center()", "down_late", 1),
    (f"{MV}.between(18, 24).day_sum()", "volume_late", 1),
]

MINUTE_DEGENERATE = [
    (f"({MQ} / {MQ}).day_sum()", "volume_late", 1),
    (f"{MQ}.day_sum() - {MQ}.day_sum()", "volume", 1),
    (f"{MQ}.day_center() * 0", "volume_late", 1),
    (f"({MR} > 0).sign().day_max()", "up_late", 1),
    (f"{MM}.day_sum()", "market_sync", 1),
    (f"{MM}.day_center()", "market_sync", 1),
]

# Decided before looking at the verdicts: right answers a careful reader would give.
MINUTE_HARD = [
    ("late dollars, not a share of the day's", f"{MQ}.between(18, 24).day_sum()", "volume_late", 1, False),
    ("late dollars smoothed, not a share", f"{MQ}.between(18, 24).day_sum().ts_mean(window=10, min_periods=8)", "volume_late", 1, False),
    ("largest minute in dollars, not a share", f"{MQ}.day_max().ts_mean(window=5, min_periods=4)", "volume_spike", 1, False),
    ("the day's total for a question about its timing", f"{MQ}.day_sum().ln()", "volume_late", 1, False),
    ("the day's return for a question about when it fell", f"{MR}.day_sum()", "down_late", 1, False),
    ("worst minute for a question about when it fell", f"{MR}.day_min()", "down_late", 1, False),
    ("volatility for a question about when it fell", f"{MR}.day_std()", "down_late", 1, False),
    ("up-minute timing for a question about falls", MUP, "down_late", 1, False),
    ("down-minute timing for a question about gains", MDOWN, "up_late", 1, False),
    ("volume timing for a question about returns", f"{MQ}.day_center()", "down_late", 1, False),
    ("return timing for a question about volume", MDOWN, "volume_late", 1, False),
    ("volatility for a question about following the market", f"{MR}.day_std()", "market_sync", 1, False),
    ("volume following the market, for returns following the market", f"{MQ}.day_corr(Minute('market_volume_share'))", "market_sync", 1, False),
    ("share of up minutes, not of volume in them", f"({MR} > 0).day_mean()", "volume_on_up", 1, False),
    ("up-minute dollars, not a share", f"where({MR} > 0, {MQ}, 0).day_sum()", "volume_on_up", 1, False),
    ("taker share for a question about up minutes", f"{MTQ}.day_sum().safe_div({MQ}.day_sum())", "volume_on_up", 1, False),
    ("rank of the latest minutes' share", f"{MQ}.between(23, 24).day_sum().safe_div({MQ}.day_sum()).sec_rank()", "volume_late", 1, True),
    ("the share of the two busiest hours' dollars in the day's", f"{MQ}.between(23, 24).day_sum().safe_div({MQ}.day_sum()).EMA(span=5, min_periods=4)", "volume_late", 1, True),
]

MINUTE_GREY = [
    # Labelled honest at first. One huge minute inflates the day's spread, so fewer minutes clear three
    # sigma: this counts how many minutes stand out, which isn't what a single spike is.
    ("how many minutes stand out above three sigma", f"({MQ}.zscore_in_day() > 3).day_sum()", "volume_spike", 1),
    ("how far the close ends below the day's high", f"-1 * {MC}.day_last().safe_div({MH}.day_max())", "down_late", 1),
    ("last-hour return for a question about when it fell", f"-1 * {MR}.between(23, 24).day_sum()", "down_late", 1),
    ("afternoon share of trades for a question about dollars", f"{MN}.between(12, 24).day_sum().safe_div({MN}.day_sum())", "volume_late", 1),
    ("how many minutes stand out above one sigma", f"({MQ}.zscore_in_day() > 1).day_sum()", "volume_spike", 1),
    ("the day's peak time, by value", f"where({MQ}.zscore_in_day() > 3, 1, 0).day_center()", "volume_late", 1),
    ("taker imbalance timing", f"(2 * {MTQ} - {MQ}).day_center()", "volume_late", 1),
    ("return variance explained by the market", f"({MR}.day_corr({MM}) ** 2)", "market_sync", 1),
    ("beta to the market", f"({MR} * {MM}).day_sum().safe_div(({MM} ** 2).day_sum())", "market_sync", 1),
]

MINUTE_OMNIBUS = " + ".join(f"({MINUTE_UP[k][1]}).sec_rank()" for k in MINUTE_UP)


def minute_dosed(key: str, other: str, dose: float, table: dict) -> str:
    good, unrelated = MINUTE_UP[key][1], table[other][1]
    return f"({unrelated}).sec_rank() + {dose} * ({good}).sec_rank()"


def build_minute_exam() -> list[dict]:
    exam = []
    for key, formulas in MINUTE_UP.items():
        for text in formulas:
            exam.append(dict(group="minute_honest", text=text, key=key, direction=1, want=True))
            exam.append(dict(group="minute_honest_bearish", text=f"-1 * ({text})", key=key, direction=-1, want=True))
            exam.append(dict(group="minute_flipped_sign", text=text, key=key, direction=-1, want=False))
            exam.append(dict(group="minute_flipped_sign", text=f"-1 * ({text})", key=key, direction=1, want=False))
    for key, formulas in MINUTE_OLD_KEYS.items():
        for text in formulas:
            exam.append(dict(group="minute_old_key_honest", text=text, key=key, direction=1, want=True))
            exam.append(dict(group="minute_flipped_sign", text=text, key=key, direction=-1, want=False))
    for name, text, key, direction in MINUTE_RESEARCH_SYSTEM:
        exam.append(dict(group="minute_research_system", name=name, text=text, key=key, direction=direction, want=True))
        exam.append(dict(group="minute_flipped_sign", name=name, text=text, key=key, direction=-direction, want=False))
    # Another inside-the-day idea's formula, or a formula from the daily bars, doesn't answer the question.
    for key, other in itertools.permutations(MINUTE_UP, 2):
        for text in MINUTE_UP[other][:4]:
            exam.append(dict(group="minute_other_inside_key", text=text, key=key, direction=1, want=False))
        exam.append(dict(group="minute_token_dose", text=minute_dosed(key, other, 0.02, MINUTE_UP), key=key, direction=1, want=False))
        for dose in (0.1, 0.3):
            exam.append(dict(group=f"minute_mostly_other_dose_{dose}", text=minute_dosed(key, other, dose, MINUTE_UP), key=key, direction=1, want=False))
        exam.append(dict(group="grey_minute_equal_blend", text=minute_dosed(key, other, 1.0, MINUTE_UP), key=key, direction=1, want=None))
    for key in MINUTE_UP:
        for formulas in UP.values():
            for text in formulas[:3]:
                exam.append(dict(group="minute_daily_formula", text=text, key=key, direction=1, want=False))
        for other, formulas in MINUTE_OLD_KEYS.items():
            exam.append(dict(group="minute_old_key_formula", text=formulas[0], key=key, direction=1, want=False))
        exam.append(dict(group="minute_token_of_daily", text=f"({UP['volume'][1]}).sec_rank() + 0.02 * ({MINUTE_UP[key][1]}).sec_rank()", key=key, direction=1, want=False))
        exam.append(dict(group="minute_omnibus", text=MINUTE_OMNIBUS, key=key, direction=1, want=False))
    for name, text, key, direction, want in MINUTE_HARD:
        exam.append(dict(group="minute_hard_honest" if want else "minute_hard_cheat", name=name, text=text, key=key, direction=direction, want=want))
    for name, text, key, direction in MINUTE_GREY:
        exam.append(dict(group="grey_minute", name=name, text=text, key=key, direction=direction, want=None))
    for text, key, direction in MINUTE_UNIT_DEPENDENT:
        exam.append(dict(group="minute_unit_dependent", text=text, key=key, direction=direction, want=False))
    for text, key, direction in MINUTE_DEGENERATE:
        exam.append(dict(group="minute_degenerate", text=text, key=key, direction=direction, want=False))
    return exam


def build_exam() -> list[dict]:
    exam = []
    for key, formulas in UP.items():
        for text in formulas:
            exam.append(dict(group="honest", text=text, key=key, direction=1, want=True))
            exam.append(dict(group="honest_bearish", text=f"-1 * ({text})", key=key, direction=-1, want=True))
            exam.append(dict(group="flipped_sign", text=text, key=key, direction=-1, want=False))
            exam.append(dict(group="flipped_sign", text=f"-1 * ({text})", key=key, direction=1, want=False))
    for name, text, key, direction in RESEARCH_SYSTEM:
        also = ["price"] if "amihud" in name or "ret_abs_log_ret" in name else []
        exam.append(dict(group="research_system", name=name, text=text, key=key, direction=direction, also=also, want=True))
        exam.append(dict(group="flipped_sign", name=name, text=text, key=key, direction=-direction, also=also, want=False))
    # Keys whose perturbation legitimately moves the other's formulas are not cheats, e.g. a
    # price jump moves the range, and more volume moves trade size.
    related = {
        ("volume", "trades"), ("volume", "trade_size"), ("trades", "volume"), ("trades", "trade_size"),
        ("trade_size", "volume"), ("return", "volatility"), ("volatility", "return"),
        ("volatility", "close_near_high"), ("close_near_high", "volatility"),
        ("close_near_high", "return"), ("return", "close_near_high"),
    }
    for key, other in itertools.permutations(UP, 2):
        if (key, other) in related:
            continue
        for text in UP[other][:6]:
            exam.append(dict(group="ignores_the_key", text=text, key=key, direction=1, want=False))
        exam.append(dict(group="token_dose_of_the_key", text=dosed(key, other, 0.02), key=key, direction=1, want=False))
        if sb.KEY_GROUPS[key] != sb.KEY_GROUPS[other]:
            for dose in (0.1, 0.3, 0.5):
                exam.append(dict(group=f"mostly_other_data_dose_{dose}", text=dosed(key, other, dose), key=key, direction=1, want=False))
            exam.append(dict(group="grey_equal_blend", text=dosed(key, other, 1.0), key=key, direction=1, want=None))
    for key in UP:
        exam.append(dict(group="omnibus", text=OMNIBUS, key=key, direction=1, want=False))
    for name, text, key, direction, want in HARD:
        exam.append(dict(group="hard_honest" if want else "hard_cheat", name=name, text=text, key=key, direction=direction, want=want))
    for name, text, key, direction in GREY:
        exam.append(dict(group="grey", name=name, text=text, key=key, direction=direction, want=None))
    for text, key, direction in UNIT_DEPENDENT:
        exam.append(dict(group="unit_dependent", text=text, key=key, direction=direction, want=False))
    for text, key, direction in DEGENERATE:
        exam.append(dict(group="degenerate", text=text, key=key, direction=direction, want=False))
    return exam


_box = None


def _start_worker(minutes: bool) -> None:
    global _box
    _box = sb.Sandbox(load_dev(), minutes=load_minutes() if minutes else None)


def sit(item: dict) -> dict:
    t0 = time.monotonic()
    node = dsl.parse(item["text"])
    memo: dict = {}
    with np.errstate(all="ignore"):
        base = evaluate(node, _box.gate_panel.fields, memo=memo, minutes=_box.minutes)
        compute = _box.check_compute(base)
        idea = _box.check_idea(node, item["key"], item["direction"], base, also=tuple(item.get("also", ())), memo=memo) if compute.ok else None
    return dict(
        **item,
        passed=bool(compute.ok and idea.ok),
        reason=compute.reason if not compute.ok else idea.reason,
        good_day_share=round(compute.good_day_share, 3),
        windows=idea.windows if idea else {},
        unit_median=idea.unit_median if idea else None,
        unrelated=idea.unrelated if idea else {},
        seconds=round(time.monotonic() - t0, 3),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/factor/verifier_exam.json")
    ap.add_argument("--minutes", action="store_true", help="the exam for formulas that look inside the day")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    exam = build_minute_exam() if args.minutes else build_exam()
    started = time.monotonic()
    if args.jobs > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(args.jobs, mp_context=context, initializer=_start_worker, initargs=(args.minutes,)) as pool:
            rows = list(pool.map(sit, exam, chunksize=4))
    else:
        _start_worker(args.minutes)
        rows = [sit(item) for item in exam]
    print(f"{len(rows)} items in {time.monotonic() - started:.0f}s; median {np.median([r['seconds'] for r in rows]):.2f}s, max {max(r['seconds'] for r in rows):.2f}s per item")
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    summary = {}
    for name, items in groups.items():
        want = items[0]["want"]
        passed = sum(r["passed"] for r in items)
        summary[name] = dict(total=len(items), passed=passed, want_pass=want)
        print(f"  {name:32s} want_pass={want!s:5s} passed {passed}/{len(items)}")
    wrong_honest = sum(not r["passed"] for r in rows if r["want"] is True)
    total_honest = sum(r["want"] is True for r in rows)
    wrong_cheat = sum(r["passed"] for r in rows if r["want"] is False)
    total_cheat = sum(r["want"] is False for r in rows)
    print(f"false fail {wrong_honest}/{total_honest} = {wrong_honest / total_honest:.1%}; false pass {wrong_cheat}/{total_cheat} = {wrong_cheat / total_cheat:.1%}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, false_fail=wrong_honest / total_honest, false_pass=wrong_cheat / total_cheat, rows=rows), f, indent=1)
    for r in rows:
        if (r["want"] is None and r["group"] not in ("grey_equal_blend", "grey_minute_equal_blend")) or (r["want"] is not None and r["passed"] != r["want"]):
            best = max(r["windows"].values(), key=lambda v: abs(v["median"])) if r["windows"] else {}
            tag = "GREY" if r["want"] is None else "WRONG"
            print(f"  {tag} [{r['group']}] {r.get('name', '')} key={r['key']} d={r['direction']} passed={r['passed']} reason={r['reason']!r} unit={r['unit_median']} unrelated={ {k: round(v, 3) for k, v in r['unrelated'].items()} } best={ {k: round(v, 3) for k, v in best.items()} }\n    {r['text'][:170]}")


if __name__ == "__main__":
    main()
