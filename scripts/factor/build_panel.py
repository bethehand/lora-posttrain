#!/usr/bin/env python3
"""Turn the frozen daily-kline snapshot into the panels the sandbox reads.

Input:  <snapshot>/raw/_fetcher_v2/kline_<field>__freq-1d_instrument_type-perp/W/*.feather
        <snapshot>/raw/_kv/instruments_binance_perp_usdt_v4.feather
Output: <out>/dev.npz     rows labelled up to 2025-12-31 (everything training may see)
        <out>/sealed.npz  all rows up to the last date every field has
        <out>/audit.json  what the snapshot contains and whether it is consistent

A row's label is the close time of the bar, so the row labelled 2024-03-12 is
the bar that opened on 2024-03-11. dev.npz holds no 2026 row at all: code that
only loads it cannot read the sealed test period by accident.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import pyarrow.feather as pf

FIELDS = ("open", "high", "low", "close", "volume", "quote_volume", "num_trades", "taker_base_vol", "taker_quote_vol")
DEV_LAST = pd.Timestamp("2025-12-31")
TOP_N = 300
MIN_AGE_DAYS = 260
LIQ_WINDOW, LIQ_MIN = 30, 20


def load_field(snapshot: str, field: str) -> pd.DataFrame:
    pattern = f"{snapshot}/raw/_fetcher_v2/kline_{field}__freq-1d_instrument_type-perp/W/*.feather"
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files for {field}")
    df = pd.concat([pf.read_table(p).to_pandas() for p in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    if df.index.has_duplicates:
        raise SystemExit(f"duplicate timestamps in {field}")
    return df


def monthly_universe(dates: pd.DatetimeIndex, quote_volume: np.ndarray, onboard: np.ndarray) -> np.ndarray:
    """Top coins by trailing liquidity, chosen on the eve of each month and held for that month."""
    trailing = pd.DataFrame(quote_volume).rolling(LIQ_WINDOW, min_periods=LIQ_MIN).mean().to_numpy()
    out = np.zeros_like(quote_volume, dtype=bool)
    periods = dates.to_period("M")
    for month in periods.unique():
        rows = np.flatnonzero(periods == month)
        eve = rows[0] - 1
        if eve < 0:
            continue
        age_days = (dates[eve].to_datetime64() - onboard) / np.timedelta64(1, "D")
        liq = trailing[eve]
        eligible = np.flatnonzero(np.isfinite(liq) & (liq > 0) & (age_days >= MIN_AGE_DAYS))
        chosen = eligible[np.argsort(-liq[eligible], kind="stable")[:TOP_N]]
        out[np.ix_(rows, chosen)] = True
    return out


def audit(dates: pd.DatetimeIndex, a: dict[str, np.ndarray], universe: np.ndarray) -> dict:
    o, h, l, c = a["open"], a["high"], a["low"], a["close"]
    v, q, n, tb, tq = a["volume"], a["quote_volume"], a["num_trades"], a["taker_base_vol"], a["taker_quote_vol"]
    tol = 1e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        vwap = q / v
        checks = {
            "cells_with_data": int(np.isfinite(c).sum()),
            "null_pattern_differs_from_close": {f: int((np.isfinite(a[f]) != np.isfinite(c)).sum()) for f in FIELDS},
            "high_below_low": int(np.nansum(h < l * (1 - tol))),
            "close_above_high": int(np.nansum(c > h * (1 + tol))),
            "close_below_low": int(np.nansum(c < l * (1 - tol))),
            "open_above_high": int(np.nansum(o > h * (1 + tol))),
            "open_below_low": int(np.nansum(o < l * (1 - tol))),
            "vwap_outside_range_by_over_0.1pct": int(np.nansum((vwap > h * 1.001) | (vwap < l * 0.999))),
            "taker_base_above_volume": int(np.nansum(tb > v * (1 + tol))),
            "taker_quote_above_quote_volume": int(np.nansum(tq > q * (1 + tol))),
            "zero_volume_bars": int(np.nansum(v == 0)),
            "non_positive_prices": int(sum(np.nansum(x <= 0) for x in (o, h, l, c))),
            "abs_daily_return_over_3x": int(np.nansum(np.abs(np.log(c[1:] / c[:-1])) > np.log(3))),
        }
    years = dates.year
    checks["median_coins_with_data_by_year"] = {
        int(y): int(np.median(np.isfinite(c[years == y]).sum(axis=1))) for y in np.unique(years)
    }
    checks["median_universe_size_by_year"] = {
        int(y): int(np.median(universe[years == y].sum(axis=1))) for y in np.unique(years)
    }
    return checks


def save(path: str, dates: pd.DatetimeIndex, sec_ids, symbols, a: dict[str, np.ndarray], universe: np.ndarray) -> None:
    np.savez(
        path,
        dates=dates.to_numpy().astype("datetime64[D]"),
        sec_ids=np.array(list(sec_ids), dtype="U"),
        symbols=np.array(list(symbols), dtype="U"),
        universe=universe,
        **{f: a[f] for f in FIELDS},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="/mnt/nvme1/bruce/data/binance_daily_20261105")
    ap.add_argument("--out", default="/mnt/nvme1/bruce/data/factor_idea_v1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = {f: load_field(args.snapshot, f) for f in FIELDS}
    # close and quote_volume were refreshed later than the rest; only rows every field has count.
    first = max(df.index[0] for df in raw.values())
    last = min(df.index[-1] for df in raw.values())
    dates = pd.date_range(first, last, freq="D")
    columns = sorted(set.intersection(*(set(df.columns) for df in raw.values())), key=int)
    a = {f: raw[f].reindex(index=dates, columns=columns).to_numpy(dtype=np.float64) for f in FIELDS}

    instruments = pf.read_table(f"{args.snapshot}/raw/_kv/instruments_binance_perp_usdt_v4.feather").to_pandas()
    instruments["symbol"] = instruments["symbol"].astype(str)
    instruments = instruments.set_index("symbol")
    missing = [c for c in columns if c not in instruments.index]
    if missing:
        raise SystemExit(f"columns without instrument rows: {missing[:5]}")
    symbols = instruments.loc[columns, "native_symbol"].astype(str).to_numpy()
    onboard = pd.to_datetime(instruments.loc[columns, "onboard_ts"]).to_numpy()

    universe = monthly_universe(dates, a["quote_volume"], onboard)

    dev = dates <= DEV_LAST
    save(f"{args.out}/dev.npz", dates[dev], columns, symbols, {f: a[f][dev] for f in FIELDS}, universe[dev])
    save(f"{args.out}/sealed.npz", dates, columns, symbols, a, universe)

    report = {
        "snapshot": args.snapshot,
        "first_row": str(dates[0].date()),
        "last_row_all_fields": str(dates[-1].date()),
        "last_row_by_field": {f: str(raw[f].index[-1].date()) for f in FIELDS},
        "dev_last_row": str(dates[dev][-1].date()),
        "rows": {"dev": int(dev.sum()), "all": len(dates)},
        "coins": len(columns),
        "universe_rule": f"on the eve of each month: top {TOP_N} by {LIQ_WINDOW}-day mean quote volume among coins listed for at least {MIN_AGE_DAYS} days",
        "checks": audit(dates, a, universe),
    }
    with open(f"{args.out}/audit.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
