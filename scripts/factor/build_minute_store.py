#!/usr/bin/env python3
"""Freeze the one-minute kline tables as the bars the sandbox reads.

Input:  <source>/kline_<field>__freq-1m_instrument_type-perp/W/*.feather, one wide table per
        ISO week, each row's timestamp being its bar's close time
Output: <out>/<field>.f32   coin-days x 1440 float32, for coin-days with any bar
        <out>/market_*.f32  panel rows x 1440 float32
        <out>/index.npz     which coin-day each line of those files is
        <out>/audit.json    what went in, whether it agrees with the daily panel, and checksums

A panel row labelled D is the day bars closing after D-1 00:00 and up to D 00:00, the same bars
its daily kline covers. Only the dev panel's rows are kept, so these files hold no 2026 bar.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import sys

import numpy as np
import polars as pl

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "envs", "factor_idea"))

from factor_idea.dsl import FIELDS, MARKET_SERIES  # noqa: E402
from factor_idea.minutes import MIN_MINUTES_SHARE, MinuteBars, market_series  # noqa: E402
from factor_idea.panel import load_dev  # noqa: E402

MINUTES = 1440


def weeks(source: str, field: str, first: np.datetime64, last: np.datetime64) -> list[str]:
    out = []
    for path in sorted(glob.glob(f"{source}/kline_{field}__freq-1m_instrument_type-perp/W/*.feather")):
        year, week = os.path.basename(path)[:7].split("W")
        monday = np.datetime64(dt.date.fromisocalendar(int(year), int(week), 1))
        if monday <= last and monday + np.timedelta64(7, "D") >= first:
            out.append(path)
    if not out:
        raise SystemExit(f"no files for {field}")
    return out


def read_days(path: str, dates: np.ndarray, column_of: dict[str, int]):
    """Yield (panel row, position in the day of each kept bar, panel columns, values as bars x columns)."""
    df = pl.read_ipc(path, memory_map=False)
    stamps = df["timestamp"].cast(pl.Datetime("us")).to_numpy().astype("datetime64[m]")
    names = [c for c in df.columns if c in column_of]
    columns = np.array([column_of[c] for c in names])
    values = df.select(names).to_numpy().astype(np.float32)
    labels = (stamps - np.timedelta64(1, "m")).astype("datetime64[D]") + np.timedelta64(1, "D")
    for label in np.unique(labels):
        r = int(np.searchsorted(dates, label))
        if r >= dates.size or dates[r] != label:
            continue
        keep = labels == label
        opened = (label - np.timedelta64(1, "D")).astype("datetime64[m]")
        position = ((stamps[keep] - opened) / np.timedelta64(1, "m")).astype(np.int64) - 1
        yield r, position, columns, values[keep]


def agree_with_daily(bars: MinuteBars, dev) -> dict:
    """Bars added up per coin-day against the daily panel, on coin-days in the universe."""
    out = {}
    mask = dev.universe[bars.rows, bars.coins] & bars.usable
    index = np.flatnonzero(mask)
    for name in FIELDS:
        daily = dev.fields[name][bars.rows[index], bars.coins[index]]
        got = np.empty(index.size)
        for start in range(0, index.size, 4096):
            part = index[start : start + 4096]
            x = np.asarray(bars.fields[name][part]).astype(np.float64)
            ok = np.isfinite(x)
            if name == "open":
                got[start : start + 4096] = x[np.arange(x.shape[0]), np.argmax(ok, axis=1)]
            elif name == "close":
                got[start : start + 4096] = x[np.arange(x.shape[0]), x.shape[1] - 1 - np.argmax(ok[:, ::-1], axis=1)]
            elif name == "high":
                got[start : start + 4096] = np.fmax.reduce(x, axis=1)
            elif name == "low":
                got[start : start + 4096] = np.fmin.reduce(x, axis=1)
            else:
                got[start : start + 4096] = np.nan_to_num(x, nan=0.0).sum(axis=1)
        have = np.isfinite(daily) & np.isfinite(got)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(got[have] - daily[have]) / np.maximum(np.abs(daily[have]), 1e-12)
        out[name] = {
            "coin_days": int(have.sum()),
            "share_within_1e-5": float((rel < 1e-5).mean()),
            "p99_9": float(np.percentile(rel, 99.9)),
            "max": float(rel.max()),
        }
    return out


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--panel", default="/mnt/nvme1/bruce/data/factor_idea_v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dev = load_dev(args.panel)
    dates, shape = dev.dates, dev.universe.shape
    column_of = {str(s): j for j, s in enumerate(dev.sec_ids)}
    first, last = dates[0], dates[-1]

    counts = np.zeros(shape, dtype=np.int32)
    for path in weeks(args.source, "close", first, last):
        for r, _, columns, values in read_days(path, dates, column_of):
            counts[r, columns] += np.isfinite(values).sum(axis=0)
    rows, coins = np.nonzero(counts)
    usable = counts[rows, coins] >= MIN_MINUTES_SHARE * MINUTES
    table = np.full(shape, -1, dtype=np.int64)
    table[rows, coins] = np.arange(rows.size)
    print(f"{rows.size} coin-days on {dates.size} rows, {first} to {last}; {usable.mean():.4f} usable", flush=True)

    sources, pattern_differs = {}, {}
    finite_close = None
    for name in ("close", *[f for f in FIELDS if f != "close"]):
        store = np.memmap(f"{args.out}/{name}.f32", dtype=np.float32, mode="w+", shape=(rows.size, MINUTES))
        store[:] = np.nan
        files = weeks(args.source, name, first, last)
        for path in files:
            for r, position, columns, values in read_days(path, dates, column_of):
                index = table[r, columns]
                have = index >= 0
                store[np.ix_(index[have], position)] = values[:, have].T
        store.flush()
        if name == "close":
            finite_close = np.isfinite(store)
        else:
            pattern_differs[name] = int((np.isfinite(store) != finite_close).sum())
        sources[name] = [{"file": os.path.basename(p), "bytes": os.path.getsize(p), "mtime": int(os.path.getmtime(p))} for p in files]
        print(f"{name}: {len(files)} weeks, {store.nbytes / 1e9:.2f} GB", flush=True)
        del store

    np.savez(f"{args.out}/index.npz", rows=rows.astype(np.int32), coins=coins.astype(np.int32), usable=usable, panel_rows=shape[0], panel_coins=shape[1], minutes=MINUTES, dates=dates, sec_ids=dev.sec_ids)
    fields = {n: np.memmap(f"{args.out}/{n}.f32", dtype=np.float32, mode="r", shape=(rows.size, MINUTES)) for n in FIELDS}
    bars = MinuteBars(rows.astype(np.int32), coins.astype(np.int32), usable, fields, {}, shape)
    for name, series in market_series(bars).items():
        series.tofile(f"{args.out}/{name}.f32")
    print("market series done", flush=True)

    quote, high, low = fields["quote_volume"], fields["high"], fields["low"]
    universe = np.flatnonzero(dev.universe[rows, coins])
    sample = universe[:: max(1, universe.size // 20000)]
    report = {
        "source": args.source,
        "first_row": str(first),
        "last_row": str(last),
        "coin_days": int(rows.size),
        "usable_share": float(usable.mean()),
        "universe_coin_days": int(universe.size),
        "universe_share_with_all_minutes": float((counts[rows[universe], coins[universe]] == MINUTES).mean()),
        "cells_whose_null_pattern_differs_from_close": pattern_differs,
        "sampled_universe_coin_days": int(sample.size),
        "zero_quote_volume_share_of_bars": float(np.nanmean(np.asarray(quote[sample]) == 0)),
        "high_below_low_bars": int((np.asarray(high[sample]) < np.asarray(low[sample])).sum()),
        "agrees_with_daily_panel": agree_with_daily(bars, dev),
        "source_files": sources,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "source_files"}, indent=1), flush=True)
    names = [*FIELDS, *MARKET_SERIES]
    report["sha256"] = {f"{n}.f32": sha256(f"{args.out}/{n}.f32") for n in names}
    report["sha256"]["index.npz"] = sha256(f"{args.out}/index.npz")
    with open(f"{args.out}/audit.json", "w") as f:
        json.dump(report, f, indent=1)
    for name in os.listdir(args.out):
        os.chmod(f"{args.out}/{name}", 0o444)
    print("checksums written; files are read-only", flush=True)


if __name__ == "__main__":
    main()
