"""One-minute bars, and what the Minute(...) part of a formula computes from them.

A formula stays a daily factor: its Minute(...) part runs inside one coin's day, and a day_
operator turns that into one number for the day. Bars are kept per coin-day, so a whole
window is one contiguous read and the numbers come out without a Python loop over days.
Nothing here looks at another day or another coin, apart from two market series fixed when
the bars are built. dev bars stop at 2025-12-31: code that only loads them can't read 2026.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from factor_idea import ops
from factor_idea.dsl import FIELDS, MARKET_SERIES, Const, MinuteField, Node, Op
from factor_idea.evaluate import EvalTimeout

DATA_DIR = os.environ.get("FACTOR_IDEA_DATA", "/mnt/nvme1/bruce/data/factor_idea_v1")
DEV_DIR = "minutes_dev"
MIN_MINUTES_SHARE = 0.6  # a coin-day with fewer bars than this, like a listing day, has no minute values
_CHUNK = 256
_CACHE_SIZE = 64


def minute_returns(close: np.ndarray, open_: np.ndarray) -> np.ndarray:
    """Each bar's close over the previous bar's. A day's first bar has only its own open to go by."""
    out = np.empty_like(close)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[0] = close[0] / open_[0] - 1.0
        out[1:] = close[1:] / close[:-1] - 1.0
    return out


def _run(node: Node, read: Callable[[str], np.ndarray], memo: dict):
    if isinstance(node, Const):
        return node.value
    if isinstance(node, MinuteField):
        return read(node.name)
    if node in memo:
        return memo[node]
    args = [_run(a, read, memo) for a in node.args]
    with np.errstate(all="ignore"):
        out = np.asarray(ops.TABLE[node.name](*args, **dict(node.params)), dtype=np.float64)
    # ops.clean, without its copy: this array belongs to this chunk and nothing else yet.
    bad = np.isinf(out)
    if bad.any():
        out = np.where(bad, np.nan, out)
    memo[node] = out
    return out


@dataclass
class MinuteBars:
    rows: np.ndarray  # per coin-day: its panel row, ascending
    coins: np.ndarray  # per coin-day: its panel column
    usable: np.ndarray  # per coin-day: enough bars to mean anything
    fields: dict[str, np.ndarray]  # coin-days x minutes, float32
    market: dict[str, np.ndarray]  # panel rows x minutes, float32
    shape: tuple[int, int]  # the panel's rows and columns
    _cache: OrderedDict = field(default_factory=OrderedDict, repr=False)

    def __post_init__(self) -> None:
        self.starts = np.searchsorted(self.rows, np.arange(self.shape[0] + 1))
        self.table = np.full(self.shape, -1, dtype=np.int32)
        self.table[self.rows, self.coins] = np.arange(self.rows.size, dtype=np.int32)

    def _reader(self, part, change: Callable | None, day: dict | None) -> Callable[[str], np.ndarray]:
        """Reads ``part``'s bars as minutes x coin-days float64, through ``change`` if given. Each once."""
        seen: dict[str, np.ndarray] = {}

        def original(name: str) -> np.ndarray:
            if name not in seen:
                source = self.market[name][self.rows[part]] if name in self.market else self.fields[name][part]
                seen[name] = np.asarray(source).T.astype(np.float64)
            return seen[name]

        changed: dict[str, np.ndarray] = {}

        def read(name: str) -> np.ndarray:
            if name in changed:
                return changed[name]
            if name == "return":
                out = minute_returns(read("close"), read("open"))
            elif change is None or name in self.market:
                out = original(name)
            else:
                out = change(name, original, day)
            changed[name] = out
            return out

        return read

    def compute(self, nodes: list[Op], index, *, change: Callable | None = None, day: dict | None = None, deadline: float | None = None) -> np.ndarray:
        """Day_ operators over coin-days ``index``, a slice or ascending positions: nodes x coin-days.

        One pass for all of them, so what they share is computed once.
        """
        whole = np.arange(self.rows.size)[index]
        out = np.empty((len(nodes), whole.size))
        for start in range(0, whole.size, _CHUNK):
            if deadline is not None and time.monotonic() > deadline:
                raise EvalTimeout("expression took too long to compute")
            part = whole[start : start + _CHUNK]
            # A run of neighbours reads as a view of the file instead of a scattered copy.
            take = slice(int(part[0]), int(part[-1]) + 1) if part[-1] - part[0] + 1 == part.size else part
            chunk_day = {k: v[start : start + _CHUNK] for k, v in day.items()} if day else None
            read, memo = self._reader(take, change, chunk_day), {}
            for i, node in enumerate(nodes):
                out[i, start : start + _CHUNK] = _run(node, read, memo)
        out[:, ~self.usable[whole]] = np.nan
        return out

    def prepare(self, nodes: list[Op], rows: int, deadline: float | None = None) -> None:
        """Make sure the panel's first ``rows`` rows of these operators are in the cache. Coin-days on later rows are not read."""
        pending: dict[int, list[Op]] = {}
        for node in nodes:
            entry = self._cache.get(node)
            done = 0 if entry is None else entry[0]
            if done < rows:
                pending.setdefault(done, []).append(node)
        for done, group in pending.items():
            first, last = int(self.starts[done]), int(self.starts[rows])
            values = self.compute(group, slice(first, last), deadline=deadline)
            for i, node in enumerate(group):
                entry = self._cache.setdefault(node, [0, np.full(self.shape, np.nan)])
                entry[1][self.rows[first:last], self.coins[first:last]] = values[i]
                entry[0] = rows
        # Everything this expression is about to read outlives the rest of the cache.
        for node in nodes:
            if node in self._cache:
                self._cache.move_to_end(node)
        while len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)

    def day_values(self, node: Op, rows: int, deadline: float | None = None) -> np.ndarray:
        """T x N values for the panel's first ``rows`` rows."""
        self.prepare([node], rows, deadline)
        out = self._cache[node][1][:rows].view()
        out.setflags(write=False)
        return out


class PatchedMinutes:
    """The same bars with a change on a few coin-days. Only those coin-days are computed again."""

    def __init__(self, bars: MinuteBars, rows: np.ndarray, coins: np.ndarray, change: Callable, day: dict, known: dict | None = None):
        self.bars, self.change, self.known = bars, change, known or {}
        index = bars.table[rows, coins]
        keep = np.flatnonzero(index >= 0)
        order = keep[np.argsort(index[keep], kind="stable")]
        self.index, self.rows, self.coins = index[order], rows[order], coins[order]
        self.day = {k: v[order] for k, v in day.items()}
        self._patched: dict[Op, np.ndarray] = {}

    def prepare(self, nodes: list[Op], rows: int, deadline: float | None = None) -> None:
        todo = [node for node in nodes if node not in self._patched]
        if todo and self.index.size:
            values = self.bars.compute(todo, self.index, change=self.change, day=self.day, deadline=deadline)
            self._patched.update(zip(todo, values))

    def day_values(self, node: Op, rows: int, deadline: float | None = None) -> np.ndarray:
        self.prepare([node], rows, deadline)
        base = self.known.get(node)
        if base is None:
            base = self.bars.day_values(node, rows, deadline)
        out = np.array(base)
        if self.index.size:
            out[self.rows, self.coins] = self._patched[node]
        out.setflags(write=False)
        return out


def market_series(bars: MinuteBars) -> dict[str, np.ndarray]:
    """What every coin's minutes are compared with: the median coin's return, and the average coin's
    share of its own day's dollar volume, in each minute."""
    rows, minutes = bars.shape[0], next(iter(bars.fields.values())).shape[1]
    out = {name: np.full((rows, minutes), np.nan, dtype=np.float32) for name in MARKET_SERIES}
    for r in range(rows):
        first, last = int(bars.starts[r]), int(bars.starts[r + 1])
        if first == last:
            continue
        ret = minute_returns(np.asarray(bars.fields["close"][first:last]).T.astype(np.float64), np.asarray(bars.fields["open"][first:last]).T.astype(np.float64))
        quote = np.asarray(bars.fields["quote_volume"][first:last]).T.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = quote / np.nansum(quote, axis=0)
        # Whole minutes with no coin at all give the all-NaN warnings these two functions would raise.
        ok = np.isfinite(ret).any(axis=1)
        out["market_return"][r, ok] = np.nanmedian(ret[ok], axis=1)
        ok = np.isfinite(share).any(axis=1)
        out["market_volume_share"][r, ok] = np.nanmean(share[ok], axis=1)
    return out


def load_minutes(data_dir: str | None = None) -> MinuteBars | None:
    """The dev bars, memory-mapped so that every worker shares one copy; None if they aren't built."""
    root = os.path.join(data_dir or DATA_DIR, DEV_DIR)
    if not os.path.isdir(root):
        return None
    with np.load(os.path.join(root, "index.npz"), allow_pickle=False) as z:
        rows, coins, usable = z["rows"], z["coins"], z["usable"]
        shape = (int(z["panel_rows"]), int(z["panel_coins"]))
        minutes = int(z["minutes"])
    fields = {
        name: np.memmap(os.path.join(root, f"{name}.f32"), dtype=np.float32, mode="r", shape=(rows.size, minutes))
        for name in FIELDS
    }
    market = {
        name: np.memmap(os.path.join(root, f"{name}.f32"), dtype=np.float32, mode="r", shape=(shape[0], minutes))
        for name in MARKET_SERIES
    }
    return MinuteBars(rows, coins, usable, fields, market, shape)
