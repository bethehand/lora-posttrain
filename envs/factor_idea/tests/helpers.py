"""A synthetic market for the tests: random walks with listings, delistings and holes."""
import numpy as np

from factor_idea.dsl import FIELDS
from factor_idea.minutes import MinuteBars, market_series


def synthetic_fields(t=400, n=40, seed=0, holes=True):
    rng = np.random.default_rng(seed)
    log_ret = rng.standard_t(4, size=(t, n)) * 0.03
    close = 10.0 ** rng.uniform(-3, 3, size=(1, n)) * np.exp(np.cumsum(log_ret, axis=0))
    open_ = np.vstack([close[0] * 0.99, close[:-1]])
    top = np.maximum(open_, close)
    bottom = np.minimum(open_, close)
    high = top * np.exp(np.abs(rng.normal(0, 0.02, size=(t, n))))
    low = bottom * np.exp(-np.abs(rng.normal(0, 0.02, size=(t, n))))
    # Dollar volume across coins spans about three decades, as it does on the real panel.
    vwap = np.sqrt(high * low)
    quote = 10.0 ** rng.uniform(5, 8, size=(1, n)) * np.exp(rng.normal(0, 0.5, size=(t, n)))
    volume = quote / vwap
    trades = np.maximum(np.round(quote / 500.0), 1.0)
    share = rng.uniform(0.35, 0.65, size=(t, n))
    fields = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": quote,
        "num_trades": trades,
        "taker_base_vol": volume * share,
        "taker_quote_vol": quote * share,
    }
    if holes:
        listed = rng.integers(0, t // 3, size=n)
        delisted = rng.integers(2 * t // 3, t + 1, size=n)
        mask = np.ones((t, n), dtype=bool)
        for j in range(n):
            mask[: listed[j], j] = False
            mask[delisted[j] :, j] = False
        # The first coins trade the whole time so that every test has a full cross section.
        mask[:, : n // 2] = True
        for name in FIELDS:
            fields[name] = np.where(mask, fields[name], np.nan)
    return fields


def synthetic_minutes(t=120, n=20, minutes=144, seed=0):
    """Minute bars and the daily fields they add up to. Days have 144 bars to keep the tests quick;
    no operator cares how many there are."""
    rng = np.random.default_rng(seed)
    listed = rng.integers(0, t // 3, size=n)
    listed[: n // 2] = 0
    rows, coins, parts = [], [], {name: [] for name in FIELDS}
    daily = {name: np.full((t, n), np.nan) for name in FIELDS}
    for j in range(n):
        days = t - listed[j]
        moves = rng.standard_t(4, size=(days, minutes)) * 0.03 / np.sqrt(minutes)
        path = np.cumsum(moves.ravel()).reshape(days, minutes)
        close = 10.0 ** rng.uniform(-3, 3) * np.exp(path)
        open_ = np.empty_like(close)
        open_[:, 1:] = close[:, :-1]
        open_[1:, 0] = close[:-1, -1]
        open_[0, 0] = close[0, 0] * 0.999
        high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.0005, size=close.shape)))
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.0005, size=close.shape)))
        # Each coin leans to its own time of day, a little differently every day.
        favourite = rng.uniform(0.45, 0.55) + rng.normal(0, 0.03, size=(days, 1))
        shape = np.exp(-((np.arange(minutes) / minutes - favourite) ** 2) / 0.25)
        quote = 10.0 ** rng.uniform(5, 8) * shape * np.exp(rng.normal(0, 1.0, size=close.shape))
        volume = quote / np.sqrt(high * low)
        trades = np.maximum(np.round(quote / 500.0), 1.0)
        share = rng.uniform(0.35, 0.65, size=close.shape)
        bars = {
            "open": open_, "high": high, "low": low, "close": close, "volume": volume, "quote_volume": quote,
            "num_trades": trades, "taker_base_vol": volume * share, "taker_quote_vol": quote * share,
        }
        at = np.arange(listed[j], t)
        rows.append(at)
        coins.append(np.full(days, j))
        for name, x in bars.items():
            parts[name].append(x.astype(np.float32))
        daily["open"][at, j] = open_[:, 0]
        daily["high"][at, j] = high.max(axis=1)
        daily["low"][at, j] = low.min(axis=1)
        daily["close"][at, j] = close[:, -1]
        for name in ("volume", "quote_volume", "num_trades", "taker_base_vol", "taker_quote_vol"):
            daily[name][at, j] = bars[name].sum(axis=1)
    rows, coins = np.concatenate(rows), np.concatenate(coins)
    order = np.lexsort((coins, rows))
    rows, coins = rows[order], coins[order]
    fields = {name: np.concatenate(parts[name])[order] for name in FIELDS}
    bars = MinuteBars(rows, coins, np.ones(rows.size, dtype=bool), fields, {}, (t, n))
    bars.market.update(market_series(bars))
    return bars, daily
