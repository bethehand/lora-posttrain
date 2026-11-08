# Chapter 3: the minute-data pilot (tried, not adopted)

The question: daily klines give a formula nine numbers per coin per day. Should a formula also read one-minute bars?

**Decision (2026-11-05): the task stays on daily klines.** The code is kept and switched off: `Minute(...)` only computes with `FACTOR_IDEA_MINUTES` set, and `env.taskset.minutes` defaults to false. With it off, every verdict of the 670-item daily exam is the same as before the pilot, and all 1,129 tests pass.

## What was built

A formula stays a daily factor. Its `Minute(...)` part runs inside one coin's UTC day, and a `day_` operator (`day_sum`, `day_max`, `day_center`, `day_corr`, ...) turns it into one number for that day. Nothing inside a day reads another day or another coin, apart from two market series fixed when the bars are built.

- **Bars:** `scripts/factor/build_minute_store.py` froze 2021-12-28 to 2025-12-31 as 19 GB of float32, one line per coin-day with any bar: 380,329 coin-days, with checksums.
- **Agreement with daily bars:** open, high, low and close agree on at least 99.998% of universe coin-days, volume fields on about 99.6%. The gaps sit on three days. On 2023-08-17 (all coins, about +3%) and for QNT on 2023-09-21 (+126%) Binance's own public daily and minute files disagree. On 2024-10-29 (all coins, about +3%) the research system's minute table has real volume for 75 minutes where the public files are almost empty.
- **Agreement with public files:** three coin-months compared. ETH 2023-06 and SOL 2024-09 match on every minute and field; BTC 2024-03 differs on a dozen or so minutes.
- **Reproduction:** 13 of the research system's Bases written in the language match the platform's stored values, mean daily rank correlation 0.993 to 1.000. Each coin-day reads only its own minutes, so those 13 Bases don't look past the day's close.

## Cost, on real formulas

| | Mean | Median | p90 | Max |
|---|---|---|---|---|
| `test_factor`, nothing cached | 3.4 s | 3.0 s | 5.9 s | 11.0 s |
| Score (gates and IC), nothing cached | 9.9 s | 8.9 s | 15.6 s | 21.6 s |

Daily formulas take about 0.9 s and 0.7 s. An RL step of 128 rollouts would add 1.8 to 3.2 minutes of sandbox time on 16 workers, without counting the cache. One factor value per minute instead would cost about 11 CPU minutes per test, and a typical one-minute move (5.6 bp) is below the account's cost of about 7 bp.

## How much information minutes add

Research window only, each field taking the better of its raw form and a 10-day mean. "Net" also removes 12 daily-kline features and their 10-day means.

| | 12 daily-kline features | 128 daily Bases, mostly from minutes |
|---|---|---|
| Median absolute incremental IC | 0.0107 | 0.0083 |
| Highest | 0.0203 | 0.0281 |
| Fields at 0.02 or more | 1 | 10 |
| Median absolute net IC | | 0.0030 |
| Fields with net IC at 0.01 or more | | 3 |

Most minute features repackage what daily bars already say. A few families don't: when in the day losses and gains fall, and how closely a coin follows the market. Researchers chose these Bases on the same history, so these numbers lean optimistic. On 2025 the concentration and last-six-hours families mostly faded, while market sync held up.

## Verifier

Six new perturbations rearrange a day's minutes and leave every daily bar alone, so a formula has to look inside the day to respond. On real data they move an honest formula's rank by 24 to 47 percentile points.

The first minute exam gave 1 false fail in 131 and 10 false passes in 549. After two new rules, and after moving the one false fail to the grey group because the question itself was mislabelled, the second exam gave none of either. This is not a blind test: the same author wrote the questions and the rules, and the rules changed after reading the errors.

## Baselines

12 minute ideas and 31 daily ideas, 4 rollouts each, scored on 2025. `F` gives the earlier result with the daily-only prompt, for comparison.

| Run | Pass rate | Incremental IC among passes |
|---|---|---|
| DeepSeek, minute ideas, with test_factor | 44/48, 91.7% | +0.0078 |
| DeepSeek, minute ideas, one shot | 40/48, 83.3% | +0.0057 |
| DeepSeek, daily ideas, with test_factor | 88/124, 71.0% (`F` 83.1%) | +0.0090 (`F` +0.0071) |
| OLMo 3 7B Instruct, minute ideas, with test_factor | 0/48 | |
| OLMo 3 7B Instruct, minute ideas, one shot | 0/48 | |
| OLMo 3 7B Instruct, daily ideas, with test_factor | 3/124, 2.4% (`F` 2.4%) | +0.0093 (`F` +0.0064) |
| OLMo 3 7B Instruct, daily ideas, one shot | 0/124 (`F` 11/124, 8.9%) | (`F` -0.0004) |

DeepSeek's daily pass rate fell mostly because more episodes used all five tests without handing in (23 against 14). OLMo's one-shot rate fell mostly because it left out the opening `<factor>` tag (105 episodes without an answer against 46). The longer prompt is a real cost to an untrained small model. The 44 passing DeepSeek formulas were read by hand: 43 answer their idea, and one weights volume by the size of each fall, which mixes in volatility.

## What the research manual said about minutes

The section below was removed from `task/manual.zh.md` when the decision was made.

## 想法说的是一天之内的事：用分钟数据

日 K 线只有一天的总量和开高低收。想法里出现"什么时候""集中在哪几分钟""上涨时放量""跟不跟大盘"这类说法，说的是一天之内的路径，日 K 线看不到，要用 `Minute(...)`。**这一节是我从你的 Base 定义总结的，公式已经和平台的因子值对过账，排名相关 0.993 到 1.000。写法习惯请你看一下。**

- **先在一天内算，再汇总成一个数。** `Minute('quote_volume')` 是一天 1,440 个值，要用 `day_sum`、`day_max`、`day_center` 这类算子变成每天一个数，之后才能用 `ts_mean`、`sec_rank`，也才能和 `Factor(...)` 放在一起算。
- **问"什么时候"用时间重心。** 成交额集中在早还是晚：`Minute('quote_volume').day_center()`。依据：`quote_vol_time_center_1d`。跌幅集中在早还是晚：`Minute('return').clip(upper=0).Abs().day_center()`。依据：`down_ret_time_center`。`day_center` 只看正的部分，所以看跌幅要先取绝对值。
- **问"哪个时段"用占比，不用绝对额。** `Minute('quote_volume').between(18, 24).day_sum().safe_div(Minute('quote_volume').day_sum())`。依据：`us_quote_vol_share_1d`。只写分子量的是这个币成交多不多，不是成交晚不晚。
- **问"有多集中"用最大值占总量的比例。** `Minute('quote_volume').day_max().safe_div(Minute('quote_volume').day_sum())`。依据：`quote_vol_concentration`。
- **问"上涨时放量"用条件求和。** `where(Minute('return') > 0, Minute('volume'), 0).day_sum().safe_div(where(Minute('return') != 0, Minute('volume'), 0).day_sum())`。依据：`dir_volume_share`，它也是 Strategy10 的分量。
- **问"跟不跟大盘"用相关系数。** `Minute('return').day_corr(Minute('market_return'))`。依据：`ret_mkt_cor`。
- **单日分钟特征噪声大，通常再平滑。** 汇总成一个数之后再 `.ts_mean(window=10, min_periods=8)`。依据：Strategy10 的 `dir_vol_share_pow2p5_mean10_ema10`。**待确认：分钟特征你习惯平滑多久？**
- **想法只说日度的事，就不要用分钟数据。** 成交量大不大用 `Factor('quote_volume')` 就够了，写成分钟再加总只会更长。
