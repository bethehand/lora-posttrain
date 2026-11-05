# Chapter 3, week 1: sandbox, verifier exam, first baselines

Task: one simple idea in, one factor expression out, one shot. Scored in a look-back-only sandbox.

## Data

Frozen read-only copy of the research system's daily wide tables, 4,549 files with hashes matching the source (`binance_daily_20261105/SOURCE_SHA256`). All nine fields match Binance's public kline files on the five coin-months checked, most days exactly; the largest difference is 0.18%, in the volume fields of BTC in 2024-03. `close` and `quote_volume` run to 2026-11-05; every other field stops at 2026-10-18, so the last sealed factor date is 2026-10-16. `high >= low` and the like hold on all 535,240 cells.

## Operators

685 tests pass. Every operator and 150 random expressions pass the poison and truncation tests, and a planted look-ahead operator is caught. 29 operator checks against polars on real rows agree to rounding, with identical null patterns.

## Verifier exam

670 formulas with verdicts decided in advance (`verifier_exam.json`). Not blind: the rules were changed after the first runs.

| Group | Should pass | Passed |
|---|---|---|
| Honest, hand-written, both directions | yes | 134/134 |
| From the research system | yes | 10/10 |
| Hard honest | yes | 5/5 |
| Sign flipped | no | 0/144 |
| Ignores the idea's data | no | 0/182 |
| Mostly other data, dose 0.02 / 0.1 / 0.3 | no | 0/31, 0/30, 0/30 |
| Mostly other data, dose 0.5 | no | 2/30 |
| One factor touching every kind of data | no | 1/7 |
| Hard cheats | no | 0/9 |
| Depends on price units | no | 0/13 |
| Degenerate | no | 0/7 |

False fails 0/149; false passes 3/483 (0.6%). Known blind spots: an equal blend of the idea's data with unrelated data passes 15/30; volume from a month ago and volume volatility pass for a volume idea.

## Baselines

31 draft ideas. Reward period 2025; the sealed period was not read.

| Model | Answers | Has a tag | Parses | Computes | Passes the idea check | Mean reward, IC arm |
|---|---|---|---|---|---|---|
| OLMo-3-7B-Instruct, no training | 248 | 61.3% | 16.5% | 15.3% | 4.8% | 0.032 |
| deepseek-flash | 124 | 100% | 100% | 100% | 100% | 0.671 |

55% of OLMo's replies ran out of the 1,500-token budget. All 124 DeepSeek formulas were read by hand and are reasonable, which is the only out-of-sample check on false fails so far.

## Does IC persist between years?

Across 109 distinct DeepSeek formulas in 31 ideas (`ic_persistence.json`), rank correlation of incremental IC, with 95% intervals from resampling ideas:

| Years | All formulas | Within one idea |
|---|---|---|
| 2023 to 2024 | 0.77 (0.59, 0.87) | 0.62 (0.28, 0.84) |
| 2024 to 2025 | 0.85 (0.71, 0.92) | 0.75 (0.57, 0.87) |
| 2023 to 2025 | 0.74 (0.50, 0.87) | 0.63 (0.31, 0.82) |

One model's formulas, and several ideas have almost no spread between formulas. It says a reward on one year's IC is not pure noise; it says nothing about 2026.

Not yet run: OLMo Think (not downloaded) and Qwen3-4B.

## With the test_factor tool

After the first baselines the task changed: the model may test up to 5 candidates on July 2022 to December 2024, then hands in one factor, scored on 2025. All four arms below ran through the same eval pipeline (31 ideas, 4 rollouts, the model's own sampling settings).

| | OLMo one-shot | OLMo with tool | deepseek-flash one-shot | deepseek-flash with tool |
|---|---|---|---|---|
| Passes the idea check | 8.9% | 0.8% | 95.2% | 83.1% |
| Mean incremental IC among passes, 2025 | -0.0004 | 0.0025 (1 episode) | 0.0017 | 0.0071 |
| Mean reward, IC arm | 0.057 | 0.005 | 0.638 | 0.608 |
| Tests per episode | | 0.8 | | 4.6 |

OLMo's tool calls: 80 of 99 results were expression errors. DeepSeek's 21 lost episodes: 14 used all turns without answering, 5 were rightly failed, and 2 were false fails, a trade-size formula with volume regressed out; that rule was fixed afterwards and these numbers predate the fix.

Does testing help on the year the model never sees? deepseek-flash, resampling the 31 ideas (`agent/deepseek-compare.json`):

| | 2022-2024, seen | 2025, unseen |
|---|---|---|
| First formula tested | 0.0023 | 0.0014 |
| Formula handed in | 0.0084 | 0.0078 |
| Gain | +0.0057 (0.0039, 0.0074) | +0.0061 (0.0041, 0.0081) |

The in-sample gain carries over almost in full, with up to 5 tests per episode. With the tool against one-shot: IC among passes +0.0047 (0.0027, 0.0067); pass rate -12 points (-19, -6); mean reward -0.030 (-0.086, +0.022), because of the lost episodes.

## The text before a tool call

vLLM's `olmo3` parser and our training parser dropped whatever the model wrote before a tool call, so a model could not read its own earlier reasoning. `olmo3_keep_text` (`setup/vllm_olmo3_keep_text.py`, `setup/renderers_olmo3.py`) keeps it; `olmo3` stays as it was for chapter 2. Checks: on 29 turns of teacher trajectories, the SFT rendering, the RL rendering and the served prompt are identical token by token (`scripts/factor/parity_teacher.py`); on 120 live OLMo turns, prompts and parsed calls match 120 of 120 and kept text matches 47 of 47, and 22 of 22 hand-written edge cases agree (`scripts/olmo3_parity.py`).

Base OLMo with the tool, rerun with this parser and with the range of `ts_rank` in the prompt: passes 3 of 124 (2.4%), against 1 of 124 before; 85 of 97 test results are expression errors. It writes text before a call in only 1 of 97 such turns, so before training it barely reasons at all.
