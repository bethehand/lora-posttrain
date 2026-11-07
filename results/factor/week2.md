# Chapter 3: idea bank, verifier additions and the teacher pilot

## Decisions

- Base model: OLMo 3 7B Instruct. OLMo 3 7B Think was dropped: its chat template can't take tools, and with a borrowed template and 2,048 tokens per turn it never finished a reply, often ending in repeated phrases. A test with more tokens did not finish before the decision, so this is not a verdict on Think in general.
- Daily klines only. See minute_pilot.md.
- Teacher: DeepSeek (`deepseek-flash`).
- `test_factor` keeps its current report. The eight strategies for SFT2 stand as drafted: start simple, who to compare with, compare windows, compare forms, denoise, handle extremes, stability, reduce style exposure.
- `neg_ret_disp_med15` is out of training and out of the manual, and the idea built on it is gone: its author could not recall when it was settled, and it was registered in October 2026.

## Idea bank

`task/ideas_v2.jsonl`: 115 ideas in 20 families. Each carries a reference formula, unseen by any model, that passes the gates, so every idea can be answered. Reference results are in `ideas_v2_check.json`, on the research window only.

Five families are held out for the exam: `taker_flow`, `trade_count`, `illiquidity`, `trend_position`, `market_relative`, 31 ideas. They were drawn with a fixed seed before any model answered these ideas: from families with at least five ideas, one taker, two flow and two price families. The reference formulas' results had been seen, though not per split.

## Verifier

Two perturbations were added for new concepts. `solid_body` shortens both shadows and keeps open and close. `volume_on_up_days` moves a window's bigger volumes onto its up days and keeps its total.

The exam grew to 924 items, 866 with a known verdict: false fail 0/173, false pass 9/693 (1.3%). The 662 items kept from the earlier exam all got the same verdict. The false passes are blends mostly made of a volume-price formula, plus one omnibus formula. The rules and the questions have one author, so this is not a blind test.

## Teacher pilot

DeepSeek with the second manual, every idea once. The held-out families' traces only check the ideas and stay out of training.

| | Before the manual (30 or 31 older ideas, 124 episodes) | With the manual (115 ideas) |
|---|---|---|
| Passed the gates | 83.1% | 104/115, 90.4% |
| Handed in an answer | 88.7% | 114/115 |
| Mean tests per episode | 4.6 | 3.8 |
| Tests with a reason written first | 5/578 | 436/436 |
| Final formula was one it tested | | 114/114 |
| Trajectories mentioning the manual or its sources | | 0 |

Tests that came back as errors: 4 of 436. So the teacher barely shows how to recover from a bad formula, which is what OLMo needs most. Recovery has to be taught separately.

## Reading the pilot by hand

One reader's judgement of every formula, the same person who wrote the verifier. It is the closest thing so far to a blind check, and it shows the exam was too kind.

**Honest formulas the verifier failed, 4 to 6 of about 105, far above the exam's 0/173:**

- `volume_longrun_up`: a 180-day median. Perturbing at most 60 days can't move it.
- `pv_corr_up`, `pv_corr_down`: correlation of price level with volume, and of returns with volume changes. The rearrangement barely moves these.
- `worst_day_bounce`: the worst day divided by volatility. Amplifying every move leaves the ratio alone.
- `rise_without_buyers_down`: a conditional formula that is zero for coins that did not rise, so the median coin does not move.
- `flow_beats_price_up`: a product of two conditions. Borderline.

**Weak answers the verifier passed, 3 to 4 of 104, above the exam's 1.3%:**

- `high_beta_up`: plain volatility, not beta.
- `taker_surge_up`: the level of the taker share, not a surge above normal. A known blind spot.
- `fresh_high_up`: the rank of the latest close, not how recently the high was made.
- `close_low_bounce`: the position in a twenty-day range, not within the day.

**Failures that are right:** `upside_vol_up` handed in a ratio of up moves to down moves; `momentum_1m` a median of daily ranks, which a large gain barely moves, borderline; `rsi_high_up` no answer. `doji_down` and `doji_up` measured body size against its own history. The ideas said "small-bodied" without saying relative to what, so their wording was made explicit afterwards.

A false fail costs little: SFT loses a good trace and RL gives an honest formula 0. A false pass is what RL can exploit, so it gets a human audit after every stage.

## Splits

Besides the 31 held-out ideas, 15 validation ideas were drawn with a fixed seed, one from each training family, which leaves 69 ideas for training. Validation chooses settings. Held-out ideas stay unread by trained models until a stage's settings are final.

## Baselines on the new bank

Each model, untrained, on all 115 ideas, 4 rollouts each, scored again by the verifier as it was fixed below. These are the numbers every later stage is compared with. Pass rates:

| | Train | Val | Held-out | All | Answered | Tests per episode | Tests that errored | IC among passes |
|---|---|---|---|---|---|---|---|---|
| OLMo 3 7B, tools | 1.8% | 3.3% | 0.0% | 1.5% | 33.0% | 0.73 | 89.5% | -0.0013 |
| OLMo 3 7B, one-shot | 6.9% | 11.7% | 2.4% | 6.3% | 65.7% | | | +0.0047 |
| Qwen3-4B, tools | 15.6% | 30.0% | 25.8% | 20.2% | 84.3% | 3.18 | 25.3% | +0.0006 |
| Qwen3-4B, one-shot | 21.0% | 31.7% | 25.0% | 23.5% | 89.6% | | | -0.0016 |
| DeepSeek, tools, no manual | 77.9% | 73.3% | 79.0% | 77.6% | 86.5% | 4.40 | 3.5% | +0.0050 |
| DeepSeek, one-shot | 84.4% | 95.0% | 86.3% | 86.3% | 89.6% | | | +0.0013 |

Testing costs DeepSeek some passes, mostly by running out of turns, and raises the IC of what passes. OLMo starts far below Qwen on the same small-model footing: most of its episodes never make a test at all, and nine in ten of the tests it does make are errors.

## Trajectories for S1

The teacher with the manual worked each training idea in 6 wordings, 5 times each: 2,067 of 2,070 episodes came back and 1,850 passed the gates (89.5%), at 3.69 tests per episode. Three were lost, two to a bug on replies with two tool calls, since fixed, and one to an empty reply from the API.

Recovery trajectories start from a formula OLMo really wrote that came back as an error. The first run lost all 70 episodes in two seconds and its log still said it was done, because the API wants a thinking field on any past tool call and the replayed turn had none. Fixed and tested. The second run also read OLMo's three older tool baselines and asked for two rollouts per formula, since 70 trajectories were under 4% of the data on the one skill the student lacks most: 228 episodes from 113 distinct broken formulas, 188 passed.

The dataset keeps what passed the gates, handed in a tested formula and stayed off the teacher's brief: 2,021 trajectories on all 69 training ideas, 1,834 ordinary and 187 recovery. Nine went for words from the brief ("as instructed") or for Chinese. The longest is 3,448 tokens against a limit of 4,096, so no hand-in gets cut off, and about 35% of tokens are trained. In recovery rows the broken turn sits in the context and out of the loss.

## Stages continue from a file

prime-rl's own resume can't carry an SFT adapter into RL: it wants orchestrator state an SFT checkpoint doesn't have, and it keeps counting steps. `LORA_POSTTRAIN_INIT_ADAPTER` loads an adapter right after the reset, and the trainer's startup broadcast then gives inference that adapter from the first rollout. Tested from the pilot's step 30: the new run's first broadcast equals the file in all 448 tensors, and its first loss is 1.09 where a fresh adapter starts at 3.37. The RL half of this hasn't been run yet.

The first launch of S1 died in 40 seconds on the dataset loader, which fixes its schema from the first rows: recovery rows carried a `train` field the earlier rows didn't. Every assistant turn now states it, and `check_sft_dataset.py` reads the file through the same loader. The offline datasets had two rows that ran out of turns and ended on a tool result, which the trainer refuses. They keep their negative weight and lose the last tool result.

## The verifier, after a red team and two audits

A model told every rule and given eight tries got 36 of its 37 answers past the gates (`verifier_redteam.jsonl`), about 30 of them unfaithful by one reader. On average they earned no more than honest answers, +0.0050 against +0.0053 on the same ideas, but the best reached +0.026. Two independent code audits found no look-ahead or leakage, and did find faults the exam had missed, some because it skipped those pairs.

Checked on the real panel, then fixed:

- A rally also raises volatility, so a low-volatility formula passed as a reversal idea, for reward 0.85 where honest answers to that idea average 0.65. Ideas keyed on return now also take a fall to half the price, and a formula that moves the way the idea wants after either one fails. `fat_tail_down` was mislabelled and is now a volatility idea. All 115 references still pass, and 20 of 20 such formulas now fail.
- `min_periods` could outlast the gates' history, which let a formula behave differently from 2025. It is capped at 365, and a formula that goes blank when scored earns nothing.
- A timeout during scoring still paid 0.3 or 1.0. It now pays nothing. The time limit didn't bind long chains, a number too large for a float crashed the parser, and one killed worker broke the pool for good.

The exam now has 951 items: false fail 0/173, false pass 8/720. The changes barely touch earlier results: DeepSeek lost 2 passes of 359 and Qwen 1 of 109, and 8 of S1's 1,834 ordinary trajectories fail now.

They close specific holes, not the gap. On the fixed verifier a second red team got 35 of 36 through (`verifier_redteam_v2.jsonl`), though with lower IC among passes, +0.0021 against +0.0047. Gates test behaviour and can't read meaning. Two ways of doing better without a model were measured and fell short:

| Check | Red team caught | Blends caught | Teacher formulas rejected |
|---|---|---|---|
| Rank correlation at least 0.8 with the key data taken away | 22% | about 0 | 0.8% |
| Rank correlation below 0.2 with the idea's reference | 58% | 5 to 10% | 6.8% |
| LLM judge on the idea and formula | 92% | 87 to 100% | 4% |

A blend here is an honest formula with a strong signal the idea doesn't mention added in. It keeps passing, and earns 0.05 to 0.18 more. The judge's rates come from the formulas its rubric was tuned on. On ones it wasn't, it rejected 32 of the second red team's 35 and 8 of 200 teacher formulas. So arm C adds the judge to arm B's reward. The owner of the research system chose not to label formulas for the rubric, so nobody independent has checked it.

## SFT1 on validation ideas

Rank 32, by half epochs, against 3.3% untrained and 8.3% for the 30-step pilot:

| Steps | 70 | 140 | 210 | 280 | 350 | 420 |
|---|---|---|---|---|---|---|
| Passed | 20.0% | 33.3% | 56.7% | 55.0% | 75.0% | 71.7% |
| Answered | 56.7% | 86.7% | 90.0% | 88.3% | 93.3% | 93.3% |
| Tests that errored | 30.9% | 27.3% | 15.2% | 8.7% | 10.1% | 8.0% |
| IC among passes | +0.0110 | +0.0093 | +0.0068 | +0.0101 | +0.0082 | +0.0072 |

Sixty episodes each, so differences of a few points are noise. Ten of the 15 val ideas mirror a train idea up to sign, which flatters these numbers, and the held-out families are the real test. At step 70 the student didn't know when to stop, and 43% of its episodes never handed in. More training fixed that without changing the prompt.

Rank 8 ran beside rank 32. Rank 32's best beat rank 8's by 8.3 points, so the rule sent rank 128 on as well:

| Steps | 70 | 140 | 210 | 280 | 350 | 420 |
|---|---|---|---|---|---|---|
| Rank 8, passed | 25.0% | 28.3% | 51.7% | 40.0% | 66.7% | 63.3% |
| Rank 32, passed | 20.0% | 33.3% | 56.7% | 55.0% | 75.0% | 71.7% |
| Rank 128, passed | 16.7% | 30.0% | 58.3% | 55.0% | 53.3% | 65.0% |

The rules give rank 32 at step 350: 75.0% passed, 93.3% answered, reward 0.559, IC among passes +0.0082, 3.8 tests per episode. At 60 episodes, one standard error near 75% is about 5.6 points, so the rule settles which adapter goes on, not that rank 32 is truly better. Rank 128 took 1 hour 52 minutes on four GPUs, at 16 seconds a step against 26 on two.

SFT2 starts from that file. Its log says so, and its first loss was 0.67, where S1 ended at 0.65 to 0.70 and a fresh adapter starts at 3.2.

## SFT2 on validation ideas

| Steps | 7 | 14 | 21 | 28 | 35 | 42 |
|---|---|---|---|---|---|---|
| Passed | 48.3% | 53.3% | 55.0% | 65.0% | 71.7% | 66.7% |
| Answered | 85.0% | 76.7% | 90.0% | 85.0% | 95.0% | 90.0% |
| Reward, arm B | 0.355 | 0.413 | 0.421 | 0.477 | 0.546 | 0.494 |
| IC among passes | +0.0073 | +0.0106 | +0.0099 | +0.0072 | +0.0096 | +0.0078 |

The rule gives step 35, against S1's 75.0%, 0.559 and +0.0082. On val, then, SFT2 hasn't clearly added anything yet: the differences are within what 60 episodes can tell apart, and H3 is decided on held-out ideas.

The dip after the start is worth a note. At step 7, 14 of 60 episodes handed in the wrong sign. The student would write "the factor should be low for wide ranges", test the formula without its `-1 *`, read the negative IC as "the idea is wrong on this history", and hand it in still without the sign. The new optimizer is the likely cause, since its first steps are large whatever the gradient, and the dip was gone by step 35, but this wasn't tested.

Paired over the 15 val ideas, S2 minus S1: pass -3.3 points (95% CI -15.0 to +8.3), reward -0.013 (-0.101 to +0.075), IC among passes +0.0003 (-0.0013 to +0.0022), faithful share +2.2 points (-4.4 to +8.9). So S2 has shown no gain, and no loss either: both picked numbers are the best of six, and the last three checkpoints average 67.2% for S1 and 67.8% for S2. A gain the size picking adds on the teacher's own trajectories, +0.047 reward, sits inside that interval, so val is too small to see it.

Wrong direction is S2's biggest failure, 7 of 17. I guessed it came from flipping the sign to chase a better test IC, and the traces say no: tests change sign in 0 to 3 of 60 episodes at every S1 and S2 checkpoint read, and in these seven the sign is wrong from the first test and stays wrong. In three the wrong sign tests well, near +0.014, since history runs against the idea, and the verifier is right to fail them. In two the student saw a negative IC and handed the formula in anyway. It isn't new in S2 either: the last three checkpoints have 8, 2 and 6 such failures in S1, and 5, 7 and 8 in S2.

## Arm C's offline data

The first build died on one formula. On a hard case, the judge's thinking used up all 4,000 tokens and the reply was empty, and at temperature 0 the same limit fails the same way each time. The limit now doubles with each try, up to 16,000. A formula that still gets no verdict is left out of its group, or of the comparison in `judge_eval.py`, instead of stopping everything.

The judge found 1,398 of the 1,449 passing strategy trajectories faithful, 96.5%, and none went without a verdict. Arm C has the same 1,617 rows as arm B, with different weights on 325, and 40 turned from positive to negative.

## The judge on the student, validation ideas

The judge read every passing formula from the chosen S1 and S2 adapters, and none went without a verdict. It found 80.0% of S1's 45 faithful and 79.1% of S2's 43, against about 96% for the teacher, so arm C's reward does differ from arm B's on the student's own work. The rejections read as fair. Most of them answer `volume_level_down` with a ratio or a rank of volume instead of its level, or `downside_vol_down` with absolute returns over all days instead of the down days.

## Offline RL, arm A

| Steps | 22 | 44 | 66 | 88 | 110 | 132 |
|---|---|---|---|---|---|---|
| Passed | 65.0% | 46.7% | 46.7% | 41.7% | 0.0% | 1.7% |
| Answered | 90.0% | 85.0% | 95.0% | 93.3% | 33.3% | 80.0% |
| IC among passes | +0.0065 | +0.0076 | +0.0060 | +0.0081 | none | -0.0037 |
| Mean grad norm, last 11 steps | 0.19 | 0.27 | 0.40 | 0.42 | 1.30 | 2.38 |

Training broke the model. By step 110 it wrote paragraphs with words like "first-bittert" and "level level level", and almost never made a working tool call. The scorer didn't cause this: the six checkpoints have 60 episodes each with none failed, and recounting them by hand gives the same curve.

The likely cause is the negative weights. A failing trajectory is trained with a negative weight, so its loss falls without limit as the model makes those tokens less likely, and failing and passing trajectories share nearly all their tokens: the same syntax, the same tool format. The gradient norm rises slowly until about step 88 and then fast, and the lowest batch loss reaches -2.35. Parse errors were the first symptom, up from 2 at step 22 to 7 at step 44. This reading comes from the curves and the text; no run with the negative weights removed has tested it.

The rule gives step 22, the first checkpoint, at 65.0% passed, reward 0.472 and IC among passes +0.0065, against S2's 71.7%, 0.546 and +0.0096. That is 6.7 points down, which is inside the 10-point stop rule, so arm A's S4 starts from this adapter. It is also about one standard error at 60 episodes, so step 22 hasn't been shown to do harm. It hasn't been shown to help either: arm A's offline stage added nothing visible on val, and the later checkpoints show what more of it does.

## Offline RL, arm B

| Steps | 56 | 112 | 168 | 224 | 280 | 336 |
|---|---|---|---|---|---|---|
| Passed | 23.3% | 45.0% | 18.3% | 36.7% | 55.0% | 0.0% |
| Answered | 50.0% | 91.7% | 95.0% | 78.3% | 93.3% | 11.7% |
| Reward, arm B | 0.187 | 0.338 | 0.124 | 0.280 | 0.397 | 0.000 |
| IC among passes | +0.0129 | +0.0086 | +0.0021 | +0.0097 | +0.0062 | none |

Each checkpoint has 60 episodes with none failed, and recounting them gives the driver's numbers. The rule gives step 280, at 55.0% passed against S2's 71.7%. That is 16.7 points down, over the stop rule, so arm B's S4 starts from S2.

This curve is worse than arm A's because it jumps around: swings of 30 points are far beyond what 60 episodes can do by chance, and the gradient norm stayed between 0.1 and 0.2 until the last 56 steps. Step 56 has the clearest fault. 25 of 60 episodes have one assistant turn, ending on `test_factor(expression=-Factor('quote_volume')...)`. The quotes round the argument are gone, the parser doesn't read that as a call, and the episode ends. Other checkpoints invent methods, such as `.sum` at step 168, or lose a closing bracket.

My first guess was a net negative weight on syntax, if failing trajectories were longer. The data says no: on `test_factor`, `Factor(`, `<factor>` and the rest, net weight is at most 6.5% of the gross in any arm, and failing and passing rows are nearly the same length.

What fits better is that nothing holds the skill in place. Weights sum to zero in each group, so on tokens every row shares, the pull and the push cancel on average. In a batch of about 14 rows they don't cancel, and Adam scales what's left up to a nearly full step whatever its size. So the model wanders on exactly the syntax it was taught. The teacher's rows contain no dropped quotes, so nothing corrects it once it drifts. Online RL should differ here, since a student who drops a quote produces a rollout that is sampled and punished, but that is an argument and not a measurement. This reading hasn't been tested on its own, apart from the run below.

## Offline RL, arm C

| Steps | 56 | 112 | 168 | 224 | 280 | 336 |
|---|---|---|---|---|---|---|
| Passed | 38.3% | 40.0% | 36.7% | 21.7% | 36.7% | 0.0% |
| Answered | 83.3% | 93.3% | 98.3% | 96.7% | 73.3% | 33.3% |
| Reward, arm C | 0.249 | 0.242 | 0.205 | 0.117 | 0.242 | 0.000 |
| IC among passes | +0.0099 | +0.0095 | +0.0078 | +0.0104 | +0.0061 | none |
| Judged faithful, of passes | 19 of 23 | 19 of 24 | 17 of 22 | 9 of 13 | 20 of 22 | none |

Each checkpoint has 60 episodes with none failed, and recomputing arm C's reward by hand gives the driver's numbers. The rule gives step 56, at 38.3% passed against S2's 71.7%. That is 33.4 points down, far over the stop rule, so arm C's S4 starts from S2. It is the worst of the three arms, and it broke the same way: at step 56, 10 episodes have no factor tag, 8 fail on an unmatched bracket, 9 don't respond to the idea and 7 move against it. The kept adapters of all three arms equal the checkpoints they were copied from, byte for byte.

## Testing the fix, with the rule set first

While S3 arm C trains, GPUs 0 and 1 are free and nothing is frozen, so I'm running arm B once more with the same data, the same start from S2, the same learning rate and the same checkpoint rule, with the one change that rows with a negative weight are left out. That is 920 of the 1,617 rows, and their weights are unchanged. Nothing can push a shared token down any more, so if the reading above is right the skill should hold.

The rule, written before any of its results: if the checkpoint this version's rule picks has at least 61.7% passed on val, S2's 71.7% less the 10-point stop rule, S3 is replaced by this version in all three arms, before the freeze. Otherwise the first S3 stands, as the rules already applied it. Either way both versions stay on disk and are reported. Whether S3 gives better formulas stays H4's question, on held-out ideas.

The online stage still rewards and punishes the student's own failures with negative weights. This change touches only the teacher's rows, which the student never wrote.

The result, arm B with only its positive rows:

| Steps | 32 | 64 | 96 | 128 | 160 | 192 |
|---|---|---|---|---|---|---|
| Passed | 65.0% | 73.3% | 65.0% | 70.0% | 70.0% | 61.7% |
| Answered | 95.0% | 91.7% | 95.0% | 93.3% | 95.0% | 85.0% |
| Reward, arm B | 0.469 | 0.541 | 0.483 | 0.519 | 0.528 | 0.452 |
| IC among passes | +0.0062 | +0.0075 | +0.0080 | +0.0078 | +0.0089 | +0.0071 |
| Mean grad norm, last 32 steps | 0.054 | 0.053 | 0.051 | 0.052 | 0.053 | 0.063 |

Each checkpoint has 60 episodes with none failed, and recounting them gives these numbers. The rule gives step 64, at 73.3% passed, over the 61.7% set beforehand, so S3 is replaced by this version in all three arms.

The one change removed the collapse. The same data, start and rate gave 23.3%, 45.0%, 18.3%, 36.7%, 55.0% and 0.0% with the negative rows in, and the gradient norm that rose to 0.650 now stays flat. That supports the reading above, since nothing else differs, though the two versions also run different step counts for an epoch.

It says nothing yet about better formulas. At step 64 the reward is 0.541 against S2's 0.546 and the IC among passes +0.0075 against +0.0096, both within what 60 episodes can do by chance, so on val this S3 has done no harm and shown no gain. The last checkpoint's 61.7% may also be the start of a slower drift; one point can't tell that from noise. H4 asks the real question on held-out ideas.

Arm A on its positive rows:

| Steps | 16 | 32 | 48 | 64 | 80 | 96 |
|---|---|---|---|---|---|---|
| Passed | 68.3% | 63.3% | 63.3% | 65.0% | 65.0% | 65.0% |
| Answered | 98.3% | 95.0% | 90.0% | 93.3% | 95.0% | 90.0% |
| Reward, arm B's formula | 0.509 | 0.463 | 0.478 | 0.478 | 0.475 | 0.489 |
| IC among passes | +0.0082 | +0.0069 | +0.0090 | +0.0073 | +0.0069 | +0.0088 |

Each checkpoint has 60 episodes with none failed, and recounting them gives the driver's numbers. The rule gives step 16, at 68.3% passed, 3.4 points below S2 and inside the stop rule, so arm A's S4 starts from this adapter. The first version of this arm fell to 0.0% at step 110. This one stays near 65% for all 96 steps, with the gradient norm flat near 0.1. Its failures at step 16 are mostly about direction, 13 moving against the idea and 5 not responding to it, and only 1 lost its factor tag, so the syntax has held.

Arm C on its positive rows:

| Steps | 34 | 68 | 102 | 136 | 170 | 204 |
|---|---|---|---|---|---|---|
| Passed | 66.7% | 61.7% | 63.3% | 63.3% | 51.7% | 60.0% |
| Answered | 91.7% | 93.3% | 98.3% | 88.3% | 91.7% | 91.7% |
| Reward, arm C | 0.419 | 0.376 | 0.403 | 0.400 | 0.354 | 0.415 |
| IC among passes | +0.0074 | +0.0084 | +0.0073 | +0.0067 | +0.0113 | +0.0068 |
| Judged faithful, of passes | 34 of 40 | 30 of 37 | 33 of 38 | 33 of 38 | 27 of 31 | 34 of 36 |

Each checkpoint has 60 episodes with none failed and no missing verdict, and recomputing arm C's reward by hand gives the driver's numbers. The rule gives step 34, at 66.7% passed, 5.0 points below S2 and inside the stop rule, so arm C's S4 starts from this adapter. The first version of this arm passed 38.3% at its pick. The gradient norm stays near 0.06 for all 204 steps. Step 170's dip to 51.7% recovers at step 204, so it is probably noise, though 60 episodes can't settle that.

The rules picked the first checkpoint for arms A and C, half an epoch at lr 3e-5, so both have moved little from S2. Any difference between the arms will have to come mostly from online RL. In all three arms the kept adapter equals its checkpoint, byte for byte.

## Held-out ideas, after the freeze

Prereg 03 froze at b898b72, with no held-out result on disk. Each model then answered the 31 held-out ideas 4 times, 124 episodes with none failed, and recounting gives the driver's numbers.

| Model | Passed | Answered | Reward, arm B | IC among passes | Judged faithful |
|---|---|---|---|---|---|
| OLMo untrained, tools | 0.0% | 33.9% | 0.000 | none | |
| Qwen, tools | 25.8% | 88.7% | 0.172 | +0.0014 | |
| DeepSeek, tools | 79.0% | 83.9% | 0.551 | +0.0041 | |
| S1 | 51.6% | 91.9% | 0.341 | +0.0009 | 48 of 64 |
| S2 | 44.4% | 79.8% | 0.305 | +0.0032 | 42 of 55 |
| S3 arm A | 55.6% | 93.5% | 0.355 | -0.0011 | 49 of 69 |
| S3 arm B | 59.7% | 96.0% | 0.395 | +0.0010 | 58 of 74 |
| S3 arm C | 52.4% | 97.6% | 0.353 | +0.0021 | 54 of 65 |

**H1 passes.** S1 minus S0, pass rate +51.6 points (95% CI +41.9 to +61.3), reward +0.341 (+0.275 to +0.409). By family S1 passes 21% on `illiquidity`, 45% on `market_relative`, 56% on `trend_position`, 65% on `trade_count` and 68% on `taker_flow`.

**H3 does not pass.** S2 minus S1, IC among passes +0.0017 (-0.0010 to +0.0044) over 25 ideas. Pass rate -7.3 points (-17.7 to +2.4) and reward -0.036 (-0.107 to +0.029). S2 hands in less often, 79.8% against 91.9%, and tests more, 4.31 against 3.82, so more of its episodes end with no answer.

**H4 does not pass, and points the other way.** S3 arm B minus S2, IC among passes -0.0038 (-0.0063 to -0.0014) over 24 ideas. It passes more, +15.3 points (+5.7 to +24.2), and earns more reward, +0.090 (+0.026 to +0.149). That is the case the prereg warned of, more reward with no better formula. Arms A and C are lower than S2 on IC among passes too, -0.0048 (-0.0075 to -0.0024) and -0.0024 (-0.0046 to -0.0004). S2's own IC may be flattered, since it passes the fewest; that reading isn't tested.

Val flattered every stage. Pass rates fall from 75.0% to 51.6% for S1, and from 73.3% to 59.7% for S3 arm B. IC among passes falls from about +0.008 to about +0.001. Part of that is the families and not the student: DeepSeek's own IC falls from +0.0093 on val to +0.0041 here. Against DeepSeek on held-out ideas S1 is clearly behind, pass rate -27.4 points (-41.1 to -13.7), where val showed no difference. So SFT taught the task on families it never saw, about two thirds as well as the teacher does it. It didn't teach signal: on these ideas the student's formulas add little beyond the five known styles.

Why arm B could pass more without earning more IC: its offline weights mostly say whether a trajectory passed. A failure earns 0 and a pass at least 0.3, while 0.01 more IC adds only 0.117. Of the squared weight in `s3-ic`, 90.1% is explained by pass or fail alone. Of the positive weight, 80.9% comes from the 76 of 200 groups that have a failure, and in those groups 144 of the 420 rows with a positive weight have an IC at or below 0, copies of known styles among them. So this S3 mostly teaches passing. That is a reading of the data, and nothing has tested it. Online RL has the same shape while many rollouts fail, which may leave H6 little room.

Online RL differs from the first S3 in a way this log hadn't noted. S3 uses the SFT trainer: cross-entropy times the weight, with no limit on how far a token's probability can go. prime-rl's online loss, `default_loss_fn`, masks tokens whose probability has already moved too far from what the rollout saw, and adds a squared log-ratio penalty that pulls back toward it. So a negative advantage there can't run away the way a negative weight did here. That comes from reading the code; online RL on this task hasn't run yet.

## Online RL, the first seeds

Both runs started from their arm's S3, as their logs say, and each kept adapter equals its step 35 byte for byte. Neither guard tripped: the highest mismatch KL was 0.0038 against a limit of 0.01, entropy stayed between 0.71 and 1.07, and no rollout had an error, so the judge had no outage in arm C's run. Each took about 1.9 hours, since steps get faster as the student tests less.

| Run | Train pass, steps 1-5 to 31-35 | Val passed | Answered | Reward, arm B | IC among passes | Tests | Judged faithful |
|---|---|---|---|---|---|---|---|
| S3 arm B | | 73.3% | 91.7% | 0.541 | +0.0075 | 4.22 | |
| S4 arm B, seed 1 | 52.7% to 69.5% | 76.7% | 100.0% | 0.562 | +0.0072 | 2.68 | |
| S3 arm C | | 66.7% | 91.7% | 0.491 | +0.0074 | 4.07 | 34 of 40 |
| S4 arm C, seed 1 | 54.8% to 73.3% | 81.7% | 100.0% | 0.585 | +0.0057 | 2.90 | 44 of 49 |
| S3 arm A | | 68.3% | 98.3% | 0.509 | +0.0082 | 3.77 | |
| S4 arm A, seed 1 | 52.5% to 64.4% | 81.7% | 100.0% | 0.586 | +0.0058 | 2.73 | |

Arm A's run came third and was checked the same way, with a highest KL of 0.0054. It rose the slowest on training ideas and still matched arm C on val, so the training curves are a poor guide to who is ahead. At this point the reward with IC in it has not beaten the one that only checks the gates on val.

Negative advantages did not break the syntax here, as they did in S3's first version, which fits the reading above but doesn't prove it. So far RL has made the student more reliable: it always hands in, and it passes more often. It hasn't raised IC among passes on val. It also tests less, 2.7 to 2.9 times against 4.1 to 4.2, which nothing in the reward asks for directly. A failed answer earns 0, so fewer tests may just mean fewer ways to run out of turns; I haven't checked that. These are single seeds on 60 val episodes, and the held-out evals come after all six runs.

## A second guard for online RL

The guard from prereg 02 watches entropy and mismatch KL, and neither is about whether the student can still do the task. So online RL now also stops if the mean reward over any five steps running is more than 30% below the mean over steps 1 to 3. At S2's level that is about 72% passed falling to 50%. As with the first guard, the run then starts again once at lr 1e-5, and a second trip stops the whole flow with the GPUs free. This was added before any online result on this task.

## Online RL, all six runs

All six ran their 35 steps and no guard tripped, so nothing ran at the lower rate. The highest mismatch KL was 0.0054 against a limit of 0.01, entropy never fell below 0.69, and no rollout had an error. Every step of both arm C runs trained all 128 rollouts, so the judge had no outage, and the pause rule, which I hadn't automated, never came up. Each kept adapter equals its step 35 byte for byte. Each run took 1.8 to 1.9 hours.

| Run | Train pass, steps 1-5 to 31-35 | Val passed | Answered | Reward, arm B | IC among passes | Tests | Judged faithful |
|---|---|---|---|---|---|---|---|
| S4 arm A, seed 1 | 52.5% to 64.4% | 81.7% | 100.0% | 0.586 | +0.0058 | 2.73 | |
| S4 arm A, seed 2 | 52.0% to 65.5% | 70.0% | 96.7% | 0.490 | +0.0043 | 2.62 | |
| S4 arm B, seed 1 | 52.7% to 69.5% | 76.7% | 100.0% | 0.562 | +0.0072 | 2.68 | |
| S4 arm B, seed 2 | 53.1% to 67.2% | 81.7% | 100.0% | 0.582 | +0.0054 | 2.68 | |
| S4 arm C, seed 1 | 54.8% to 73.3% | 81.7% | 100.0% | 0.585 | +0.0057 | 2.90 | 44 of 49 |
| S4 arm C, seed 2 | 54.7% to 73.4% | 76.7% | 100.0% | 0.554 | +0.0063 | 2.57 | 41 of 46 |

Two seeds of arm A differ by 11.7 points on val, as much as anything between arms, so val can't order the arms.

## Online RL on held-out ideas

Each seed answered the 31 held-out ideas 4 times, 124 episodes with none failed, pooled to 248 per arm as the prereg says. Recounting the rows gives the driver's numbers. `heldout_online.py`, which ran this, was written after the freeze and calls the pinned `compare_stages.py`.

| Model | Passed | Answered | Reward, arm B | IC among passes | Raw IC | Tests | Judged faithful |
|---|---|---|---|---|---|---|---|
| S1 | 51.6% | 91.9% | 0.341 | +0.0009 | +0.0023 | 3.82 | 48 of 64 |
| S2 | 44.4% | 79.8% | 0.305 | +0.0032 | +0.0055 | 4.31 | 42 of 55 |
| S3 arm B | 59.7% | 96.0% | 0.395 | +0.0010 | -0.0005 | 4.00 | 58 of 74 |
| S4 arm A | 71.4% | 99.6% | 0.471 | +0.0008 | +0.0016 | 2.76 | 123 of 177 |
| S4 arm B | 68.5% | 98.4% | 0.453 | +0.0009 | +0.0029 | 2.63 | 120 of 170 |
| S4 arm C | 65.7% | 99.6% | 0.427 | -0.0000 | -0.0015 | 2.69 | 125 of 163 |
| DeepSeek, tools | 79.0% | 83.9% | 0.551 | +0.0041 | +0.0088 | | |

The seeds pass 72.6% and 70.2% in arm A, 66.9% and 70.2% in arm B, 69.4% and 62.1% in arm C.

**H5 does not pass.** S4 arm B minus S3 arm B, IC among passes -0.0000 (95% CI -0.0017 to +0.0019) over 27 ideas. Pass rate +8.9 points (-2.0 to +20.2), reward +0.058 (-0.012 to +0.132).

**H6 does not pass.** S4 arm B minus S4 arm A, IC among passes -0.0004 (-0.0019 to +0.0011) over 29 ideas. On the ideas outside mirrored pairs it is +0.0008 (-0.0009 to +0.0030) over 14 ideas. This reward taught no more signal than rewarding the gates alone. That turned out to be a weak test, as "After the exam" explains: IC is 3% of this reward's spread within a group.

RL did make the student more reliable. Arm A passes 15.7 points more than its S3 (+8.1 to +23.8) and arm C 13.3 more (+5.2 to +21.0). Arm B passes 16.9 more than S1 (+7.3 to +26.6) and 24.2 more than S2 (+14.1 to +33.9), with reward +0.112 (+0.045 to +0.180) and +0.148 (+0.081 to +0.215). The price against S2 is IC among passes, -0.0027 (-0.0047 to -0.0008).

It passes more often without reaching more ideas. The share of ideas with at least one pass in 4 rollouts is 87.1% to 96.8% per S4 seed, against 90.3% for S1 and 87.1% for S2. So RL turned ideas the student could sometimes do into ideas it usually does.

Against DeepSeek, arm B's pass rate is -10.5 points (-25.0 to +3.6), where S3 arm B was -19.4 with an interval below 0. Reward is still -0.099 (-0.198 to -0.003) and IC among passes -0.0031 (-0.0051 to -0.0010). By family, arm B against the teacher: 0.35 / 0.79 on `illiquidity`, 0.59 / 0.75 on `trend_position`, 0.70 / 0.80 on `market_relative`, 0.90 / 0.90 on `trade_count`, 0.91 / 0.75 on `taker_flow`. So the student now passes more than the teacher on one family, and its largest gap is illiquidity.

## The audit

Passing held-out formulas of S2 and the three S4 arms, up to 60 per model evenly across ideas, drawn with a fixed seed by `audit.py`. S2 has 55, and all were read. A pair of idea and formula handed in by more than one model was read once, which leaves 163 items from 235 formulas. I read them shuffled, with the idea and the formula only, by the judge's rubric, and wrote the verdicts and my rulings on what the rubric leaves open (`results/factor/audit/reading_rules.md`) before opening the key. `hashes_before_key.txt` holds their hashes. I wrote the verifier and the rubric, and a model's style may have given something away. The owner of the research system hasn't read the sample.

| Model | Reader: faithful | Judge: faithful | Reader and judge agree |
|---|---|---|---|
| S2 | 44 of 55, 80.0% | 42 of 55 | 49 of 55 |
| S4 arm A | 47 of 60, 78.3% | 45 of 60 | 48 of 60 |
| S4 arm B | 42 of 60, 70.0% | 43 of 60 | 49 of 60 |
| S4 arm C | 48 of 60, 80.0% | 44 of 60 | 52 of 60 |

**H7 passes exactly on the line, and it isn't robust.** S4 arm B is 10.0 points below S2, and the rule allows no more than 10. One formula fewer would fail it, and the interval beside it runs from -25.6 to +5.6. The first run of the scorer printed "fails": in floats 42/60 minus 44/55 falls just under -0.1. It also counted no borderline items, since a loop variable overwrote them. I fixed both after seeing that output, and the verdict file didn't change.

**H9 does not pass.** S4 arm C minus arm B, +10.0 points (-5.0 to +25.0). Its cost in IC among passes is -0.0001 (-0.0019 to +0.0016).

Other readings of the same verdicts, since 54 items were marked borderline before the key was opened:

| Reading | H7, arm B minus S2 | H9, arm C minus arm B |
|---|---|---|
| As read | -10.0, passes | +10.0 (-5.0 to +25.0) |
| All 54 borderline items not faithful | -0.5, passes | +8.3 (-8.3 to +25.0) |
| All 54 borderline items faithful | -1.2, passes | +1.7 (-6.7 to +10.0) |
| Only volatility as beta not faithful | -9.8, passes | +5.0 (-11.7 to +21.7) |
| Only a level for an idea about a change faithful | +3.2, passes | +0.0 (-13.3 to +13.3) |
| Only distance from a high for its recency faithful | -8.3, passes | +11.7 (-3.3 to +26.7) |
| Only a formula against own past for an idea about a level faithful | -17.7 (-31.1 to -4.5), fails | +10.0 (-5.0 to +25.0) |

H9 fails every way. H7 depends on one ruling, that a formula against a coin's own past doesn't answer an idea about a standing level. S2 hands in many of those, 8 of its 11 rejections, and the rubric's "rank" and "ratio" could be read to allow them. One more thing to weigh: the pilot reading earlier in this log called plain volatility "not beta", and this audit accepts it as a proxy, marked borderline. The fourth row shows that doesn't change the result.

The models fail differently. 9 of arm B's 18 rejections and 7 of arm A's 13 are a plain level for an idea about a change, such as the taker share for "suddenly rises above normal". Arm C has 3 of 12 of those, and S2 1 of 11. So RL's student tends to hand in the simple form, which fits it testing less, though I haven't tested that link.

Per idea over the ideas both have, as exploration and not the prereg's bootstrap over formulas: arm C minus arm B +15.5 points (+4.6 to +28.2), arm B minus arm A -12.6 (-22.4 to -4.6), arm B minus S2 -4.9 (-21.6 to +11.7). That hints that a reward with IC in it costs some faithfulness and a judge wins it back. It doesn't settle H9.

The reader and the judge agree on 198 of 235 formulas. On 29 items they differ. The reader is stricter on a level for an idea about a change, 8 items, which the rubric calls fine, and on a formula against own past for an idea about a level, 4. The judge is stricter on a plain return for an idea about the market, 6, on volatility for beta, 6, and on odd forms, 5. IC ranks coins within a day, so a return common to all coins doesn't change the order, which is why I accepted the plain return. So the judge isn't simply right or wrong here: most differences are about how far a proxy may go.

## The sealed exam

`sealed_exam.py` scored all 8 answer files on 2025 first and required each to give back its own verdicts and ICs. All did, with a largest IC change of 0. It then opened the sealed period once per file, at 18:55 UTC on 2026-11-06, on S1, both seeds of each arm and the held-out rows of DeepSeek's file. `sealed_access.log` has those 8 lines and no others. The answers are the same rows and no verdict changed, so pass rates stay as they were. The audit's verdicts were locked before this, and the files the exam relies on match their pins. `panel.py` isn't among the pins, and it equals the freeze commit.

Mean IC among passes per idea with a 95% CI over ideas, which is the prereg's measure, so it differs a little from the per-episode means above:

| Model | Passed | IC, 2025 | IC, 2026 | Raw IC, 2026 |
|---|---|---|---|---|
| S1 | 51.6% | +0.0015 | +0.0025 (-0.0006 to +0.0060) | +0.0029 (-0.0056 to +0.0109) |
| S4 arm A | 71.4% | +0.0012 | +0.0009 (-0.0010 to +0.0026) | +0.0008 (-0.0069 to +0.0081) |
| S4 arm B | 68.5% | +0.0008 | +0.0012 (-0.0008 to +0.0032) | +0.0019 (-0.0045 to +0.0085) |
| S4 arm C | 65.7% | +0.0004 | +0.0014 (-0.0009 to +0.0037) | +0.0019 (-0.0058 to +0.0097) |
| DeepSeek, tools | 79.0% | +0.0042 | +0.0050 (+0.0025 to +0.0075) | +0.0084 (+0.0026 to +0.0143) |

**H8 does not pass.** S4 arm B minus S1, IC among passes -0.0011 (-0.0034 to +0.0011) over 27 ideas. Arm B's own mean is +0.0012, with an interval across 0.

Arm B against arm A is +0.0004 (-0.0013 to +0.0022), and arm C against arm B +0.0002 (-0.0019 to +0.0025), so no reward separates from another on the clean year either. Arm B against DeepSeek is -0.0038 (-0.0068 to -0.0009).

The teacher's result is the one that gives this exam its meaning. On a year nothing here was chosen on, its formulas keep an incremental IC with an interval above 0. So these ideas hold signal and the measure can see it, and the student's flat numbers mean it didn't learn to find it. The student's numbers barely moved from 2025 to 2026, with arm A down 0.0003 and the others up, so RL shows no sign of overfitting its reward year, though there was little to lose.

Reward does hold up, since passing doesn't depend on the year: S4 arm B minus S1 on the 2026 reward, +0.108 (+0.037 to +0.177).

## Why online RL tests less

Exploration, on the rollouts RL trained on and on the 2025 scores.

Reward doesn't count tests up to the limit of five, yet they fell in every run, from 3.5 to 3.9 over the first three steps to 2.6 to 2.9 over the last five. The likely reason is that using all five goes badly. Over steps 1 to 5, across the six runs, an episode with 5 tests hands in 75% of the time, against 97% to 99% with 3 or 4. It passes 29% to 41% of the time, against 53% to 69%. Within a group, on one idea, its advantage under arm B's reward is -0.06 to -0.11, and theirs is positive. So its share falls from 15% to 30% of episodes to 0% to 2%. A better formula found on a fifth test earns little under the reward, and failing to hand in costs everything.

Testing may be where the IC was coming from. Among passes, within one idea and one model, each extra test goes with this much more IC, with whole ideas resampled:

| Models | IC per extra test |
|---|---|
| DeepSeek | +0.0011 (+0.0005 to +0.0023) |
| S1, S2, S3, pooled | +0.0019 (+0.0010 to +0.0029) |
| S4, pooled | +0.0001 (-0.0003 to +0.0005) |

So after online RL the student tests less, and its tests no longer go with better formulas. This is correlation among formulas that passed, and a formula tested more isn't randomly assigned, so it suggests a cause and doesn't show one. A student who stops early and hands in a plain form fits both this and the audit. It isn't the main reason IC is low, though: IC had already fallen in offline RL with tests near 4, and the fall from 4.00 to 2.63 tests left IC flat. "After the exam" has the numbers.

This also settles a loose end. An idea came up as few as 8 times in a run where the median was 64. `advantage_range` in `configs/factor-idea/rl.toml` rejects a group whose advantage is exactly 0, as when all 8 rollouts earn the same. `rise_without_buyers_down`, whose honest formulas the verifier fails, is the usual case, so RL rarely trains on it, and an idea the student can't pass at all gets no lift from RL.

SFT2's dataset is also settled. It was rebuilt at 01:43 on Nov 6, before training at 05:00, and it matches its pin.

## Where the hypotheses stand

| ID | Result | Number |
|---|---|---|
| H1, primary | Passes | Pass rate +51.6 points (+41.9 to +61.3) |
| H2 | Passes | Pass rate +71.7 points (+58.3 to +83.3), on val, read as an upper bound |
| H3 | Does not pass | IC among passes +0.0017 (-0.0010 to +0.0044) |
| H4 | Does not pass, opposite way | -0.0038 (-0.0063 to -0.0014) |
| H5 | Does not pass | -0.0000 (-0.0017 to +0.0019) |
| H6 | Does not pass | -0.0004 (-0.0019 to +0.0011) |
| H7 | Passes on the line, depends on one ruling | -10.0 points, limit 10 |
| H8 | Does not pass | -0.0011 (-0.0034 to +0.0011) |
| H9 | Does not pass | +10.0 points (-5.0 to +25.0) |

SFT taught OLMo 3 7B the task on families it never saw, from 0% to 51.6%. RL made it reliable, to about 70%, against the teacher's 79%, and that holds on 2026. No hypothesis on better formulas passed, on 2025 or on the clean year, and the teacher shows there was more to find.

## After the exam

Everything here came after the sealed exam, so it is exploration, and I read the 2026 numbers below knowing prereg 03's results. No panel was reopened: `sealed_access.log` still has its 8 lines, and 2026 ICs come from the exam's own files. The prereg listed research IC against later IC, and variety of formulas, as exploratory; the rest wasn't foreseen. `scripts/factor/explore_after_exam.py` prints all of it, with its output in `results/factor/explore/`.

**The reward could hardly say anything about IC.** Within a group, 97% of arm B's reward spread comes from passing or failing and 2.8% from IC among passes, at the start of a run and at its end. The reward maps IC onto a range of ±0.03, and passes in a group typically differ by about 0.003, which is 0.04 of reward. That is my own fault in scaling it. IC among passes on training ideas didn't rise either, from +0.0023 to +0.0019 in arm B, a change of -0.0004 (-0.0010 to +0.0000), and arm B's change minus arm A's is -0.0001 (-0.0006 to +0.0005). So H6 shows that this reward taught no more signal than the gates did. It says little about whether a reward can.

**Research IC says a lot about later IC.** Among one model's passing formulas for one idea, research IC correlates +0.63 to +0.91 with 2025 and +0.69 to +0.84 with 2026, so what the student reads in a test is a good guide.

**S2's formulas can't be told from the teacher's, and the fall comes with offline RL.** On held-out ideas, per idea:

| Model | Passed | No answer | Wrong direction | Doesn't respond | IC among passes | Minus the teacher | Tests |
|---|---|---|---|---|---|---|---|
| S1 | 52% | 8% | 17% | 21% | +0.0015 | -0.0017 (-0.0046 to +0.0013) | 3.82 |
| S2 | 44% | 20% | 16% | 17% | +0.0034 | -0.0001 (-0.0023 to +0.0022) | 4.31 |
| S3 arm B | 60% | 4% | 14% | 19% | +0.0010 | -0.0033 (-0.0059 to -0.0010) | 4.00 |
| S4 arm B | 69% | 2% | 6% | 17% | +0.0008 | -0.0031 (-0.0051 to -0.0010) | 2.63 |
| DeepSeek | 79% | 16% | 3% | 0% | +0.0042 | | 4.47 |

Val ideas say the same: S2 minus the teacher +0.0001 (-0.0018 to +0.0025), S4 arm B -0.0020 (-0.0039 to -0.0003). The gap isn't about transfer, since on training ideas arm B sits -0.0015 (-0.0024 to -0.0006) below the teacher as RL starts and -0.0020 (-0.0030 to -0.0011) as it ends. These intervals are wide, so S2 isn't shown to equal the teacher, and S2's IC may be flattered because it passes the fewest. With the picking described below, though, S2 passes 83.9% with IC +0.0051 and S4 arm B 83.1% with +0.0016, though S2 spends three times the tests to get there.

This corrects what I wrote earlier in this log. Fewer tests aren't the main reason IC is low: IC fell in offline RL with tests near 4, and online RL's fall in tests left it flat. Offline RL may have done it by mixing ordinary rows into what S2 learned only from the best rows, which nothing has tested.

**Online RL's gain in pass rate is mostly two repairs.** From S2 to S4 arm B, episodes with no answer fall from 20% to 2% and wrong directions from 16% to 6%. "Doesn't respond" stays at 17%, against the teacher's 0%, and most of that is the illiquidity family. Neither repair obviously needs worse formulas.

**The student already writes good formulas, and doesn't pick them.** Take k episodes on an idea and keep one by its research IC. The gates read data through 2024 only, so a real process could run them first. On held-out ideas:

| Process | Tests | Passed | Judged faithful | IC on 2026 | Reward on 2026 |
|---|---|---|---|---|---|
| S4 arm B, one episode | 2.6 | 68.5% | 69% | +0.0012 (-0.0009 to +0.0033) | 0.455 |
| S4 arm B, k=4, gates then research IC | 10.5 | 90.7% | 65% | +0.0040 (+0.0015 to +0.0065) | 0.634 |
| S4 arm B, k=4, gates and judge then research IC | 10.5 | 90.7% | 81% | +0.0029 (+0.0004 to +0.0052) | 0.622 |
| S4 arm B, k=8, gates and judge then research IC | 21.1 | 96.8% | 83% | +0.0044 (+0.0015 to +0.0073) | 0.679 |
| DeepSeek, one episode | 4.5 | 79.0% | | +0.0050 (+0.0025 to +0.0075) | 0.566 |
| DeepSeek, k=2, gates then research IC | 8.9 | 92.5% | | +0.0057 (+0.0028 to +0.0084) | 0.668 |

At about 10 tests each, S4 arm B minus the teacher is -0.0017 (-0.0051 to +0.0016) on IC and -0.033 (-0.121 to +0.046) on reward. At one episode each it is -0.0038 (-0.0068 to -0.0009) and -0.111 (-0.211 to -0.013). The rule has nothing to tune, each seed alone gains about the same, +0.0028 and +0.0030 at k=4, and val ideas agree on 2025. But this is a model and a process, not the model alone, the teacher with the same process stays ahead on point estimates, and it was found after the exam.

The gates can't be skipped. Picking by research IC alone keeps IC and loses passes: arm A at k=8 falls to 54.8%, and arm B on val to 60.0%. Picking also costs some faithfulness, 69% down to 60% at k=8, and the judge wins it back while keeping most of the gain.

Given the same process, S2 has the higher IC and S4 the higher pass rate. At k=4 and k=8 on held-out ideas, gates first, with about 17 and 21 tests, S2 has +0.0051 and S4 arm B +0.0038 on 2025, and they pass 83.9% and 96.8%. On val, S2 at k=4 matches the teacher at k=4: 100% passed both, IC +0.0085 and +0.0089, reward 0.750 and 0.753. Handing in the best test of one episode, with no more tests, helps less: arm B's IC among passes gains +0.0007 (+0.0002 to +0.0013), and an episode with no answer scored on its best test takes S2 only from 44.4% to 50.8% passed.

**A better-scaled reward wouldn't by itself keep the tests.** On rollouts from the first five steps, a percentile reward with no zero for a missing answer halves the penalty for using all five tests, from -0.068 to -0.031 in mean advantage, and 2 or 3 tests stay best. So tests may still fall. It would raise IC's share of the spread: a percentile table with a floor of 0.3 gives 17% to 21%, and with a floor of 0.1 gives 30% to 36%.

**No form of formula pays everywhere.** Within an idea, EMA goes with higher IC for students and not for the teacher, and more fields goes with higher IC for the teacher and is unclear for students. None of the forms I tried is clear in students on training ideas, students on val and the teacher together, and more fields runs against faithfulness. More tests go with higher IC for students on training ideas, +0.0009 a test (+0.0006 to +0.0012). Of the spread of IC among passes on training ideas, 68% is between ideas and 32% within one. So a reward on IC may teach which formula suits a training idea more than a skill that carries over, and search and picking are the most likely things to carry over.

**So the next thing to try is the simplest.** S2's formulas with online RL's two repairs: online RL straight from S2, skipping offline RL. `prereg/03b-pilot.md` sets that out, with a second run on the better-scaled reward and the rules for going on, before either starts.
