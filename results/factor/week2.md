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

## A second guard for online RL

The guard from prereg 02 watches entropy and mismatch KL, and neither is about whether the student can still do the task. So online RL now also stops if the mean reward over any five steps running is more than 30% below the mean over steps 1 to 3. At S2's level that is about 72% passed falling to 50%. As with the first guard, the run then starts again once at lr 1e-5, and a second trip stops the whole flow with the GPUs free. This was added before any online result on this task.
