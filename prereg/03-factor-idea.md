# Pre-registration 03: teaching OLMo 3 7B to turn a one-sentence idea into a factor

**Status: FROZEN on 2026-11-06, before any trained model read a held-out idea.** The commit that adds this text is the freeze, and the sha256 values under Pinned files pin everything listed under Fixed design. Rank and step counts were picked by runs on validation ideas, and the run that won is the stage's model, so the settings and the model come from the same run. Anything decided after that commit is reported as exploratory, and changes are listed under Deviations.

## Question

Prereg 01 and 02 reproduced a post-training recipe on someone else's task. This one builds the task: one simple investment idea in, one factor formula out, on daily Binance perpetual klines, with up to five tests on 2022-07 to 2024-12 before answering. A verifier the model never sees then scores it: gates on rows up to 2024, then incremental IC on 2025.

Untrained, OLMo 3 7B Instruct passes the gates on 0.0% of held-out episodes with the tool and 2.4% without. Qwen3-4B passes 25.8% and 25.0%, DeepSeek 79.0% and 86.3%. The questions:

1. Can SFT teach OLMo the task on families of ideas it never trained on?
2. Does each later stage, following Samaya's Sterling, make the formulas better on those families?
3. Does a reward with a signal in it (incremental IC) teach more signal than a reward that only checks the gates?
4. Gates that test behaviour can be gamed. Does RL on such a reward drift away from what the idea says, and does an LLM judge in the reward stop that, at what cost in IC?

## Fixed design

| Item | Value |
|---|---|
| Base model | allenai/Olmo-3-7B-Instruct @ 6e5971d |
| Data | Binance USDT perpetual daily klines, 9 fields, delisted coins included. Research 2022-07-01 to 2024-12-29, reward 2025-01-01 to 2025-12-29, sealed 2026-01-01 to 2026-10-16. `dev.npz` and `sealed.npz` |
| Universe | On the eve of each month, the top 300 by 30-day mean quote volume among coins listed for at least 260 days |
| Target | fwd[t] = close[t+2] / close[t+1] - 1 |
| Ideas | `task/ideas_v2.jsonl`, 115 ideas in 20 families |
| Splits | train 69 ideas; val 15, one from each training family; heldout 31 in 5 families, `taker_flow`, `trade_count`, `illiquidity`, `trend_position`, `market_relative`. Seeded before any model answered them, and reproduced by `scripts/factor/draw_splits.py` |
| Verifier | `envs/factor_idea/factor_idea/{sandbox,dsl,ops,evaluate}.py`. Exam of 951 items in `results/factor/verifier_exam.json`: false fail 0/173, false pass 8/720, not blind. Two audits and a red team found it weaker than the exam says; see Known limits |
| Reward | Arm A, "gates": 1 if the gates pass. Arm B, "ic": 0.3 + 0.7 x clip((incremental IC + 0.03) / 0.06, 0, 1) if they pass. Arm C, "judged": as arm B if the judge also finds the formula faithful, else 0. All arms: 0 for no answer or more than five tests |
| Judge | `deepseek-flash`, temperature 0, rubric `task/judge_rubric.md`, code `factor_idea/judge.py`. It reads only the idea and the formula. The verifier's author wrote the rubric. The owner of the research system chose not to label formulas for it, so nothing independent checks it; its rates are measured on the red team's, the teacher's and blended formulas, and the audit reads what it passed |
| Teacher | `deepseek-flash` with `task/manual.zh.md` and `scripts/factor/teacher_traces.py` |
| Task code | `factor_idea/{taskset,prompt,tools,workers}.py`, parsers `setup/{renderers_olmo3,vllm_olmo3_keep_text}.py` |
| Trainer | prime-rl v0.9.0 (ab5de8f) with `setup/patch_prime_rl_sft.py`. Stages run by `scripts/factor/run_stages.py` with `configs/factor-idea/` |
| Datasets | `sft/s1`, `s2`, `s3p-gates`, `s3p-ic`, `s3p-judged`, each a `train.jsonl`. The first version of S3, `s3-gates`, `s3-ic`, `s3-judged`, was replaced on val and is kept to show why |
| Adapters | The chosen adapters of S1, S2 and S3 are pinned below. S4's go to `results/factor/adapters.md` when they are chosen |
| LoRA | all linear projections, alpha 32; rank 32, chosen from {8, 32, 128} on val only |
| Lengths | 512 tokens per turn, 8,192 per episode, as in prereg 02 |
| Tool parser | `olmo3_keep_text`, which keeps text before a call |
| Training seeds | SFT and offline RL: 1. Online RL: 2 per arm, which differ in `inference.vllm.seed`, so in sampling only |

## Stages

Each stage starts from the adapter file the stage before it was evaluated on, with a new optimizer and step count (`LORA_POSTTRAIN_INIT_ADAPTER`, tested for SFT: the new run's first broadcast equals the file in all 448 tensors, and its first loss was 1.09 where a fresh adapter starts at 3.37).

| Stage | Data | Weight | What it should teach |
|---|---|---|---|
| S0 | none, the base model | | the baseline |
| S1, SFT1 | Teacher trajectories on train ideas: 6 wordings x 5 rollouts each, that passed the gates as they stood when the data was built, hand in a tested formula, make one call per turn and stay off the teacher's brief; plus recovery trajectories starting from the broken formulas in OLMo's four untrained tool baselines, each distinct formula once per idea, at most 8 per idea, 2 teacher rollouts each, the broken turn not trained on | 1 | the syntax and the research loop |
| S2, SFT2 | 3 wordings of each train idea x 8 strategies; each group's best by research-window incremental IC, read from what the teacher saw | 1 | better research choices, without the reward year |
| S3, offline RL | The trajectories of S2 that beat their group's mean, each weighted by its reward on 2025 minus that mean, once per arm | advantage, positive only | what earns reward, cheaply |
| S4, online RL | OLMo's own rollouts on train ideas, GRPO, group 8 x 16 ideas, 128 rollouts per step, 35 steps, lr 3e-5, once per arm and seed | advantage | what earns reward, from its own mistakes |

S4 starts from S3's adapter trained under the same arm, or from S2's if the arm's S3 hit the stop rule.

S3 was first trained on all its rows, the negative weights included, and that version lost the skill on val in all three arms: the checkpoints the rule picked passed 65.0%, 55.0% and 38.3%, and each arm's last passed 2% or less. The negative rows were then left out of arm B alone, with the test written down before the result: if the checkpoint the rule picks passes at least 61.7%, S2's rate less the 10-point stop rule, S3 is replaced in all three arms. It passed 73.3%. So S3 is now imitation weighted by advantage, and nothing in it pushes a bad answer down. Only S4 does that, on the student's own rollouts. The account is in `results/factor/week2.md`.

Settings chosen on val ideas only, before the freeze:

- **LoRA rank:** 8, 32 or 128. The rank with the highest val pass rate at its best checkpoint. Within 5 points, the smaller one. Unless 32 beats 8 by more than 5 points, 128 isn't run.
- **Learning rate:** S1 and S2, 1e-4 from the pilot, constant, changed only if val shows it failing. S3, 3e-5, as S4 uses, set while S3 still had negative weights and kept for the version that replaced it.
- **Steps for S1:** adapters are written and evaluated on val every half epoch, up to 3 epochs. The checkpoint with the highest val pass rate; within 5 points, the earliest.
- **Steps for S2 and S3:** the same way, by val reward under the stage's arm, arm B for S2, since these stages exist to improve what is handed in, not whether it passes. Within 0.02, the earliest.

`scripts/factor/run_stages.py` applies these rules and logs each choice to `results/factor/selection.jsonl`. Picking among checkpoints makes the winner's val number optimistic. Held-out ideas play no part.

What the rules chose, from 60 val episodes each:

| Stage | Choice | Pass | Reward under its arm | IC among passes |
|---|---|---|---|---|
| S1 | rank 32, step 350 of 420. Best pass rate at rank 8 was 66.7%, at 32 75.0%, at 128 65.0% | 75.0% | 0.559 | +0.0082 |
| S2 | step 35 of 42, lr 1e-4 | 71.7% | 0.546 | +0.0096 |
| S3 arm A | step 16 of 96, lr 3e-5 | 68.3% | 0.683 | +0.0082 |
| S3 arm B | step 64 of 192, lr 3e-5 | 73.3% | 0.541 | +0.0075 |
| S3 arm C | step 34 of 204, lr 3e-5 | 66.7% | 0.419 | +0.0074 |
| First S3 arm A, replaced | step 22 of 132, lr 3e-5 | 65.0% | 0.650 | +0.0065 |
| First S3 arm B, replaced | step 280 of 336, lr 3e-5 | 55.0% | 0.397 | +0.0062 |
| First S3 arm C, replaced | step 56 of 336, lr 3e-5 | 38.3% | 0.249 | +0.0099 |

All three arms of the S3 that stands are within the 10-point stop rule, so each arm's S4 starts from its own S3.

## Evaluation

- **Sets:** `heldout` (31 ideas, primary), `val` (15), `train` (69), original wordings only.
- **Sampling:** 4 rollouts per idea, generation_config defaults (temperature 0.6, top_p 0.95), `configs/factor-idea/eval.toml`. Each model is served on its own under the same scripts. S0 and the other baselines keep their answers from the untrained runs, scored again by the final verifier with `scripts/factor/rescore_eval.py`.
- **When:** after each stage. Held-out ideas are not read until a stage's settings are final.
- **Metrics, per idea and then averaged over ideas:** pass rate; incremental IC among passes; reward under the stage's arm; tests per episode; share of tests that came back as errors; judge-faithful share among passes, for every stage and arm alike (`scripts/factor/judge_eval.py`).
- **Seeds:** an arm's two seeds are one model for every comparison, 8 rollouts per idea. No seed is picked.
- **Failed episodes:** `compare_stages.py` refuses a set with failed episodes or uneven rollouts. They are rerun, or the comparison is reported with `--allow-missing` and says so.
- **Statistics:** 95% CIs from a paired bootstrap over ideas, 10,000 resamples. IC among passes uses the ideas with a pass on both sides and says how many. With five held-out families, results per family are reported beside them.
- **Sealed:** once, at the end, `rescore_eval.py --period sealed` on the held-out answers of S1, of the final model of each arm, and of DeepSeek. A model's formula can't depend on the scored period, since test_factor stops at 2024. It is rehearsed on 2025 first and must reproduce the file's IC. `load_sealed` logs every opening.
- **Comparison:** DeepSeek and Qwen3-4B on the same sets under the same scripts.

## Hypotheses and pass criteria

| ID | Hypothesis | Passes if, on `heldout` unless stated |
|---|---|---|
| H1 (primary) | SFT teaches the task on unseen families | S1 minus S0 pass rate has a 95% CI lower bound above 0 |
| H2 | SFT teaches the same on unseen ideas of seen families | The same on `val`. Read as an upper bound: 10 of 15 val ideas are near mirrors of train ideas, and the checkpoint was picked on val |
| H3 | Choosing by research IC gives better formulas | S2 minus S1, IC among passes, 95% CI lower bound above 0 |
| H4 | Offline RL gives better formulas | S3 arm B minus S2, IC among passes, 95% CI lower bound above 0 |
| H5 | Online RL gives better formulas | S4 arm B minus S3 arm B (S2 if arm B's S3 hit the stop rule), IC among passes, 95% CI lower bound above 0 |
| H6 | A reward with a signal teaches more signal | S4 arm B minus S4 arm A, IC among passes, 95% CI lower bound above 0. Also reported on the 15 held-out ideas outside mirrored pairs |
| H7 | RL doesn't trade validity for score | In the audit, the share judged faithful after S4 arm B is no more than 10 points below S2's |
| H8 | The signal holds out of time | On the sealed period, the final arm B model minus S1, IC among passes, 95% CI lower bound above 0. Its own mean IC is reported beside it |
| H9 | A judge in the reward keeps formulas faithful | In the audit, S4 arm C's share judged faithful minus arm B's, 95% CI lower bound above 0, from a bootstrap over formulas. S4 arm C minus arm B, IC among passes, is reported as its cost |

H3 to H6 read 2025, the year S3 and S4 were rewarded on: the ideas are unseen, the year is not. H8 is the only clean test of the signal. Pass rate and reward are reported beside every IC, and a stage can raise reward by passing more without improving a formula, so reward alone settles nothing here.

All results are reported whether or not they pass. Nine hypotheses are not corrected for multiple testing, so each CI should be read on its own.

## Audit

Passing `heldout` formulas of S2 and of S4 arms A, B and C are read once, together, at the end, shuffled with the model names hidden, and each is judged faithful or not to its idea. Each model gives up to 60, sampled evenly across ideas. If a model has fewer, all of them are read and the count is given. Below 30 for a model, H7 or H9 is reported as undecided. The verifier's author reads them; the owner of the research system reads the same sample if available, and disagreement is reported.

## Known limits, stated in advance

- **The gates can be gamed.** They test behaviour and can't read meaning. A red team that knew the rules got 36 of its 37 answers past the first version, about 30 of them unfaithful by one reader, and 35 of 36 past the fixed one. A blend of an honest formula with a strong signal the idea doesn't mention still passes and earns up to 0.18 more reward. A formula whose key data only matters under the test's large shock passes too: on real data one such condition never fired on any day. These remain in the verifier as frozen. That is what arm C and H7 and H9 are for.
- **The exam is kind.** The verifier's author wrote it, and it skipped the pairs that hid the worst fault, since fixed, where a volatility formula passed for a return idea. Hand reading and the red team found more than it did.
- **The judge can be wrong.** With the final rubric it rejected 92% of the red team's formulas, 87% to 100% of blends, 4% of 200 teacher formulas and 9% of 100 of DeepSeek's. Those are the formulas the rubric was tuned on: a stricter rule was tried and withdrawn, since it doubled the teacher's rejections for no gain and contradicted the manual. On formulas it wasn't tuned on, it rejected 32 of the second red team's 35 and 8 of 200 teacher formulas (`results/factor/judge_fresh.json`). Nobody independent checked it, and RL may find what fools it. On a hard case its thinking can use up the 4,000 tokens it was measured with and leave no reply, so a retry allows 8,000 and then 16,000. Formulas that needed the extra room weren't part of those rates.
- **Copies of known styles** earn exactly 0.65 under arm B, since their incremental IC is 0, which beats an honest formula with a negative one.
- **S1 was filtered by the first verifier.** 8 of its 1,834 ordinary rows fail the frozen one, and none of its recovery rows. They stay in, because SFT1 was training when the verifier changed, and a teacher that wasn't trying to game anything wrote them.
- **"Unseen" has limits.** Held-out families share data fields with training families. Reference formulas of 9 held-out ideas appear verbatim among the tests of 21 of S1's 2,021 trajectories, 3 of them as the answer. The teacher's manual quotes the references of `taker_up`, `illiquid_up` and, from val, `close_high_up`; the student never sees it.
- **Mirrors.** 10 of 15 val ideas have a train idea with the same formula up to sign, so val leans toward recall when picking settings, and no held-out idea does. 16 of 31 held-out ideas form 8 pairs with the same formula and opposite signs, whose ICs cancel when one formula serves both, which leaves H6 little room.
- **Selection bias.** Some ideas and seeds come from a research system whose factors were chosen on data through 2025, so reward-period IC may lean optimistic. Only the sealed period is clean of that.
- **The teacher's ability caps what SFT can teach.** Offline and online RL are the only stages that can exceed it.
- **S3 was changed after seeing val.** Its replacement was tested on val ideas, on arm B only, so arm B's 73.3% is the best of six checkpoints on the set the fix was judged by. No held-out idea played a part. That negative weights on shared syntax caused the collapse rests on this one comparison, whose two runs also differ in steps per epoch. The replacement showed no gain on val: reward 0.541 against S2's 0.546, IC among passes +0.0075 against +0.0096. With its negative rows gone, S3 also can't show what pushing down a teacher's bad answers would do. The rules picked the first checkpoint for arms A and C, half an epoch at a low rate, so those two have moved little from S2 and the difference between arms has to come mostly from S4.
- **Reward can't see a clock.** A formula can't stay blank through 2024 any more, but one built on the number of listed coins could still behave differently in 2025. It's unlikely by chance and the audit would read it.

## Stop rules

- **Online RL guard, steps 1 to 10:** as in prereg 02, stop if `mismatch_kl/all/mean` is above 0.01 for two consecutive steps, or `entropy/all/mean` is below 0.3 or above 6.0 for two consecutive steps; then rerun from the start once with lr 1e-5.
- **Online RL reward guard, the whole run:** stop if the mean reward over any five consecutive steps is more than 30% below the mean of steps 1 to 3; then rerun from the start once with lr 1e-5. A second trip stops the flow with the GPUs free. S3's first version showed that skill can go with entropy and KL looking fine, and this was added before any online result on this task.
- **SFT:** no early stop inside a run. A sweep may end once its val curve has been flat for two checkpoints, and that is reported.
- **Offline RL:** if an arm's val pass rate falls by more than 10 points from S2, the stage is reported and that arm's S4 starts from S2 instead.
- **Judge outages:** a rollout whose judge call fails is left out of its group, as in prereg 01. If more than 10% of a step's rollouts are lost that way for three steps running, the run is paused until the API recovers.

## Exploratory, reported as such

- LoRA ranks on the final SFT1 data.
- The first version of S3, with its negative rows. Its val checkpoints are in `results/factor/agent/`.
- A second round of recovery trajectories, from broken formulas of the SFT model itself.
- Research-window IC against 2025 IC of each stage's formulas. On the teacher's strategy trajectories, within-group correlation is 0.75, and picking by research IC adds 0.047 reward.
- Variety of formulas handed in, counted by skeleton.

## Pinned files

| File or artifact | sha256 |
|---|---|
| `dev.npz` | `7261b702f49c422e5ad8858e6f688ce68523d66891bb984c83a75b392ed5051e` |
| `sealed.npz` | `b8f622419c2b81deea7b378925eb415bc426d87f0458bb07abee308232256b03` |
| `task/ideas_v2.jsonl` | `bb194a304d29581411aae828b0b4a60b80fb673647235549700c5dd5b1f0eb5c` |
| `envs/factor_idea/factor_idea/sandbox.py` | `f28df8a6fa6ecb530e644a51afc2ab02c26cdda6f3b0eef7d55f08d49af8af8d` |
| `envs/factor_idea/factor_idea/dsl.py` | `12238b200de2d504f9f83137fbadb7755f6444dbc2b338f186eb78f947876667` |
| `envs/factor_idea/factor_idea/ops.py` | `f26b2a48285be439bfc5cf945ac9a5132b1684675495d6360d615b7ec8741ea1` |
| `envs/factor_idea/factor_idea/evaluate.py` | `1b816f6404cdb5f4dfb97fb3923e0be9b3aef6d8765c07e9e27478d7cbb236d0` |
| `task/judge_rubric.md` | `4084d546e7bc1b1348c916fc5512e2ae5e195eeb61e15ae927e8b6a2a6caf427` |
| `envs/factor_idea/factor_idea/judge.py` | `84f58a25d3582e915923bceabb8ac921d36753359f2ee8ed049bf200ea99e9f9` |
| `task/manual.zh.md` | `1f1a1b814af07b2efbc74ddcbe6d4378e4cf7bcf086e0a79219b282f9d249445` |
| `scripts/factor/teacher_traces.py` | `91b778b00e28e0330712ee418f3eb5a32f3e575be2d63de2047134013b06a348` |
| `envs/factor_idea/factor_idea/taskset.py` | `1e6540d4174c7bf4663df57371f06c97a9b54747dfe0adddf8accdb510b11e49` |
| `envs/factor_idea/factor_idea/prompt.py` | `b80bc2598e06b121f28c47dde1901ccee06d94bf0142da2c94e1e707329ccde6` |
| `envs/factor_idea/factor_idea/tools.py` | `f45d461beef850caac449bd527e4b9efe5e10e430fe8d90053c78c2ce9da670d` |
| `envs/factor_idea/factor_idea/workers.py` | `64093d2cf7e5550a2b071028c218ce668b775b38f34cf57f149dd0a221868290` |
| `setup/renderers_olmo3.py` | `29359105bfb75a69782c9cb2a55dc8e1d14798b0b3994aeac0ed76f9c484f402` |
| `setup/vllm_olmo3_keep_text.py` | `944aa9390c856e4687dab813d03db5e95492ab8bd657b4214f4355cfb9abacb7` |
| `setup/patch_prime_rl_sft.py` | `69062fc7d850b1ddcddccfed7bda95d4567211a1c2b73b4b6fb729451f5d2714` |
| `scripts/factor/run_stages.py` | `efc8719b1667a10826efd7f5e2bc4956171e65f46d14c9f0a45dfc76324b1f8b` |
| `scripts/factor/make_stage_datasets.py` | `5459b5e46dc5fb484371f5e434445409933131aa69cfd03f1093f5c3d07e8e33` |
| `scripts/factor/judge_eval.py` | `263e6f73951a7597d2dfe01077736dc4a8502d5599590342aad8da6613904a7b` |
| `scripts/factor/compare_stages.py` | `4abcac23e3530355f7e3658e6ca3e40c49cf3afcc3dc7d1898f8eefaf96278a2` |
| `scripts/factor/rescore_eval.py` | `141100a9d0bdfeed90d541bfa6a24a3ec6b9b5c7a82ef3ba4dc596319738f150` |
| `configs/factor-idea/sft.toml` | `8b2baee35b48ba9bf6f058749c16ee3be20d05620b672a288b8135eb9a463ead` |
| `configs/factor-idea/rl.toml` | `7fa07091029b658edc703951a7ec86b35b344efda8761dd8d50f258db9118838` |
| `configs/factor-idea/rl-gates.toml` | `fbc4544d0a6270fdcf5eae35abf3f11bda5ed0c6f7251a16a3659a7bd9624fe9` |
| `configs/factor-idea/rl-judged.toml` | `ece3aec62e54a948d1f420b05a80e37691079153048e093b29f1384f67d12e1e` |
| `configs/factor-idea/eval.toml` | `b472f5e84b13149a482d47a11cbea616b6ec04094ba38ac9f4a54cd440e1dce6` |
| dataset `sft/s1` | `6eff339c63215c3a812851048e1aa5e4bde86788ae27fa6f3e10fac2f14ba311` |
| dataset `sft/s2` | `450d26a8dc1aee29db6e304b2f8088056f23a8b032a24cae9d8f2ab97312cd34` |
| dataset `sft/s3p-gates` | `4697b54e267f864583bcba81baa9d5eb84859d959768f0748e7d3d5b5d7ec406` |
| dataset `sft/s3p-ic` | `5775c6ac25c747f0cb895af504026f9ba9ce15968694a3340745150317d07350` |
| dataset `sft/s3p-judged` | `901462035e230fec5d98707c051a3d9a59085f82509d731f4881fbe35fdbf3a5` |
| dataset `sft/s3-gates` | `6c69395a3f9312a1b9eb25e5b23f9326b724b2907aafa56c7b13815c2af78720` |
| dataset `sft/s3-ic` | `400ece8314c1d914b1cee90b27b9440b9ccdb35196cd4a66b252d0072c260dfa` |
| dataset `sft/s3-judged` | `6cd0cb0390bc7a9af7430807e9b725b4312180a507cee52cfc177249421525fa` |
| adapter `factor-s1` | `9bebbd843147f55b44d3bdce6d0a2d93737adbc98b5c7603c7d8b8dbbf8b0199` |
| adapter `factor-s2` | `e2af1b3f12f712967cbe631b8fe768edf3a606875ba9c614bb63de886e36e003` |
| adapter `factor-s3p-gates` | `85a5331907942e1b3d3f625f0ebbce99913d9af3d77a5f401f630ea2dc647d7c` |
| adapter `factor-s3p-ic` | `8557a72135790ddea9b5deac5bca02d11c1a0344302351ff43b93bc887e1c98d` |
| adapter `factor-s3p-judged` | `77bf9db573c4ec1c58d0bbfc2ccef716d6be562416e32cedb456f6c37803823e` |
| adapter `factor-s3-gates` | `ef5f084aa0537dcc32fb5a17fea3dc823c96b7be5f15865747298d30f350d86b` |
| adapter `factor-s3-ic` | `472f38ad134506122961ee2098edb45457c7ccbe4212ce2a087d50f583032918` |
| adapter `factor-s3-judged` | `d04255ac5263b652a747fe42d612d40a20c2e8ffb6a44f1a95c1f4a9ddd61f20` |

## Deviations

- **Scripts written after the freeze, not pinned.** `scripts/factor/heldout_online.py`, `audit.py` and `sealed_exam.py` carry out the rules above and call the pinned `compare_stages.py` and `rescore_eval.py`. What they settled that the rules left open is listed here.
- **Judge outages.** The pause rule was not automated. It never came up: every step of both arm C runs trained all 128 rollouts.
- **Audit, repeats.** A pair of idea and formula handed in by more than one model was read once, and its verdict counts for each of them: 163 items from 235 formulas.
- **Audit, rulings.** The rubric leaves some cases open. My rulings were written with the verdicts, before the key was opened, in `results/factor/audit/reading_rules.md`, and 54 items were marked borderline at the same time. Other readings of them are reported in `results/factor/week2.md`. H7 fails under one of them.
- **Audit, scorer.** The first run of `audit.py score` printed "fails" on H7 and counted no borderline items. The first was a float falling just under -0.1 at exactly 10 points, the second a variable overwritten in a loop. Both were fixed after seeing that output. The verdict file didn't change, and its hash was written before the key was opened.
- **Audit, second reader.** The owner of the research system has not read the sample.
- **Sealed exam.** DeepSeek's held-out rows were taken from its file of all ideas before scoring, so only held-out answers read the sealed period. `panel.py`, which logs each opening, was never among the pins. It equals the freeze commit.
- **After the exam.** The exam's stored 2026 scores were read again for exploration, in `results/factor/week2.md` under "After the exam". The sealed panel was not reopened. None of it changes a result above.
