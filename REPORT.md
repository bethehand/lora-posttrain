# Post-training small models with LoRA on 4× RTX 4090

*Chapters 1 and 2: wiki-search with Qwen3-4B and OLMo 3 7B. Draft for review, 2026-11-03.*

## Summary

Prime Intellect reported that LoRA RL lifts Qwen3-4B-Instruct-2507 from 0.55 to 0.89 on wiki-search, level with gpt-4.1 (0.88). The 100 questions behind those numbers are rows 0-99 of the question bank the model trained on. We split the bank by source page and, for each of two base models, trained one LoRA adapter for 35 steps on four RTX 4090s with the same recipe, after freezing the hypotheses in a pre-registration: Qwen3-4B in chapter 1, OLMo 3 7B Instruct in chapter 2.

- **Qwen3-4B: a real but small held-out gain, from reading the right section.** On 490 questions from 88 pages never trained on, accuracy goes from 0.567 to 0.633 (+0.066, 95% CI +0.042 to +0.090), about a fifth of the published gain. It gains +0.371 on the 174 questions it trained on and +0.207 on held-out questions of the same kind; most of that gap is that 94% of held-out questions are ones the base model always or never answers. Wrong-section failures fall from 30.1% to 25.9% of episodes, and two thirds of the new correct answers appear in a tool output.
- **OLMo 3 7B: the held-out score more than doubles, mostly from answering from memory.** With the read tool's argument renamed to `section`, which leaves Qwen's scores unchanged, accuracy goes from 0.080 to 0.180 (+0.101, +0.079 to +0.122), and all six pre-registered hypotheses pass. RL repairs the call format: calls the parser cannot read fall from 34.7% to 8.4% of episodes. But 158 of the 197 additional correct episodes contain the answer in no tool output, and in a hand-read sample 18 of 20 such answers came from memory after the tools failed. The adapter also learned to skip listing a page's sections, from 67.7% to 49.5% of episodes, and its guessed section ids fail.
- **Same recipe, different lessons.** Where the base model could use the tools, RL improved retrieval; where it could not, RL found the cheaper path the reward allowed. The judge sees only the final reply and cannot tell the two apart. A held-out split, statistics on where the answers come from, and reading trajectories can.
- **The published gain is mostly on trained questions the base model never answers.** Seven tenths of the published +0.34 is on 36 of the 100 questions our Qwen base model never answers; the published adapter, trained on them, answers 0.78 of them, ours 0.13.

## 1. Motivation

A recurring claim of 2025 is that a small open model, post-trained with RL on a narrow agentic task, matches frontier models on that task. Such claims are cheap to make and rarely checked end to end, because the check needs the training run, not just the eval. One machine with four consumer GPUs is enough to do it for one claim: audit the original evaluation, rerun the training on a clean split, and decide pre-registered hypotheses with paired statistics. This report does that for wiki-search with two base models from different families under one recipe, and builds the pipeline the next chapters reuse.

## 2. Audit of the published numbers

- **The eval questions were training questions.** The published eval (verifiers baea257) scored rows 0-99 of `willcb/wiki-trivia-questions`, the file the RL run trained on.
- **The comparison with gpt-4.1 is noise.** On those 100 questions the RL model and gpt-4.1 disagree on 11, split 6 to 5 (exact McNemar p = 1.0).
- **The judge erred in one direction.** We re-graded all 700 published verdicts (7 models × 100 questions) with two DeepSeek judges. Agreement with the original gpt-4.1-mini judge is 95.7% for deepseek-flash (Cohen's kappa 0.872) and 96.0% for deepseek-v4-pro (0.882). Most disagreements are the original judge saying no where ours say yes; it scored gpt-4.1 at 0.88, both DeepSeek judges at 0.96 (`results/judge_agreement/`).

## 3. Setup

### The recipe, chapter 1

| Item | Value |
|---|---|
| Base model | Qwen3-4B-Instruct-2507 @ cdbee75 |
| Environment | wiki-search from verifiers b2e4e81: search pages, list a page's sections, read one section; a judge compares the final reply with the reference answer |
| Split | by source page: train 1,771 questions from 331 pages; held-out 490 from 88 pages, which include the published 100 |
| Training questions | `train_mixed`, the 174 training questions the base model answers right in 1 to 7 of 8 rollouts. Of the rest, 1,005 are always right and 592 never, so they give zero advantage |
| Trainer | prime-rl v0.9.0; LoRA rank 8, alpha 32, on all linear projections (16.5M parameters); AdamW, lr 3e-5 constant; GRPO advantages (reward minus group mean); group 8 × 16 questions per step, 35 steps; 512 tokens per turn, 8,192 per episode |
| Hardware | 3 GPUs serve the policy with vLLM, 1 GPU trains; asynchronous, a batch ships once inference serves the policy from two steps earlier |
| Judges | training reward: deepseek-flash; evaluation: deepseek-v4-pro, with every rollout re-graded by flash |
| Evaluation | base and adapter served by one LoRA-enabled vLLM server in one session; 4 rollouts per question; paired bootstrap over questions (10,000 resamples) |

The recipe starts from prime-rl's wiki-search example (8 GPUs, batch 512, group 16, 200 steps, lr 1e-5, 4,096 tokens). Three changes, all made before any training result:

- **Only mixed questions.** The base model is close to deterministic per question, so only about one training question in ten carries a learning signal.
- **lr 3e-5.** Adam moves each weight by at most about lr per step, so 35 steps at the recipe's 1e-5 would move the adapter about a sixth as far as its 200 steps, with LoRA's B matrix starting at zero.
- **8,192 tokens per episode.** Longer reads must not be cut off at zero reward.

The design, hypotheses and thresholds are in `prereg/01-wiki-search.md`, frozen in commit 979250a before training started.

### What changes for OLMo, chapter 2

Everything not listed is as in chapter 1. The design is in `prereg/02-olmo-wiki-search.md`, frozen in commit 6b6d470 before training started.

| Item | Chapter 2 | Why |
|---|---|---|
| Base model | OLMo-3-7B-Instruct @ 6e5971d | a second family, fully open down to its pretraining data |
| Read argument | `section` instead of `section_id`, in training and in every eval but one | with `section_id`, 80% of OLMo's held-out episodes end on a call the parser cannot read; Qwen scores 0.566 and 0.568 under the two names, +0.066 and +0.065 after RL |
| Training questions | `train_mixed_olmo`, the 489 training questions OLMo answers right in 1 to 7 of 8 rollouts; 5 are always right, 1,277 never | chapter 1's rule on OLMo's own rollouts. All are used, so each is trained on about 1.1 times, against 3.4 for Qwen |
| Renderer | renderers' default renderer with OLMo's chat template, and an OLMo tool parser ported from vLLM's (`setup/renderers_olmo3.py`) | renderers 0.1.10 has none for OLMo. On 300 prompts, the training path's prompt tokens and parsed calls equal those of vLLM's chat endpoint in all 300 (`results/olmo3-parity`) |
| Hardware | 2 GPUs serve the policy, 2 train with FSDP | the trainer keeps fp32 weights, 29 GB for 7.3B parameters. bf16 weights would round away updates at lr 3e-5, and full optimizer offload turns gradient clipping off |
| Instability guard | entropy band 0.3 to 6.0; KL threshold 0.01 as before | chapter 1's band, 0.02 to 0.5, sat at about 0.28 and 4 times its pilot's entropy; OLMo's pilot gave 1.14 and 1.57 |

## 4. Chapter 1 results: Qwen3-4B

### Accuracy

Judge deepseek-v4-pro, 4 rollouts per question, no errored episodes.

| Questions | Base | Adapter | Gain | 95% CI |
|---|---|---|---|---|
| `heldout`: 490, pages never trained on | 0.567 | 0.633 | +0.066 | [+0.042, +0.090] |
| `slide100`: the published 100, inside `heldout` | 0.585 | 0.645 | +0.060 | |
| `train_sample`: 300 from training pages | 0.599 | 0.692 | +0.092 | [+0.059, +0.128] |
| of which the 265 not trained on | 0.601 | 0.659 | +0.058 | |
| `train_mixed`: the 174 trained on | 0.483 | 0.853 | +0.371 | [+0.315, +0.427] |
| `heldout` without tools | 0.301 | 0.317 | +0.016 | [+0.002, +0.031] |

The split of `train_sample` into trained and untrained questions is exploratory. On `slide100` our adapter scores 0.645; the published adapter, which trained on these questions, scores 0.89 on its published rollouts re-graded by v4-pro, and our adapter reaches 0.853 on the questions it trained on.

### Pre-registered hypotheses

| ID | Hypothesis | Result | Verdict |
|---|---|---|---|
| H1 (primary) | RL improves held-out accuracy | gain +0.066, CI lower bound +0.042 | pass |
| H1b | The gain is at least 0.20 | +0.066 | fail |
| H2 | The setup reproduces the published base score | base on `slide100` 0.585, within 0.45-0.65 | pass |
| H3 | The gain does not depend on having trained on the page | `train_sample` gain minus `heldout` gain: +0.027 [-0.014, +0.069], at most 0.10 | pass |
| H4 | RL teaches the model to read further | 0.985 section reads per episode against a threshold of 1.5; reads up +0.048 [+0.030, +0.068]; wrong-section failures down -0.042 [-0.064, -0.020] | fail |

`results/prereg01_tests.md` is the full table, made by `scripts/prereg01_tests.py`.

### Where the held-out gain comes from

The base model gets nearly every held-out question either right 4 times out of 4 or wrong 4 times out of 4.

| Base on the question | Questions | Base | Adapter | Contribution to the gain |
|---|---|---|---|---|
| 0 of 4 | 196 | 0.000 | 0.158 | +0.063 |
| 1 to 3 of 4 | 32 | 0.500 | 0.695 | +0.013 |
| 4 of 4 | 262 | 1.000 | 0.981 | -0.010 |

Grouping and measuring with the same base rollouts invites regression to the mean, but with so few mixed questions the effect is small. Nearly the whole gain is on questions the base model never answers, and 43 of those 196 are now answered at least once. The adapter trained only on mixed questions, so this is transfer to harder questions, not practice on the same ones.

Grouping by an earlier, independent eval of the base model (`results/evals/qwen3-4b-base`, the H2 check before the freeze) avoids the regression to the mean, and its middle group is the untrained counterpart of the trained questions, which were also picked for mixed outcomes. This analysis was added after the results.

| `heldout` by the earlier base eval | Questions | Base | Adapter | Gain | Trained gain minus this, 95% CI |
|---|---|---|---|---|---|
| `train_mixed`, trained | 174 | 0.483 | 0.853 | +0.371 | |
| 0 of 4 | 196 | 0.013 | 0.161 | +0.148 | +0.223 [+0.150, +0.294] |
| 1 to 3 of 4 | 29 | 0.448 | 0.655 | +0.207 | +0.164 [+0.006, +0.316] |
| 4 of 4 | 265 | 0.991 | 0.980 | -0.010 | +0.381 [+0.324, +0.438] |

Untrained questions with mixed outcomes start where the trained ones did, 0.448 against 0.483, and gain +0.207, a little over half the trained gain. So the gap between +0.371 on trained questions and +0.066 on held-out ones is mostly a matter of which questions are measured: 94% of held-out questions are ones the base model always or never answers, where a gain is capped or hard to reach. Training on the question itself adds about +0.16, with a wide CI from 29 questions.

The first grouping, by the base rollouts of this session, also splits the published gain on `slide100`. The published columns are the published rollouts, one per question with the published prompt, re-graded by v4-pro.

| `slide100` by our base score | Questions | Our base | Our adapter | Published base | Published adapter | Contribution to the published gain |
|---|---|---|---|---|---|---|
| 0 of 4 | 36 | 0.000 | 0.125 | 0.111 | 0.778 | +0.240 |
| 1 to 3 of 4 | 9 | 0.389 | 0.583 | 0.556 | 0.778 | +0.020 |
| 4 of 4 | 55 | 1.000 | 0.995 | 0.836 | 0.982 | +0.080 |

Seven tenths of the published gain, +0.24 of +0.34, is on the 36 questions our base model never answers. The published adapter trained on these questions and answers 0.78 of them; ours, which trained on no question its base model never answers, answers 0.13. Both adapter columns are measured apart from the base rollouts that form the groups. So the two gains on trained questions, +0.371 on `train_mixed` and the published +0.34, are not the same measurement: `train_mixed` holds only questions the base model sometimes answers, the easiest to move. `scripts/by_base_score.py` makes all three tables (`results/by_base_score.md`).

The navigation metrics (`scripts/navigation_metrics.py`) say how. An episode still makes about three calls, search, list sections, read one section: 2.98 calls per episode before and 3.12 after. What changes is which section. The share of episodes that read the source page, never see the answer in any tool output, and fail falls from 30.1% to 25.9%. Of the 129 additional correct episodes, 86 have the answer in a tool output (`scripts/evidence_use.py`), so most of the gain is retrieval. The published adapter, trained on about 23 times as many rollouts, reads 1.85 sections per episode in its published rollouts; ours reads 0.99.

### Checks

- **Judge.** Re-graded by deepseek-flash, the training judge, the held-out gain is +0.069 [+0.046, +0.093], and the two judges agree on 98.8% of episodes. A policy that had learned to please the training judge would gain more under it.
- **Tool interface.** With the read tool's argument renamed from `section_id` to `section`, the gain is +0.065 [+0.040, +0.091], and each model's score moves by at most 0.002.
- **Memory.** Without tools the adapter gains +0.016, so the gain needs the tools.
- **Trajectories.** In 50 random held-out episodes of the adapter (`results/audit/rl35_heldout_labels.md`), none of the 28 correct replies names more than one candidate, and the two replies that hedge across candidates were judged wrong. In 21 of the 22 failures the answer never appears in any tool output: the model read one section, did not find it, and answered or denied the premise instead of reading on. Two of these are flawed questions about page metadata.

### Training

Training reward on `train_mixed` (deepseek-flash) averages 0.53 over steps 1-10 and 0.76 over steps 26-35. Online held-out evals during training (flash, 1 rollout per question, exploratory) read 0.569, 0.596, 0.637 and 0.643 at steps 0, 15, 30 and 35. The pre-registered instability guard did not trigger: mismatch KL stayed between 0.0006 and 0.0038 and entropy between 0.047 and 0.179 (`results/train/prereg01/steps.txt`).

## 5. Chapter 2 results: OLMo 3 7B

### Starting point

Before any training, OLMo 3 7B Instruct scores 0.020 on `heldout` with the original `section_id` argument and 0.252 without tools. 80% of its episodes with tools end on a call the parser cannot read, such as `wiki_read_section(section_ "...")`; the fault is in the model, since the same malformed call appears in 11 of 20 raw greedy samples. Renaming the argument to `section` raises well-formed calls from 22% to 70% in a probe and accuracy to 0.094 (`results/olmo-tool-call-diagnosis/`). Even then, 74% of episodes fail on a broken call, one the parser cannot read or one the tool answers with an error (`scripts/call_failures.py`). AI2 reports OLMo 3 level with Qwen3-8B on SimpleQA with its own tool harness, so the gap is robustness across harnesses.

### Accuracy

Judge deepseek-v4-pro, 4 rollouts per question, `section` argument unless noted.

| Questions | Base | Adapter | Gain | 95% CI |
|---|---|---|---|---|
| `heldout`: 490, pages never trained on | 0.080 | 0.180 | +0.101 | [+0.079, +0.122] |
| `slide100`: the published 100, inside `heldout` | 0.092 | 0.175 | +0.083 | |
| `train_sample`: 300 from training pages | 0.087 | 0.196 | +0.108 | [+0.081, +0.137] |
| `train_mixed_olmo`: the 489 trained on | 0.263 | 0.540 | +0.277 | [+0.246, +0.308] |
| `heldout_mixed_olmo`: 97 held-out questions OLMo sometimes answered before the freeze | 0.245 | 0.433 | +0.188 | [+0.126, +0.250] |
| `heldout` with `section_id` | 0.022 | 0.074 | +0.052 | [+0.038, +0.067] |
| `heldout` without tools | 0.249 | 0.234 | -0.015 | [-0.029, -0.002] |

A DeepSeek API outage between 16:50 and 18:20 failed the judge calls of 942 of the base model's 1,960 held-out episodes and 666 of the adapter's; the rollouts were complete. After the run the failed calls were retried with the same judge, prompt and inputs, keeping every verdict the run gave; re-grading 200 already-scored episodes per affected set agreed with their original verdicts in 98.5 to 100% (`scripts/rescore_failed.py`, prereg 02's Deviations). The table uses the retried verdicts. With the failed calls dropped, as pre-registered, 275 held-out questions are scored by both runs, H1 reads +0.095 [+0.065, +0.125], and every verdict below is the same (`results/prereg02_tests-prereg02-rl35.md`).

### Pre-registered hypotheses

| ID | Hypothesis | Result | Verdict |
|---|---|---|---|
| H1 (primary) | RL improves OLMo's held-out accuracy | gain +0.101, CI lower bound +0.079 | pass |
| H2 | The setup reproduces OLMo's base score | base on `heldout` 0.080, within 0.06-0.13 | pass |
| H3 | The gain does not depend on having trained on the page | `train_sample` gain minus `heldout` gain: +0.008 [-0.027, +0.043], at most 0.10 | pass |
| H4 | RL repairs the tool calls | share of episodes failing on a broken call: 0.743 to 0.561, -0.182 [-0.214, -0.151] | pass |
| H5 | The gain carries to untrained questions of the trained kind | `heldout_mixed_olmo` gain +0.188 [+0.126, +0.250]; 0.68 [0.45, 0.92] of the trained-question gain | pass |
| H6 | The repair carries to the original argument name | gain with `section_id` +0.052 [+0.038, +0.067] | pass |

`results/prereg02_tests-prereg02-rl35-rescored.md` is the full table, made by `scripts/prereg02_tests.py`.

### What RL changed

**The call format is repaired; the arguments are not.** On `heldout`:

| Episodes failing on a broken call | Base | Adapter | Change, 95% CI |
|---|---|---|---|
| ended on a call the parser cannot read | 0.347 | 0.084 | -0.263 [-0.290, -0.238] |
| got an error back from a tool | 0.396 | 0.477 | +0.081 [+0.051, +0.111] |
| either | 0.743 | 0.561 | -0.182 [-0.214, -0.151] |

**Most new correct answers did not come from the tools.** Over all 1,960 held-out episodes per model (`scripts/evidence_use.py`):

| Episodes judged right | Base | Adapter |
|---|---|---|
| with the gold answer in some tool output | 92 | 131 |
| with the gold answer in no tool output | 64 | 222 |
| of which no read returned any text | 31 | 135 |
| all | 156 | 353 |

The check is a normalized substring match, so spelling variants count as unseen. Reading the episodes settles it (`results/audit/olmo_rl35_heldout_labels.md`). Of 20 random adapter episodes judged right with the answer in no tool output, 18 retrieved nothing relevant and answered from memory, several saying so ("based on well-known information"); one had the answer in its section under other wording; one was judged right for a wrong answer. So roughly 7 of the 10.1 points come from answering from memory instead of ending on a broken call, and about 2 from answers read in a section. Reading did improve: episodes that had the answer in a tool output and still answered wrong fell from 258 to 130. In 50 random adapter episodes, 5 of the 9 correct answers came from a section and 4 from memory.

**The adapter learned to skip listing sections.** Episodes that call `wiki_view_sections` fall from 1,327 (67.7%) to 971 (49.5%); of the episodes that read a section, those that never listed the page's sections rise from 526 of 1,604 to 939 of 1,728. The guessed section ids fail:

| Tool error, episodes with at least one | Base | Adapter |
|---|---|---|
| `read_section` argument without the `page_id:` prefix | 733 | 834 |
| tool that does not exist (`wiki_read_sections`) | 401 | 357 |
| `read_section` with an unknown page id, often the page title | 483 | 346 |
| `view_sections` with an unknown page id | 313 | 209 |
| a required argument missing | 189 | 3 |

**Long sections garble the reply.** About half of all episodes that read a section over 4,000 characters end in a garbled reply, repeated fragments or runs of `##`, in both models: 77 of 146 for the base model, 44 of 83 for the adapter. In 5 of the 7 garbled replies of the random sample, the answer was in the text just read. RL made long reads rarer but did not fix this. Qwen garbles 6 of 1,960 replies, before and after.

### Checks

- **Judge.** Re-graded by deepseek-flash, the training judge, the held-out gain is +0.107 [+0.084, +0.130]; the two judges agree on 98.8% of the base model's episodes and 97.7% of the adapter's. In the 70 episodes read, the judge erred once each way.
- **Memory.** Without tools the adapter loses 0.015, so its knowledge did not grow; it uses that knowledge more often when the tools fail. With tools it still scores below its own no-tools score, 0.180 against 0.234.
- **Navigation.** Section reads per episode fall from 2.44 to 1.62, and wrong-section failures do not move (0.263 and 0.265).

### Training

Training reward on `train_mixed_olmo` (deepseek-flash, all rollouts) averages 0.125 over steps 1-10 and 0.280 over steps 26-35. Online held-out evals (flash, 1 rollout per question, exploratory) read 0.088, 0.121, 0.188 and 0.182 at steps 0, 15, 30 and 35; the last lost 9.4% of its judge calls at the start of the outage. The guard did not trigger: over steps 1-10 entropy stayed between 1.33 and 1.81 and mismatch KL between 0.0013 and 0.0029. Over the whole run entropy stayed between 1.09 and 1.91, and KL peaked once at 0.0086, at step 18 (`results/train/prereg02/steps.txt`).

## 6. Two models, one recipe

| | Qwen3-4B | OLMo 3 7B |
|---|---|---|
| Base on `heldout` | 0.567 | 0.080 |
| Held-out gain | +0.066 [+0.042, +0.090] | +0.101 [+0.079, +0.122] |
| Gain on trained questions | +0.371 | +0.277 |
| Gain on untrained held-out questions of the trained kind | +0.207 (29 questions) | +0.188 (97 questions) |
| Ratio of the two | 0.56 | 0.68 |
| Times each trained question was trained on | 3.4 | about 1.1 |
| Gain without tools | +0.016 | -0.015 |
| New correct episodes with the answer in a tool output | 86 of 129 | 39 of 197 |
| Episodes that list a page's sections | 94.6% to 98.0% | 67.7% to 49.5% |
| Garbled final replies | 0.3%, both | 7.9% to 6.2% |
| What RL changed | which section is read | the call format, and answering after the tools fail |

**What RL teaches depends on what the base model can already do.** Qwen arrives using the tools correctly, so the only way up is better retrieval, and that is what it learns. OLMo arrives with broken calls and with more knowledge than its tool use can show, 0.252 without tools against 0.094 with them. The cheapest way up the reward is to stop failing and answer, so that is what it learns first, along with dropping a step, listing sections, that the reward does not pay for.

**The final-reply judge cannot see the difference.** Both models' gains pass every pre-registered test, and for OLMo the judge check, the flash re-grade, agrees with v4-pro. Only the statistics on where the answers come from and the trajectories show that OLMo's gain is mostly memory. For a task where the process is the point, the reward has to check the evidence: that the answer appears in what the tools returned, or that the code actually ran.

**Transfer looks higher for OLMo, but the comparison is descriptive.** OLMo's gain on comparable untrained questions is 0.68 of its trained-question gain against Qwen's 0.56, but OLMo trained on each question about a third as often, which by itself shrinks the trained-question gain. And much of OLMo's transfer is the general habit of answering from memory, which carries to any question.

## 7. Cost on 4× RTX 4090

| Item | Qwen3-4B | OLMo 3 7B |
|---|---|---|
| Pilot | 8.5 minutes for 2 steps | 18 minutes for 2 steps; a first try with one trainer GPU ran out of memory while loading |
| Training, 35 steps | 72 minutes, about 2 minutes per step | 4 h 36 min, about 6 minutes per step plus four online evals of 10 to 28 minutes |
| Trainer peak memory | 18.2 GiB on 1 GPU | 18.3 GiB on each of 2 GPUs |
| Samples per episode in the trainer | 1.0 | 4.6 to 5.4 |
| Evaluation | 16 minutes, 11,632 episodes | 2 h 33 min, 18,072 episodes, slowed by the judge outage |
| Judge tokens, v4-pro evaluation | about 3.7M | about 5.7M, plus the retried calls |

OLMo's 8-rollout difficulty filter over the 1,771 training questions took 55 minutes on four GPUs. OLMo is slower for four reasons: 7.3B against 4B parameters, no grouped-query attention (512 KiB of KV cache per token against 144 KiB), two inference GPUs instead of three, and a renderer that renders the whole conversation again each turn, so each turn becomes its own training sample.

## 8. Pitfalls

The first five are in `results/smoke-test.md`: stalled GitHub downloads during install, `vllm-router` missing from the installed extras, port 8000 taken on a shared server, vLLM refusing Qwen's 262k default context, and the wiki tool server's ONNX embedder taking every CPU core. Later ones:

- **The question bank is almost deterministic for the base model.** Without the difficulty filter, nine in ten training questions would cost rollouts and give no gradient.
- **Judge outages drop rollouts, safely, during training.** DeepSeek connection errors fail the rollout, and prime-rl leaves failed rollouts out of the group baseline, so they are lost data, not zero rewards.
- **Judge outages during evaluation lose whole stretches of questions.** The rollouts survive in the traces, so the failed calls can be retried with the same judge; report both versions.
- **The router answers `/v1/models` with a 500 under data parallelism.** Requests route fine; ask the engine behind it instead (`scripts/load_adapter.sh`).
- **A local model path needs the renderer named.** Without `orchestrator.renderer.name = "qwen3"` the default renderer has no tool support.
- **prime-rl has no OLMo renderer.** The default renderer with a ported tool parser works; check token-level parity with the eval path before training (`scripts/olmo3_parity.py`).
- **The trainer keeps fp32 weights.** A 7B model needs two trainer GPUs on 24 GB cards.
- **Stop rules are model-specific.** OLMo's entropy at the training temperature is ten times Qwen's; a band copied from Qwen would have stopped a healthy run at step 1.

## 9. Limitations and next steps

- **One seed and one budget per model.** The published run used about 23 times as many rollouts as either of ours. A longer run and a second seed would show whether the held-out gains keep growing and whether Qwen learns to read further.
- **The judge sees only the final reply.** For Qwen this mattered little; for OLMo most of the gain is answers from memory. A reward that requires the answer to appear in retrieved text, or that credits a successful read, would separate retrieval from memory. This is the most direct next experiment on wiki-search.
- **OLMo's remaining failures are teachable.** Missing page prefixes, a tool name that does not exist, and skipped section listings are what SFT on correct tool-use trajectories before RL would target.
- **Long tool outputs garble OLMo's replies.** Capping section length in the environment may matter more than more RL steps.
- **Never-right training questions give no signal**: 592 for Qwen, 1,277 for OLMo. A curriculum, hints, or credit for reaching the right section could open them.
- **Throughput.** The trainer sets the pace. Two trainer GPUs holding replicas of the LoRA model, rather than shards, might roughly halve Qwen's step time on a machine without peer-to-peer.
