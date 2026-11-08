# Post-training small models with LoRA on 4× RTX 4090

*Chapters 1 and 2: wiki-search with Qwen3-4B and OLMo 3 7B. Chapter 3: from an investment idea to a factor, with OLMo 3 7B. Last updated 2026-11-09.*

## Summary

Prime Intellect reported that LoRA RL lifts Qwen3-4B-Instruct-2507 from 0.55 to 0.89 on wiki-search, level with gpt-4.1 (0.88). The 100 questions behind those numbers are rows 0-99 of the question bank the model trained on. We split the bank by source page and, for each of two base models, trained one LoRA adapter for 35 steps on four RTX 4090s with the same recipe, after freezing the hypotheses in a pre-registration: Qwen3-4B in chapter 1, OLMo 3 7B Instruct in chapter 2.

- **Qwen3-4B: a real but small held-out gain, from reading the right section.** On 490 questions from 88 pages never trained on, accuracy goes from 0.567 to 0.633 (+0.066, 95% CI +0.042 to +0.090), about a fifth of the published gain. It gains +0.371 on the 174 questions it trained on but +0.066 on all held-out questions, and most of that gap is that 94% of held-out questions are ones the base model always or never answers; on held-out questions of the trained kind, which the base model sometimes answers, it gains +0.207. Wrong-section failures fall from 30.1% to 25.9% of episodes, and two thirds of the new correct answers appear in a tool output.
- **OLMo 3 7B: the held-out score more than doubles, mostly from answering from memory.** With the read tool's argument renamed to `section`, which leaves Qwen's scores unchanged, accuracy goes from 0.080 to 0.180 (+0.101, +0.079 to +0.122), and all six pre-registered hypotheses pass. RL repairs the call format: calls the parser cannot read fall from 34.7% to 8.4% of episodes. But 158 of the 197 additional correct episodes contain the answer in no tool output, and of 20 such answers read one by one, by Claude, 18 came from memory after the tools failed. The adapter also learned to skip listing a page's sections, from 67.7% to 49.5% of episodes, and its guessed section ids fail.
- **Same recipe, different lessons.** Where the base model could use the tools, RL improved retrieval; where it could not, RL found the cheaper path the reward allowed. The judge sees only the final reply and cannot tell the two apart. A held-out split, statistics on where the answers come from, and reading trajectories can.
- **The published gain is mostly on trained questions the base model never answers.** Seven tenths of the published +0.34 is on 36 of the 100 questions our Qwen base model never answers; the published adapter, trained on them, answers 0.78 of them, ours 0.13.

Chapter 3 builds a narrow task from the owner's quant research and teaches it to OLMo 3 7B under a frozen pre-registration: from a one-sentence investment idea, write a factor formula, test it up to five times on past data, and hand one in. Unlike a judge, an unopened year of market data can't be talked round, so this task can tell whether a model learned to do the work or only to pass.

- **SFT teaches the task on families it never saw.** Pass rate on 31 held-out ideas goes from 0.0% to 51.6% (+51.6 points, +41.9 to +61.3).
- **RL makes it reliable.** After offline and then online RL the student hands in nearly every time and gets the direction right, and passes 65.7% to 71.4%, against 79.0% for the teacher model answering as the student does, without its manual.
- **No stage was shown to teach better formulas, and the pre-registered tests of that all fail.** On the sealed 2026 period the final model, S4 arm B, has an incremental IC among passes of +0.0012 (-0.0008 to +0.0032), against the teacher's +0.0050 (+0.0025 to +0.0075). So the signal is there, and the student doesn't reliably find it.
- **Reading the answers afterwards, as exploration, explains much of this.** IC is 2.8% of the reward's spread within a group, a scaling fault of ours, so the reward could hardly teach it. On 2025, S2's formulas can't be told from the teacher's, and the fall comes with offline RL. Keeping one of four of the final model's answers by research IC, after the gates, reaches an IC of +0.0040 (+0.0015 to +0.0065) on 2026 at a 90.7% pass rate. These were found after the exam, and needed testing on new ideas.
- **Then those readings were tested, with the rules written first.** Two online-RL pilots from S2 failed their rules: a better-scaled reward raised IC a little and lost passes. A second exam on 21 new ideas confirmed three of four readings. The online-RL model is more reliable than S2, +19.0 points (+11.6 to +26.5). Research IC tracks 2026 IC within one model's passes on one idea, at a correlation of +0.85. Keeping one of four of the online-RL model's answers by research IC, after the gates, adds +0.0041 (+0.0027 to +0.0055) of 2026 IC. That S2 writes better formulas than online RL's model is still only a direction, +0.0020 (-0.0005 to +0.0044).

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
| Base model | Olmo-3-7B-Instruct @ 6e5971d | a second family, fully open down to its pretraining data |
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

The check is a normalized substring match, so spelling variants count as unseen. Reading the episodes settles it (`results/audit/olmo_rl35_heldout_labels.md`). Of 20 random adapter episodes judged right with the answer in no tool output, 18 retrieved nothing relevant and answered from memory, several saying so ("based on well-known information"); one had the answer in its section under other wording; one was judged right for a wrong answer. So roughly 7 of the 10.1 points come from answering from memory instead of ending on a broken call, and about 2 from answers read in a section. Reading did improve: episodes in which a read returned text and the answer was in a tool output, yet the reply was wrong, fell from 258 to 130. In 50 random adapter episodes, 5 of the 9 correct answers came from a section and 4 from memory.

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

**The final-reply judge cannot see the difference.** Both models' held-out gains pass the pre-registered tests that they exist and don't depend on having trained on the page, H1 and H3, and for OLMo the judge check, the flash re-grade, agrees with v4-pro. Only the statistics on where the answers come from and the trajectories show that OLMo's gain is mostly memory. For a task where the process is the point, the reward has to check the evidence: that the answer appears in what the tools returned, or that the code actually ran.

**Transfer looks higher for OLMo, but the comparison is descriptive.** OLMo's gain on comparable untrained questions is 0.68 of its trained-question gain against Qwen's 0.56, but OLMo trained on each question about a third as often, which by itself shrinks the trained-question gain. And much of OLMo's transfer is the general habit of answering from memory, which carries to any question.

### What this says about the wiki-search setup

| Part | What worked here | What did not |
|---|---|---|
| Environment: search pages, list sections, read a section | Both models gain on pages never trained on, +0.066 and +0.101; Qwen's gain is better retrieval | The interface is brittle. One argument name decides whether OLMo can call the read tool at all; section ids must be written `page_id:section`, and leaving out the page is the most common error; whole-page reads of 10,000 characters and more garble small models' replies |
| Judge: the final reply against the reference answer | Judges agree with each other on 96 to 99% of episodes; replies that hedge across candidates are judged wrong; no judge gaming found | It sees only the final reply, so an answer from memory earns what an answer found with the tools earns. Most of OLMo's gain is such answers |
| Question bank: 2,261 trivia questions | Large enough to split by page for a clean held-out set | For any one model most questions are always or never answered, so only 10% (Qwen) and 28% (OLMo) carry a learning signal; many are answerable without tools, 0.30 and 0.25 accuracy; a few ask about page metadata |
| Published evaluation | | It scored training questions. This, not the environment, is what made the published gain look like parity with gpt-4.1 |

So the setup gives a real training signal that carries to unseen pages, smaller than published, and its judge rewards the answer rather than the search. What this report cannot say is how much more the same setup gives at the published budget, 23 times our rollouts.

## 7. Chapter 3: from an idea to a factor

Chapter 2 ended on a question: for a task where the process is the point, can a reward check the work? This chapter tries one where it can, on the owner's own trade. The design is in `prereg/03-factor-idea.md`, frozen in commit b898b72 before any trained model read a held-out idea. The working log, with every stage and failure, is `results/factor/week2.md`.

### The task

The model gets one sentence, such as "Coins with many trades per day tend to do better". It may call `test_factor` up to five times, which returns the formula's rank IC and incremental IC on July 2022 to December 2024. Then it hands in one formula.

| Item | Value |
|---|---|
| Data | Binance USDT perpetual futures, daily bars, 9 fields, delisted coins included. Research period to 2024-12-29, reward period 2025, sealed period 2026-01-01 to 2026-10-16 |
| Ideas | 115 in 20 families: 69 train, 15 val, and 31 held out in 5 families never trained on |
| Gates | The formula computes, and it behaves as the idea says: a shock to the data the idea names moves its rank the right way, and other data doesn't dominate. The gates read data through 2024 |
| Quality | Incremental IC: rank IC with the next day's return after removing liquidity, volatility, 1-day and 5-day reversal, and 30-day momentum |
| Teacher | `deepseek-flash` with a manual of the owner's research practice, which the student never sees. In every comparison below, "the teacher" is the same model answering as the student does, with the tools and without the manual ("DeepSeek" in the tables) |
| Judge | `deepseek-flash` reads the idea and the formula and says whether it is faithful |

Three reward arms differ only in online and offline RL. Arm A pays 1 for a pass. Arm B pays 0.3 + 0.7 × clip((IC + 0.03) / 0.06, 0, 1) for a pass. Arm C pays arm B's reward if the judge finds the formula faithful. A failure earns 0 in every arm.

### The recipe

Four stages, after Samaya's Sterling, all on one LoRA adapter of rank 32 from OLMo 3 7B Instruct, each continuing from the one before. S0 is the untrained model.

| Stage | Data | Chosen on val | Passed on val |
|---|---|---|---|
| S1, SFT | 2,021 teacher trajectories that passed, 187 of them recovering from the student's own broken formulas | rank 32 against 8 and 128, step 350 of 420 | 75.0% |
| S2, SFT | 206 trajectories: for each idea and wording, the best of 8 strategies by research IC | step 35 of 42 | 71.7% |
| S3, offline RL | The teacher's strategy trajectories with a positive advantage, weighted by it: 453, 920 and 967 rows. So it is imitation weighted by advantage, and nothing in it pushes a bad answer down | A step 16, B step 64, C step 34 | 68.3%, 73.3%, 66.7% |
| S4, online RL | 35 steps of 128 rollouts in groups of 8, lr 3e-5, from each arm's S3, two seeds per arm | step 35 | 70.0% to 81.7% |

Offline RL's first version kept failing trajectories with negative weights, as the first design said, and broke the model: its last checkpoints fell to 2% or less passed, with invented methods and dropped quotes. With the negative rows out, on the same data, start and rate, that collapse disappeared. The rule for that swap was written before its result, as the prereg says, though the two entered the repo in the same commit, and it was tested on val only.

All six online runs ran their 35 steps with no guard tripped, a highest mismatch KL of 0.0054 and no failed rollout. Each kept adapter equals its step 35 byte for byte.

### Pre-registered hypotheses

`results/factor/heldout_comparisons.txt` (H1, H3, H4), `results/factor/heldout_online_comparisons.txt` (H5, H6) and `results/factor/sealed/comparisons.txt` (H8) hold the comparisons, by the pinned `scripts/factor/compare_stages.py`, per idea with a paired bootstrap, and `results/factor/audit/report.txt` holds H7 and H9. H2's interval, on val, is only in the working log. Seeds of an arm count as one model.

| ID | Hypothesis | Result | Verdict |
|---|---|---|---|
| H1 (primary) | SFT teaches the task on unseen families | S1 minus S0, pass rate +51.6 points [+41.9, +61.3] | pass |
| H2 | The same on unseen ideas of seen families | +71.7 points [+58.3, +83.3], on val, read as an upper bound | pass |
| H3 | Choosing by research IC gives better formulas | S2 minus S1, IC among passes +0.0017 [-0.0010, +0.0044] | fail |
| H4 | Offline RL gives better formulas | S3 arm B minus S2, -0.0038 [-0.0063, -0.0014] | fail, the other way |
| H5 | Online RL gives better formulas | S4 arm B minus S3 arm B, -0.0000 [-0.0017, +0.0019] | fail |
| H6 | A reward with a signal teaches more signal | S4 arm B minus S4 arm A, -0.0004 [-0.0019, +0.0011]; outside mirrored pairs +0.0008 [-0.0009, +0.0030] | fail |
| H7 | RL doesn't trade validity for score | In the audit, S4 arm B is 10.0 points below S2, at the limit of 10 | pass, on the line |
| H8 | The signal holds out of time | On 2026, S4 arm B minus S1, -0.0011 [-0.0034, +0.0011]; its own mean +0.0012 [-0.0008, +0.0032] | fail |
| H9 | A judge in the reward keeps formulas faithful | In the audit, S4 arm C minus arm B +10.0 points [-5.0, +25.0] | fail |

### Held-out ideas

124 episodes per seed, none failed, scored on 2025. Reward is arm B's formula throughout.

| Model | Passed | Answered | Reward | IC among passes | Tests | Judged faithful |
|---|---|---|---|---|---|---|
| OLMo untrained | 0.0% | 33.9% | 0.000 | none | | |
| S1 | 51.6% | 91.9% | 0.341 | +0.0009 | 3.82 | 48 of 64 |
| S2 | 44.4% | 79.8% | 0.305 | +0.0032 | 4.31 | 42 of 55 |
| S3 arm B | 59.7% | 96.0% | 0.395 | +0.0010 | 4.00 | 58 of 74 |
| S4 arm A | 71.4% | 99.6% | 0.471 | +0.0008 | 2.76 | 123 of 177 |
| S4 arm B | 68.5% | 98.4% | 0.453 | +0.0009 | 2.63 | 120 of 170 |
| S4 arm C | 65.7% | 99.6% | 0.427 | -0.0000 | 2.69 | 125 of 163 |
| DeepSeek, tools, no manual | 79.0% | 83.9% | 0.551 | +0.0041 | 4.47 | |

S4 arm B passes 16.9 points more than S1 [+7.3, +26.6] and 24.2 more than S2 [+14.1, +33.9]. Against the teacher its pass rate is -10.5 points [-25.0, +3.6], and its IC among passes -0.0031 [-0.0051, -0.0010]. It passes more often without reaching more ideas: at least one pass in four rollouts holds 87.1% to 96.8% per seed, against 90.3% for S1. So SFT widens what the student can do, and RL makes it do so more often.

### The audit

The verifier's author, Claude, read 163 distinct pairs of idea and formula, covering 235 passing formulas of S2 and the three S4 arms, shuffled and with model names hidden, by the judge's rubric. The verdicts and the rulings on what the rubric leaves open were locked by hash before the key was opened (`results/factor/audit/`); the hashes and the key entered the repo in the same commit, so that order rests on the working log.

| Model | Reader: faithful | Judge: faithful | They agree |
|---|---|---|---|
| S2 | 44 of 55, 80.0% | 42 of 55 | 49 of 55 |
| S4 arm A | 47 of 60, 78.3% | 45 of 60 | 48 of 60 |
| S4 arm B | 42 of 60, 70.0% | 43 of 60 | 49 of 60 |
| S4 arm C | 48 of 60, 80.0% | 44 of 60 | 52 of 60 |

H7 passes by no margin, and it depends on one ruling. If a formula against a coin's own past is allowed to answer an idea about a standing level, S2 rises to 49 of 55 and the gap to 14.1 points [-27.7, -0.3], and H7 fails. It passes under the other rulings tried, and H9 fails under all of them. Half of arm B's rejections are a plain level for an idea about a change, such as the taker share for "suddenly rises above normal". That would fit a student who tests less, though the link isn't tested. The owner of the research system hasn't read the sample.

### The sealed exam

Every answer file was first scored on 2025 and gave back its verdicts and ICs exactly. The sealed period was then opened once per file, 8 openings in its access log (copied to `results/factor/sealed/access.log`), the first about 15 hours after prereg 03 was frozen. Passing doesn't change, since the gates read data through 2024.

| Model | Passed | IC among passes, 2025 | IC among passes, 2026 |
|---|---|---|---|
| S1 | 51.6% | +0.0015 | +0.0025 [-0.0006, +0.0060] |
| S4 arm A | 71.4% | +0.0012 | +0.0009 [-0.0010, +0.0026] |
| S4 arm B | 68.5% | +0.0008 | +0.0012 [-0.0008, +0.0032] |
| S4 arm C | 65.7% | +0.0004 | +0.0014 [-0.0009, +0.0037] |
| DeepSeek | 79.0% | +0.0042 | +0.0050 [+0.0025, +0.0075] |

This table takes each idea's mean first, as the prereg says, so it differs a little from the per-episode means above.

On a year nothing here was chosen on, the teacher's formulas keep an incremental IC with an interval above 0. So these ideas hold signal and the measure can see it, and the student's flat numbers mean it didn't learn to find it. S4 arm B against the teacher is -0.0038 [-0.0068, -0.0009]. RL shows no sign of overfitting its reward year, though there was little to lose. Reward holds up, since passing doesn't depend on the year: S4 arm B minus S1, +0.108 [+0.037, +0.177].

### After the exam, as exploration

Everything in this subsection came after the exam, and the 2026 numbers were read knowing the results above. The sealed panel wasn't reopened: they come from the exam's own files. `scripts/factor/explore_after_exam.py` prints them, and `results/factor/explore/` holds its output.

**The reward could hardly say anything about IC.** Within a group, 97% of arm B's reward spread comes from passing or failing and 2.8% from IC among passes. The reward maps IC onto a range of ±0.03, and passes in a group typically differ by about 0.003. That scaling is our fault. IC on training ideas didn't rise during RL, -0.0004 [-0.0010, +0.0000] in arm B. So H6 shows that this reward taught no more signal than the gates did, and says little about whether a reward can.

**S2's formulas can't be told from the teacher's, and the fall comes with offline RL.** Per idea on held-out ideas, scored on 2025, S2 minus the teacher is -0.0001 [-0.0023, +0.0022], and on val +0.0001 [-0.0018, +0.0025]. S3 arm B is -0.0033 [-0.0059, -0.0010] and S4 arm B -0.0031 [-0.0051, -0.0010]. The gap appears on training ideas too, so it isn't about transfer. IC falls in offline RL, with tests near 4. Online RL then cuts tests from 4.00 to 2.63 and leaves IC flat. From S2 to S4 arm B the pass rate rises through two repairs, fewer episodes with no answer, 20% to 2%, and fewer in the wrong direction, 16% to 6%. The first came almost all in offline RL, which cut no-answer episodes to 4% and raised passes 15.3 points [+5.7, +24.2]; online RL added 8.9 [-2.0, +20.2], mostly by cutting wrong directions from 14%. On val and on the second exam's new ideas the split runs the other way: S2, S3 arm B and S4 arm B pass 71.7%, 73.3% and 79.2% on val, and 46.1%, 52.4% and 65.2% on the new ideas. These intervals are wide, and S2 passes the fewest, which may flatter its IC.

**The student already writes good formulas, and doesn't pick them.** Among one model's passes on one idea, research IC correlates +0.69 to +0.84 with 2026. So draw k episodes, run the gates, and keep the answer with the highest research IC:

| Process, held-out ideas | Tests | Passed | Judged faithful | IC on 2026 | Reward on 2026 |
|---|---|---|---|---|---|
| S4 arm B, one episode | 2.6 | 68.5% | 69% | +0.0012 [-0.0009, +0.0033] | 0.455 |
| S4 arm B, k=4, gates | 10.5 | 90.7% | 65% | +0.0040 [+0.0015, +0.0065] | 0.634 |
| S4 arm B, k=4, gates and judge | 10.5 | 90.7% | 81% | +0.0029 [+0.0004, +0.0052] | 0.622 |
| S4 arm B, k=8, gates and judge | 21.1 | 96.8% | 83% | +0.0044 [+0.0015, +0.0073] | 0.679 |
| DeepSeek, one episode | 4.5 | 79.0% | | +0.0050 [+0.0025, +0.0075] | 0.566 |
| DeepSeek, k=2, gates | 8.9 | 92.5% | | +0.0057 [+0.0028, +0.0084] | 0.668 |

At about 10 tests each, S4 arm B minus the teacher is -0.0017 [-0.0051, +0.0016] on IC and -0.033 [-0.121, +0.046] on reward. At one episode each it is -0.0038 [-0.0068, -0.0009] and -0.111 [-0.211, -0.013]. The rule has nothing to tune, each seed alone gains about the same when picking by research IC without the gates, +0.0028 and +0.0030 at k=4 against +0.0035 pooled, and val ideas agree on 2025. But it is a model and a process, not the model alone, and the teacher with the same process stays ahead on point estimates. The gates can't be skipped: by research IC alone, arm A at k=8 passes only 54.8%. Picking also costs faithfulness, 69% to 60% at k=8 by the judge, which is why a judge belongs in the process.

**What this suggested.** S2, from SFT on the best rows, wrote formulas level with the teacher's on 2025, though H3 hadn't shown them better than S1's, and offline RL lost that in all three arms. Both RL stages made the student more reliable: on these held-out ideas offline RL did more of it, on val online RL did. So online RL straight from S2 might keep S2's formulas and still gain the reliability. `prereg/03b-pilot.md` set out a pilot of that, with a better-scaled reward beside it, and its rules were public before either run started. A second exam on new ideas, `prereg/04-second-exam.md`, tested the readings. The next two subsections report both.

### Pilot 03b: online RL straight from S2

This is exploration too, but its rules were pushed before either run started (`prereg/03b-pilot.md`). Each run starts from S2 with one seed, trains 35 steps under prereg 03's guards, and then answers the 15 val ideas 16 times each. A run passes only if three things hold against S4 arm B. Its training IC among passes in steps 26-35, minus S4 arm B's, has a CI lower bound above 0. Its val IC among passes is above S4 arm B's. Its val pass rate is no more than 5 points below. `scripts/factor/pilot_compare.py` applies the rules (`results/factor/pilot/`). A recount from the raw traces that shares no code with it apart from the sandbox that scores a formula, `scripts/factor/recount_pilot.py` and `scripts/factor/recount_pilot_p2b.py`, gives the same point estimates, with intervals from its own draws (`results/factor/pilot/recount.txt`).

| Run | Reward | Training IC, steps 26-35, minus S4 arm B | Val IC minus S4 arm B | Val pass rate, S4 arm B 79.2% | Verdict |
|---|---|---|---|---|---|
| Run 1, `factor-p1-s2-ic` | Arm B's, unchanged | +0.0003 [-0.0003, +0.0008] | +0.0005 [-0.0006, +0.0016] | 76.2% | Fails rule 1 |
| Run 2, rerun as `factor-p2b-s2-rank` | Percentile among passes on the same idea, and an episode with no answer scored on its best test | +0.0012 [+0.0005, +0.0019] | +0.0009 [+0.0000, +0.0019] | 71.7% counting best tests, 69.2% on answers alone | Fails rule 3 |

**Skipping offline RL wasn't enough.** Run 1 showed S4's pattern again. It became more reliable: on training ideas, from steps 1-5 to 31-35, the pass rate rose from 48% to 71% and answers handed in from 81% to 97%. Its formulas didn't get better: its own IC moved -0.0005 [-0.0012, +0.0002] within the run.

**A reward that can hear IC moved IC a little, and cost passes.** Run 2's training IC ended +0.0009 [+0.0001, +0.0017] above run 1's. But it handed in fewer answers, 94.2% on val against run 1's 97.9%, and passed fewer. Two parts of the reward plausibly explain that. Paying an answer-less episode for its best test takes away part of the reason to hand one in. And when IC can be heard, passing alone earns less. Its own IC barely moved within the run, +0.0001 [-0.0008, +0.0011], so the reward may have kept S2's formulas from slipping rather than taught better ones. Each run has one seed, and the intervals are about as wide as the effects.

**Run 2 first trained on run 1's reward.** The framework rebuilds every task it hands out as the type named in the taskset's generic base, and the new taskset inherited its parent's. So every rollout of the first attempt was scored by arm B's reward, as its traces show. A new test now rebuilds a task the way the server does, and run 2 ran again under the same rules. The first attempt, `factor-p2-s2-rank`, is reported as a second run of run 1's recipe, and it landed close to run 1, +0.0002 [-0.0005, +0.0009] on training IC.

Neither run passed, so neither joined the second exam, and its E5 isn't tested.

### The second exam

`prereg/04-second-exam.md` tests four readings from the exploration on 21 new ideas in 7 families no model had read. Its rules were frozen and pushed before any model answered. The models keep their weights and answer 16 times per idea. They are scored on 2026, which prereg 03's exam had already read once, and the owner agreed to that second opening. Every file came back whole, gave back its own 2025 scores before 2026 was opened, and was opened once. A recount that shares no code with the exam's script reaches the same verdicts, with the same point estimates apart from E3's, which its own random draws move by 0.0001 (`results/factor/exam2/recount.txt`).

| ID | Reading | On 2026 | Verdict |
|---|---|---|---|
| E1 | S2 writes better formulas than online RL's model | S2 minus S4 arm B, IC among passes: +0.0020 [-0.0005, +0.0044] | Does not pass |
| E2 | Online RL is more reliable than S2 | S4 arm B minus S2, pass rate: +19.0 points [+11.6, +26.5] | Passes |
| E3 | Keeping one of four by research IC recovers signal | S4 arm B at k=4 minus one episode, IC among passes: +0.0041 [+0.0027, +0.0055] | Passes |
| E4 | Research IC guides later IC | Within one model's passes on one idea: correlation +0.852 [+0.766, +0.908] | Passes |

Three of the four held. E1 points the same way as before but its interval includes 0, so it stays a reading. Against the teacher, S2's IC among passes is level again, +0.0000 [-0.0033, +0.0033], and it passes 33.6 points fewer. At about the same number of tests, S4 arm B at k=4 minus the teacher at k=2 is +0.0021 [-0.0008, +0.0054] on IC and -6.0 points [-24.5, +11.6] on passes. Within these wide intervals the student with the process can't be told from the teacher with it.

One thing got worse on new families. The judge finds 47.5% to 58.7% of the students' passes faithful, against 64% to 78% on prereg 03's held-out ideas, both averaged per idea, unlike the pooled counts in the held-out table above. A process for new families needs the judge in it. The numbers are in `results/factor/exam2/comparisons.txt`.

### What three chapters say together

The same thing happened three times, in three forms. Reward rises by the cheapest path the reward can see, and what it can't see is left alone or lost. Qwen could use the tools, so it read better sections. OLMo couldn't, so it answered from memory and dropped a step the judge didn't pay for. On factors the student stopped failing and handed in plainer formulas after fewer tests. In chapter 3 market data shows the cost, since it doesn't care how a formula was written: on held-out ideas offline RL lowered incremental IC on 2025 in every arm, and on 2026 the final model stayed below the teacher there, though on the second exam's new ideas that gap was within noise. A verifier's pass rate is not evidence of the skill it was built to check, and the share of a reward's spread that carries the skill is worth measuring before a run starts.

## 8. Cost on 4× RTX 4090

| Item | Qwen3-4B | OLMo 3 7B |
|---|---|---|
| Pilot | 8.5 minutes for 2 steps | 18 minutes for 2 steps; a first try with one trainer GPU ran out of memory while loading |
| Training, 35 steps | 72 minutes, about 2 minutes per step | 4 h 36 min, about 6 minutes per step plus four online evals of 10 to 28 minutes |
| Trainer peak memory | 18.2 GiB on 1 GPU | 18.3 GiB on each of 2 GPUs |
| Samples per episode in the trainer | 1.0 | 4.6 to 5.4 |
| Evaluation | 16 minutes, 11,632 episodes | 2 h 33 min, 18,072 episodes, slowed by the judge outage |
| Judge tokens, v4-pro evaluation | about 3.7M | about 5.7M, plus the retried calls |

OLMo's 8-rollout difficulty filter over the 1,771 training questions took 55 minutes on four GPUs. OLMo is slower for four reasons: 7.3B against 4B parameters, no grouped-query attention (512 KiB of KV cache per token against 144 KiB), two inference GPUs instead of three, and a renderer that renders the whole conversation again each turn, so each turn becomes its own training sample.

For chapter 3 on OLMo 3 7B: S1 at rank 128 took 1 h 52 min on four GPUs, S2 about 12 minutes on four, S3's versions on positive rows 43 to 93 minutes on two, and each online RL run 1.8 to 1.9 hours on four. An eval of 124 episodes takes about 4 minutes on two GPUs.

## 9. Pitfalls

The first five are in `results/smoke-test.md`: stalled GitHub downloads during install, `vllm-router` missing from the installed extras, port 8000 taken on a shared server, vLLM refusing Qwen's 262k default context, and the wiki tool server's ONNX embedder taking every CPU core. Later ones:

- **The question bank is almost deterministic for the base model.** Without the difficulty filter, nine in ten training questions would cost rollouts and give no gradient.
- **Judge outages drop rollouts, safely, during training.** DeepSeek connection errors fail the rollout, and prime-rl leaves failed rollouts out of the group baseline, so they are lost data, not zero rewards.
- **Judge outages during evaluation lose whole stretches of questions.** The rollouts survive in the traces, so the failed calls can be retried with the same judge; report both versions.
- **The router answers `/v1/models` with a 500 under data parallelism.** Requests route fine; ask the engine behind it instead (`scripts/load_adapter.sh`).
- **A local model path needs the renderer named.** Without `orchestrator.renderer.name = "qwen3"` the default renderer has no tool support.
- **prime-rl has no OLMo renderer.** The default renderer with a ported tool parser works; check token-level parity with the eval path before training (`scripts/olmo3_parity.py`).
- **The trainer keeps fp32 weights.** A 7B model needs two trainer GPUs on 24 GB cards.
- **Stop rules are model-specific.** OLMo's entropy at the training temperature is ten times Qwen's; a band copied from Qwen would have stopped a healthy run at step 1.
- **Negative weights in an SFT trainer have no floor.** Cross-entropy times a negative weight falls without limit, and failing and passing trajectories share nearly all their tokens. Offline RL's first version broke the model's syntax this way, with entropy and KL looking fine, so online RL also got a guard on reward. prime-rl's online loss has its own limits, and online RL didn't do this.
- **Scale a reward's parts against each other before a run.** Arm B's IC term spans 0.7 of reward over ±0.03, and passes in a group differ by about 0.003. Nothing in a training curve shows that. A few minutes on pilot rollouts would have.
- **A check at an exact limit needs exact arithmetic.** 42/60 minus 44/55 is exactly -1/10, and in floats just under it, so the audit's scorer first printed "fails". Found on reading its output, and fixed with fractions.
- **Check a plug-in the way the runtime builds it.** The verifiers framework rebuilds every task it hands out as the type named in the taskset's generic base, not as the type `load` returns. A taskset that only overrode `load` was scored by its parent's reward for a whole run. A test that loaded the taskset passed. A test that rebuilds a task as the framework's `verifiers/v1/serve/server.py` does would have failed, and now does on the old code.
- **Gates limit which ideas can be asked.** Eight ideas drafted for the second exam had no formula the gates could judge: a shock moves a correlation either way, for instance. So a bank built this way leans toward what its gates can test.

## 10. Limitations and next steps

Chapter 3:

- **One task, one model, one author.** These results say nothing general about RL. The verifier's author, Claude, wrote the bank, the gates, the rubric and the audit's rulings, and read the audit, and nobody independent has checked them. H7 turns on one of those rulings.
- **Most of chapter 3's explanation came after its exam.** The pre-registered result is that no stage was shown to teach better formulas. The reasons offered were readings. A second exam then tested four of them on new ideas, and three held. Its ideas came from the same author after the first exam, and it reused 2026.
- **"Unseen" has limits.** Held-out families share data fields with training families, and the reference formulas of 9 held-out ideas appear verbatim among the tests in 21 of the 2,021 teacher trajectories S1 trained on, 3 of them as the answer (prereg 03).
- **The reward was mis-scaled.** So chapter 3 doesn't show that a reward can't teach IC. In a one-seed pilot from S2, a better-scaled reward ended 0.0009 above the old one on training IC and passed fewer val episodes, 69.2% against 76.2% counting answers alone. What to solve next is that trade-off, not the scale alone.
- **Two seeds, 35 steps.** Online RL's arms differ by about as much as its seeds do on val.
- **Selection bias.** Some ideas come from a research system whose factors were chosen on data through 2025, so 2025 IC may lean optimistic. 2026 was the clean year, and it has now been read twice.

Chapters 1 and 2:

- **One seed and one budget per model.** The published run used about 23 times as many rollouts as either of ours. A longer run and a second seed would show whether the held-out gains keep growing and whether Qwen learns to read further.
- **The judge sees only the final reply.** For Qwen this mattered little; for OLMo most of the gain is answers from memory. A reward that requires the answer to appear in retrieved text, or that credits a successful read, would separate retrieval from memory. This is the most direct next experiment on wiki-search.
- **The trajectories were read by Claude.** The episodes read for chapters 1 and 2 were labelled by Claude, not by a person, as the label files say and the preregs' Deviations now record.
- **OLMo's remaining failures are teachable.** Missing page prefixes, a tool name that does not exist, and skipped section listings are what SFT on correct tool-use trajectories before RL would target.
- **Long tool outputs garble OLMo's replies.** Capping section length in the environment may matter more than more RL steps.
- **Never-right training questions give no signal**: 592 for Qwen, 1,277 for OLMo. A curriculum, hints, or credit for reaching the right section could open them.
- **Throughput.** The trainer sets the pace. Two trainer GPUs holding replicas of the LoRA model, rather than shards, might roughly halve Qwen's step time on a machine without peer-to-peer.
