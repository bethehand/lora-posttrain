# Post-training small models with LoRA on 4× RTX 4090

*Chapter 1: wiki-search. Draft for review, 2026-11-03.*

## Summary

Prime Intellect reported that LoRA RL lifts Qwen3-4B-Instruct-2507 from 0.55 to 0.89 on wiki-search, level with gpt-4.1 (0.88). The 100 questions behind those numbers are rows 0-99 of the question bank the model trained on. We split the bank by source page, froze four hypotheses in a pre-registration, and trained one LoRA adapter for 35 steps on four RTX 4090s in 72 minutes.

- **Held-out pages: a real but small gain.** On 490 questions from 88 pages never trained on, accuracy goes from 0.567 to 0.633, a gain of +0.066 (95% CI +0.042 to +0.090). H1 passes; H1b, a gain of at least 0.20, fails. The gain is about a fifth of the published one.
- **Trained questions: the published size.** On the 174 questions it trained on, accuracy goes from 0.483 to 0.853 (+0.371). Untrained questions gain about the same whether or not their page was seen in training (+0.058 against +0.066), so the large gain belongs to the questions, not the pages. Our trained questions are ones the base model sometimes answers; seven tenths of the published gain is on questions it never answers, where the published adapter, trained on them, scores 0.78 and ours 0.13.
- **The mechanism is section choice, not more reading.** The adapter does not read further into a page (0.94 to 0.99 section reads per episode; H4 fails its threshold of 1.5), but it reads the right section more often: wrong-section failures fall from 30.1% to 25.9% of episodes, two thirds of the drop in failures.
- **The gain survives the checks.** It is the same under the training judge (+0.069) and with a renamed tool argument (+0.065), it is not memory (+0.016 without tools), and 50 trajectories read by hand show no judge gaming.

## 1. Motivation

A recurring claim of 2025 is that a small open model, post-trained with RL on a narrow agentic task, matches frontier models on that task. Such claims are cheap to make and rarely checked end to end, because the check needs the training run, not just the eval. One machine with four consumer GPUs is enough to do it for one claim: audit the original evaluation, rerun the training on a clean split, and decide pre-registered hypotheses with paired statistics. This chapter does that for wiki-search and builds the pipeline the next chapters reuse.

## 2. Audit of the published numbers

- **The eval questions were training questions.** The published eval (verifiers baea257) scored rows 0-99 of `willcb/wiki-trivia-questions`, the file the RL run trained on.
- **The comparison with gpt-4.1 is noise.** On those 100 questions the RL model and gpt-4.1 disagree on 11, split 6 to 5 (exact McNemar p = 1.0).
- **The judge erred in one direction.** We re-graded all 700 published verdicts (7 models × 100 questions) with two DeepSeek judges. Agreement with the original gpt-4.1-mini judge is 95.7% for deepseek-flash (Cohen's kappa 0.872) and 96.0% for deepseek-v4-pro (0.882). Most disagreements are the original judge saying no where ours say yes; it scored gpt-4.1 at 0.88, both DeepSeek judges at 0.96 (`results/judge_agreement/`).

## 3. Setup

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

The design, hypotheses and thresholds are in `prereg/01-wiki-search.md`, frozen in commit 979250a before training started. The analysis code was committed before the evaluation results existed; the pre-registration's Deviations section gives the timing.

## 4. Results

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

The split of `train_sample` into trained and untrained questions is exploratory. On `slide100` our adapter scores 0.645; the published adapter, which trained on these questions, scores 0.89 on its published rollouts re-graded by v4-pro, and our adapter reaches 0.853 on the questions it trained on. The budgets differ (Section 7), but the pattern is the one the split predicts.

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

The same grouping splits the published gain on `slide100`. The published columns are the published rollouts, one per question with the published prompt, re-graded by v4-pro.

| `slide100` by our base score | Questions | Our base | Our adapter | Published base | Published adapter | Contribution to the published gain |
|---|---|---|---|---|---|---|
| 0 of 4 | 36 | 0.000 | 0.125 | 0.111 | 0.778 | +0.240 |
| 1 to 3 of 4 | 9 | 0.389 | 0.583 | 0.556 | 0.778 | +0.020 |
| 4 of 4 | 55 | 1.000 | 0.995 | 0.836 | 0.982 | +0.080 |

Seven tenths of the published gain, +0.24 of +0.34, is on the 36 questions our base model never answers. The published adapter trained on these questions and answers 0.78 of them; ours, which trained on no question its base model never answers, answers 0.13. Both adapter columns are measured apart from the base rollouts that form the groups. So the two gains on trained questions, +0.371 on `train_mixed` and the published +0.34, are not the same measurement: `train_mixed` holds only questions the base model sometimes answers, the easiest to move. Whether reaching the never-answered questions needs training on them, or only a larger budget, is what a longer run on all training questions would show. `scripts/by_base_score.py` makes both tables (`results/by_base_score.md`).

The navigation metrics (`scripts/navigation_metrics.py`) say how. An episode still makes about three calls, search, list sections, read one section: 2.98 calls per episode before and 3.12 after. What changes is which section. The share of episodes that read the source page, never see the answer in any tool output, and fail falls from 30.1% to 25.9%. The published adapter, trained on about 23 times as many rollouts, reads 1.85 sections per episode in its published rollouts; ours reads 0.99. Whether a longer run would also learn to read further is open.

### Checks

- **Judge.** Re-graded by deepseek-flash, the training judge, the held-out gain is +0.069 [+0.046, +0.093], and the two judges agree on 98.8% of episodes. A policy that had learned to please the training judge would gain more under it.
- **Tool interface.** With the read tool's argument renamed from `section_id` to `section`, the gain is +0.065 [+0.040, +0.091], and each model's score moves by at most 0.002.
- **Memory.** Without tools the adapter gains +0.016, so the gain needs the tools.
- **Trajectories.** In 50 random held-out episodes of the adapter (`results/audit/rl35_heldout_labels.md`), none of the 28 correct replies names more than one candidate, and the two replies that hedge across candidates were judged wrong. In 21 of the 22 failures the answer never appears in any tool output: the model read one section, did not find it, and answered or denied the premise instead of reading on. Two of these are flawed questions about page metadata.

### Training

Training reward on `train_mixed` (deepseek-flash) averages 0.53 over steps 1-10 and 0.76 over steps 26-35. Online held-out evals during training (flash, 1 rollout per question, exploratory) read 0.569, 0.596, 0.637 and 0.643 at steps 0, 15, 30 and 35. The pre-registered instability guard did not trigger: mismatch KL stayed between 0.0006 and 0.0038 and entropy between 0.047 and 0.179 (`results/train/prereg01/steps.txt`).

### A second base model: OLMo 3 7B Instruct

OLMo 3 7B Instruct scores 0.020 with tools on `heldout` and 0.252 without. 80% of its episodes with tools end in a tool call the parser cannot read, such as `wiki_read_section(section_ "...")`. The fault is in the model, not the parser: the same malformed call appears in 11 of 20 raw greedy samples. Renaming the argument to `section` raises well-formed calls from 22% to 70% in a probe and held-out accuracy to 0.094 (`results/olmo-tool-call-diagnosis/`). AI2 reports OLMo 3 level with Qwen3-8B on SimpleQA with its own tool harness, so the gap is robustness across harnesses. We did not train OLMo on wiki-search.

## 5. Cost on 4× RTX 4090

| Item | Value |
|---|---|
| Training | 72 minutes for 35 steps, about 2 minutes per step. The trainer GPU is the bottleneck (peak 18.2 GiB at 8,192 tokens) while the three inference GPUs often wait |
| Pilot | 8.5 minutes for 2 steps, about 2 of them start-up |
| Evaluation | 16 minutes for base and adapter on four question sets, 11,632 episodes |
| Judge | about 3.7M tokens for the v4-pro evaluation, and the same number of calls again for the flash re-grading |

## 6. Pitfalls

The first five are in `results/smoke-test.md`: stalled GitHub downloads during install, `vllm-router` missing from the installed extras, port 8000 taken on a shared server, vLLM refusing Qwen's 262k default context, and the wiki tool server's ONNX embedder taking every CPU core. Later ones:

- **The question bank is almost deterministic for the base model.** Without the difficulty filter, nine in ten training questions would cost rollouts and give no gradient.
- **Judge outages drop rollouts, safely.** DeepSeek connection errors fail the rollout, and prime-rl leaves failed rollouts out of the group baseline, so they are lost data, not zero rewards.
- **The router answers `/v1/models` with a 500 under data parallelism.** Requests route fine; ask the engine behind it instead (`scripts/load_adapter.sh`).
- **A local model path needs the renderer named.** Without `orchestrator.renderer.name = "qwen3"` the default renderer has no tool support.

## 7. Limitations and next steps

- **One seed and one budget.** The published run used about 23 times as many rollouts and 6 times as many steps. A longer run and a second seed would show whether the held-out gain keeps growing and whether reading further appears.
- **592 never-right training questions give no signal.** A curriculum, hints, or credit for reaching the right section could open them; they are exactly the wrong-section failures the adapter still makes.
- **The judge sees only the final reply.** Answers from memory count. Here memory moved little (+0.016 without tools), but a task where it matters needs a judge that checks the evidence.
- **Throughput.** The trainer GPU sets the pace. Two trainer GPUs holding replicas of the LoRA model, rather than shards, might roughly halve the step time on a machine without peer-to-peer.
