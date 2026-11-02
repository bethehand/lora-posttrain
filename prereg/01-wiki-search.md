# Pre-registration 01: wiki-search on a held-out split

**Status: FROZEN on 2026-11-03, before the main training run started.** The commit that adds this text is the freeze; the sha256 values below pin the data split and the configs. Anything decided after that commit is reported as exploratory, and changes are listed under Deviations.

## Question

Prime Intellect reported that LoRA RL takes Qwen3-4B-Instruct-2507 from 0.55 to 0.89 on wiki-search, level with gpt-4.1 (0.88). Those 100 eval questions were rows 0-99 of the training question bank. Does the gain hold on questions from Wikipedia pages the model never trained on, and how large is it?

## Fixed design

| Item | Value |
|---|---|
| Base model | Qwen/Qwen3-4B-Instruct-2507 @ cdbee75, weights checked by sha256 |
| Environment | wiki-search tools, prompt and judge prompt from verifiers b2e4e81. Its system prompt names the three tools and how to use them; the published eval used "Use the provided Wikipedia search tools to help answer questions." A more explicit prompt can raise the base score, so it is the first suspect if H2 fails high |
| Questions | willcb/wiki-trivia-questions @ bd21382, split by source page: train 1,771 (331 pages), heldout 490 (88 pages) |
| Training questions | `train_mixed`: the 174 training questions the base model answered right in 1 to 7 of 8 rollouts (judge deepseek-flash, `results/evals/qwen3-4b-base-filter`). The other 1,597 are always or never right, so they give zero advantage at the start |
| splits.json sha256 | `87fbdbc76a6250bb52bb45119582f4ab0a3632c24bae15f88bf139e517458484` |
| Judge | Training reward: `deepseek-flash`. Evaluation: `deepseek-v4-pro` (primary) and `deepseek-flash` (secondary), so a policy that learns to please the training judge shows up as a gap between the two. Agreement with the 700 published gpt-4.1-mini verdicts: flash 95.7%, kappa 0.872; v4-pro 96.0%, kappa 0.882 (`results/judge_agreement/`) |
| Trainer | prime-rl v0.9.0 (ab5de8f), `configs/wiki-search/rl.toml` run by `scripts/rl_train.sh` on 3 inference GPUs and 1 trainer GPU |
| rl.toml sha256 | `6a10ab36a9d5d80b482187f4708d3d6b16502274068265a8034cd36d8dd3ffe0` |
| LoRA | rank 8, alpha 32, on q, k, v, o, gate, up and down projections; 16.5M trainable parameters |
| Optimization | AdamW, lr 3e-5 constant, weight decay 0, gradient norm clipped at 1.0; GRPO advantages (reward minus group mean, no length penalty); prime-rl's default loss |
| Batch and steps | group 8 x 16 questions = 128 rollouts per step, 35 steps, so each training question is drawn about 3 times; asynchronous with prime-rl's defaults, a batch ships once inference runs the policy from two steps earlier |
| Lengths | 512 tokens per turn, 8,192 tokens per episode |
| Model evaluated | the adapter after step 35 |
| Training seeds | 1 |

**Why the recipe changed.** The starting point is prime-rl's `examples/basic/wiki-search/rl.toml` (8 GPUs, batch 512, group 16, 200 steps, lr 1e-5, seq_len 4096). Every change was decided before any training result:

- `train_mixed` instead of all training questions: only questions with mixed outcomes carry a learning signal.
- lr 3e-5: Adam moves each weight by at most about lr per step, so 35 steps at 1e-5 would move the adapter about a sixth as far as the recipe's 200 steps, with LoRA's B starting at zero.
- seq_len 8,192: the base model reads one section and stops (`results/audit`), and longer reads must not be cut off at zero reward.

A 2-step pilot at lr 1e-5 checked memory and time only: trainer peak 18.2 GiB at 8,192 tokens, about 2 minutes per step with the trainer as the bottleneck, no rollout errors in the trained batches. Its rewards were not used for any decision.

## Evaluation

- **When and how:** after training, base and trained models are evaluated in the same session from one vLLM server that has the adapter loaded, so serving differences cannot pass for a training effect. The earlier base results in `results/evals/qwen3-4b-base` were the pre-training check for H2 and are not reused for H1.
- **Sets:** `heldout` (490 questions); `slide100` (rows 0-99 of heldout, the published 100); `train_sample` (300 training questions, seed 0); `train_mixed` (the 174 trained questions).
- **Sampling:** 4 rollouts per question, the model's generation_config defaults, 512 tokens per turn, up to 8,192 tokens per episode; `configs/wiki-search/eval.toml` run by `scripts/eval_suite.sh`.
- **eval.toml sha256:** `86115ff4f45dd3184984055d16ae0085537b14974ee9d4f4ccf267dd850715df`
- **Metric:** mean judge reward per question, averaged over questions; 95% CI from a paired bootstrap over questions (10,000 resamples). Rollouts whose judge call fails are dropped, and the failure rate is reported.

## Hypotheses and pass criteria

| ID | Hypothesis | Passes if |
|---|---|---|
| H1 (primary) | RL improves held-out accuracy | trained minus base on `heldout` has a 95% CI lower bound above 0 |
| H1b | The gain is as large as the published one | trained minus base on `heldout` is at least 0.20 |
| H2 | Our setup reproduces the published base score | base on `slide100` is within 0.45-0.65. **Passed before the freeze: 0.568** |
| H3 | The gain does not depend on having trained on the page | gain on `train_sample` minus gain on `heldout` is at most 0.10 |
| H4 | RL teaches the model to read further into a page | on `heldout`: the trained model averages more than 1.5 `wiki_read_section` calls per episode, with the CI of trained minus base above 0; and the share of wrong-section failures falls, with the CI of trained minus base below 0 |

A wrong-section failure is an episode with reward 0 that read at least one section of the question's source page while none of its tool outputs contains the gold answer, compared after lowercasing and removing punctuation (so spelling variants count as misses). It is the automatic version of the label that covered 21 of 30 never-right training questions in `results/audit`.

All results are reported whether or not they pass.

## Exploratory, reported as such

- Held-out evals during training at steps 0, 15 and 30 (judge deepseek-flash, 1 rollout per question).
- The gain on `train_mixed`, the questions actually trained on.
- All eval rollouts re-judged with deepseek-flash: a gain that is larger under the training judge than under v4-pro suggests judge gaming.
- Tool calls per episode, rate of answering without any tool call, final-answer length.
- No-tools baseline (`heldout-notools`) for base and trained models: how much of the score is memory.
- The trained model in the renamed environment (`read_arg = section`).
- 50 random trained-model trajectories read by hand for judge gaming, such as listing several candidate answers.
- Anything trained past step 35.

## Stop rules

- Done before the freeze: judge agreement kappa 0.872 and 0.882, above the required 0.7.
- **Instability guard, steps 1 to 10.** Stop the run if `mismatch_kl/all/mean` is above 0.01 for two consecutive steps, or `entropy/all/mean` is below 0.02 or above 0.5 for two consecutive steps (prime-rl's `metrics.jsonl`). Then rerun from scratch with lr 1e-5 and nothing else changed; the rerun becomes the primary run and both are reported. Pilot values at lr 1e-5: mismatch KL 0.0010 and 0.0013, entropy 0.071 and 0.126. After step 10 no training metric stops the run.
- If the run crashes, it is restarted from scratch with the same config, and the crash is reported.

## Deviations

None yet.
