# Pre-registration 01: wiki-search on a held-out split

**Status: DRAFT.** Frozen by committing it, with the sha256 of `splits.json` and `configs/wiki-search/rl.toml` filled in, before the main training run starts. Anything decided after that commit is reported as exploratory.

## Question

Prime Intellect reported that LoRA RL takes Qwen3-4B-Instruct-2507 from 0.55 to 0.89 on wiki-search, level with gpt-4.1 (0.88). Those 100 eval questions were rows 0-99 of the training question bank. Does the gain hold on questions from Wikipedia pages the model never trained on, and how large is it?

## Fixed design

| Item | Value |
|---|---|
| Base model | Qwen/Qwen3-4B-Instruct-2507 @ cdbee75, weights checked by sha256 |
| Environment | wiki-search tools, prompt and judge prompt from verifiers b2e4e81 |
| Questions | willcb/wiki-trivia-questions @ bd21382, split by source page: train 1,771 (331 pages), heldout 490 (88 pages) |
| splits.json sha256 | TBD at freeze |
| Judge | DeepSeek API, `deepseek-flash` or `deepseek-v4-pro` (the two models the API lists), whichever agrees better with the original gpt-4.1-mini verdicts |
| Trainer | prime-rl v0.9.0 (ab5de8f), LoRA rank 8 on all linear projections, lr 1e-5, group size 16, 512 tokens per turn, seq_len 4096 |
| Batch and steps | TBD from the pilot |
| Training seeds | 1 |

## Evaluation

- **Sets:** `heldout` (490 questions); `slide100` (the published 100, a subset of heldout); `train-sample` (300 training questions, fixed seed), for the train/held-out gap.
- **Sampling:** 4 rollouts per question, the model's generation_config defaults, 512 tokens per turn.
- **Metric:** mean judge reward per question, averaged over questions; 95% CI from a paired bootstrap over questions (10,000 resamples).

## Hypotheses and pass criteria

| ID | Hypothesis | Passes if |
|---|---|---|
| H1 (primary) | RL improves held-out accuracy | trained minus base on `heldout` is at least 0.20, and the CI lower bound is above 0.10 |
| H2 | Our setup reproduces the published base score | base on `slide100` is within 0.45-0.65 |
| H3 | The gain is not mostly memorization | gain on `train-sample` minus gain on `heldout` is at most 0.10 |

All results are reported whether or not they pass.

## Exploratory, reported as such

- Tool calls per episode, rate of answering without any tool call, final-answer length.
- 50 random trained-model trajectories read by hand for judge gaming, such as listing several candidate answers.

## Stop rules

- If judge agreement with the original verdicts has Cohen's kappa below 0.7, change the judge before training.
- If a pilot step takes over 10 minutes at batch 256, drop to batch 128 and record the change here before freezing.
