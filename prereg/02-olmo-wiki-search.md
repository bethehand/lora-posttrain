# Pre-registration 02: OLMo 3 7B on wiki-search with the recipe of prereg 01

**Status: FROZEN on 2026-11-03, before the main training run started.** The commit that adds this text is the freeze; the sha256 values below pin the data split, the configs and the tool parser. Anything decided after that commit is reported as exploratory, and changes are listed under Deviations.

## Question

Prereg 01 trained Qwen3-4B-Instruct-2507 with LoRA RL on wiki-search: +0.066 on held-out questions, +0.371 on the questions it trained on, and +0.207 on held-out questions of the trained kind, ones the base model sometimes answered (an analysis added after prereg 01). OLMo 3 7B Instruct fails in another way: even with the read tool's argument named `section`, 74% of its held-out episodes fail on a tool call that cannot be read or that returns an error. Does the same recipe improve OLMo's held-out accuracy, does it repair the tool calls, and does the gain reach questions OLMo did not train on?

## Fixed design

| Item | Value |
|---|---|
| Base model | allenai/Olmo-3-7B-Instruct @ 6e5971d |
| Environment | wiki-search-heldout as in prereg 01, with read_section's argument named `section` instead of `section_id`, in training and in every evaluation except `heldout-sectionid` |
| Questions | willcb/wiki-trivia-questions @ bd21382, the split of prereg 01: train 1,771 (331 pages), heldout 490 (88 pages) |
| Training questions | `train_mixed_olmo`: the 489 training questions OLMo answered right in 1 to 7 of 8 rollouts (judge deepseek-flash, `section` argument, `results/evals/olmo3-7b-base-filter`, accuracy 0.089); 5 always right, 1,277 never right. 52 of them are also in Qwen's `train_mixed`, 81 in `train_sample` |
| splits.json sha256 | `284ea0d9ffedcede11c5b0c24a523c9d4dd44974a38599eefa8b751c117826fd`; prereg 01's file plus `train_mixed_olmo` and its rule, the other keys unchanged |
| Judge | Training reward: `deepseek-flash`. Evaluation: `deepseek-v4-pro` (primary) and `deepseek-flash` (secondary), as in prereg 01 |
| Trainer | prime-rl v0.9.0 (ab5de8f), `configs/wiki-search/rl-olmo.toml` run by `scripts/rl_train.sh` on 2 inference GPUs and 2 trainer GPUs (FSDP) |
| rl-olmo.toml sha256 | `5b2ee1a24bb6053996ad9fdb88926602772f403f112582bd8cdba2bcfb0aa943` |
| Renderer | renderers' default renderer, which applies OLMo's chat template, with tool calls read by `setup/renderers_olmo3.py` (sha256 `b2b58261afdd5eeb7608eb31da0539a247704193ba7c04ce3f1abb341a842d14`, installed by `setup/install_olmo3_parser.sh`), a port of the vLLM olmo3 parser used at evaluation. On 300 prompts from OLMo's earlier eval, the training path's prompt tokens and parsed tool calls equal those of vLLM's chat endpoint in 300 of 300 (`results/olmo3-parity`) |
| LoRA | rank 8, alpha 32, on q, k, v, o, gate, up and down projections; 20.0M trainable parameters |
| Optimization | as prereg 01: AdamW, lr 3e-5 constant, weight decay 0, gradient norm clipped at 1.0; GRPO advantages; prime-rl's default loss |
| Batch and steps | as prereg 01: group 8 x 16 questions = 128 rollouts per step, 35 steps |
| Lengths | as prereg 01: 512 tokens per turn, 8,192 tokens per episode |
| Model evaluated | the adapter after step 35 |
| Training seeds | 1 |

**Why only these changes.** The point is to compare two models under one recipe, so the recipe of prereg 01 is kept and only what the model forces is changed:

- `section` instead of `section_id`: with `section_id`, OLMo writes `wiki_read_section(section_ "...")`, and 80% of its held-out episodes end on a call the parser cannot read (`results/olmo-tool-call-diagnosis`). The name does not matter to Qwen: 0.566 and 0.568 before training, +0.066 and +0.065 after. The rename removes a failure of one model at one argument name without changing the task, and `heldout-sectionid` measures whether training reaches the original name.
- `train_mixed_olmo` instead of `train_mixed`: the rule of prereg 01, only questions with mixed outcomes, applied to OLMo's own rollouts, and all of them are used. Qwen's 174 questions carry no signal for a model that answers 9% of questions. OLMo has 489 mixed questions to Qwen's 174, so with the same 35 steps of 16 questions each is trained on fewer times, about 1.1 against 3.4: in prereg 01, 167 of the 174 questions entered a trained batch. A random 174 would match that exposure but drop half of the signal; all of them are used, to see how far the same budget takes OLMo with more varied questions. H5 is therefore a comparison within OLMo, and its comparison with Qwen is descriptive.
- The renderer: renderers 0.1.10 has no OLMo renderer, so the default one renders with OLMo's own template, and the tool parser is ported from vLLM so that a call unreadable at evaluation is unreadable in training.
- Two GPUs serve the policy and two train, instead of three and one. The trainer keeps fp32 weights, 29 GB for OLMo's 7.3B parameters, which do not fit one 4090: the first pilot ran out of memory while loading them. Two FSDP shards fit. bf16 weights would round away most updates at lr 3e-5, and full optimizer offload turns gradient clipping off, so both would change the recipe. The batch, the steps and the asynchrony are unchanged.

A 2-step pilot of this config checked memory and time only: trainer peak 18.3 GiB per GPU; 6.3 minutes per trainer step against about 2.3 minutes to generate a batch, so the trainer sets the pace; 4.6 and 5.1 training samples per episode, because the default renderer renders the whole conversation again each turn and each turn becomes its own sample (Qwen's renderer gave 1.0); mismatch KL 0.0013 in both steps, entropy 1.14 and 1.57, grad norm 0.068 and 0.056. Its rewards were not used for any decision.

## Evaluation

- **When and how:** as in prereg 01. After training, base and trained models are evaluated in the same session from one vLLM server with the adapter loaded and the olmo3 tool parser; the earlier OLMo results in `results/evals/olmo3-7b-base*` are not reused for any test.
- **Sets:** `heldout` (490 questions), `train_sample` (300 training questions, seed 0), `train_mixed_olmo` (the trained questions) and `heldout-notools`, all with the `section` argument; `heldout-sectionid`, the held-out questions with `section_id`.
- **Sampling:** 4 rollouts per question, the model's generation_config defaults (temperature 0.6, top_p 0.95), 512 tokens per turn, up to 8,192 tokens per episode; `configs/wiki-search/eval.toml` (sha256 `86115ff4f45dd3184984055d16ae0085537b14974ee9d4f4ccf267dd850715df`, as in prereg 01) run by `scripts/eval_suite.sh` with `EVAL_EXTRA="--env.taskset.read-arg section"` and `TAGS="heldout train_sample train_mixed_olmo heldout-notools heldout-sectionid"`.
- **Metric:** as in prereg 01: mean judge reward per question, averaged over questions; 95% CIs from a paired bootstrap over questions (10,000 resamples); failed judge calls dropped and reported.

## Hypotheses and pass criteria

| ID | Hypothesis | Passes if |
|---|---|---|
| H1 (primary) | RL improves OLMo's held-out accuracy | trained minus base on `heldout` has a 95% CI lower bound above 0 |
| H2 | The setup reproduces OLMo's base score | base on `heldout` is within 0.06-0.13; it was 0.094 before the freeze |
| H3 | The gain does not depend on having trained on the page | gain on `train_sample` minus gain on `heldout` is at most 0.10 |
| H4 | RL repairs the tool calls | the share of call failures on `heldout`, trained minus base, has a 95% CI upper bound below 0 |
| H5 | The gain carries to untrained questions of the trained kind | on `heldout_mixed_olmo`, the held-out questions OLMo answered right in 1 to 3 of 4 rollouts in its earlier eval (`results/evals/olmo3-7b-base-renamed`, 97 questions), trained minus base has a 95% CI lower bound above 0. Reported with it: that gain divided by the gain on `train_mixed_olmo`, with a bootstrap CI; for Qwen the same ratio, computed after prereg 01, was 0.207 / 0.371 = 0.56 |
| H6 | The repair carries to the original argument name | trained minus base on `heldout-sectionid` has a 95% CI lower bound above 0 |

A call failure is an episode with reward 0 that ended on a tool call the parser could not read, its final reply still holding `<function_calls>`, or that got an error back from a tool (`Error executing tool ...` or `error: unknown tool ...`). `scripts/call_failures.py` computes it per question; on the earlier OLMo runs it gives 0.743 with `section` and 0.933 with `section_id`. `scripts/prereg02_tests.py` makes the table. Both were committed before any OLMo training result existed.

All results are reported whether or not they pass.

## Exploratory, reported as such

- Held-out evals during training at steps 0, 15, 30 and 35 (judge deepseek-flash, 1 rollout per question).
- All `heldout` rollouts re-judged with deepseek-flash, for judge gaming.
- Navigation metrics as in prereg 01, and accuracy by the base model's score per question.
- 50 random trained-model trajectories read by hand.
- The 100 published questions inside `heldout`.

## Stop rules

- **Instability guard, steps 1 to 10:** stop if `mismatch_kl/all/mean` is above 0.01 for two consecutive steps, or `entropy/all/mean` is below 0.3 or above 6.0 for two consecutive steps; then rerun from scratch with lr 1e-5 and nothing else changed. The KL threshold is prereg 01's. Its entropy band, 0.02 to 0.5, sat at about 0.28 and 4 times the lowest and highest values of its pilot (0.071, 0.126); OLMo samples a much flatter distribution at the training temperature of 1.0, and its pilot gave 1.14 and 1.57, outside that band from the first step. The band here uses the same factors on OLMo's pilot. After step 10 no training metric stops the run.
- If the run crashes, it is restarted from scratch with the same config, and the crash is reported.

## Deviations

None yet.
