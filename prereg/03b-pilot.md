# Pilot 03b: online RL straight from S2

**Status: written and pushed before either run starts.** This comes after prereg 03's exam and changes none of its results. It reads no held-out idea and no sealed data. It was designed after seeing those results, so it is exploration. Whatever it suggests is tested later on ideas no model has read, under its own prereg.

The owner agreed on 2026-11-07 to skip offline RL for this follow-up only. It departs from the four stages of prereg 03, which stay reported as they ran.

## Why

Reading the exam's answers afterwards (`results/factor/week2.md`, "After the exam"):

- S2's IC among passes can't be told from the teacher's: per idea, -0.0001 (95% CI -0.0023 to +0.0022) on held-out ideas and +0.0001 (-0.0018 to +0.0025) on val. S3 arm B and S4 arm B are below it, -0.0033 and -0.0031 on held-out ideas, with intervals below 0.
- That fall comes with offline RL, +0.0034 to +0.0010 on held-out ideas, and online RL leaves it flat. Online RL's gain in pass rate from S2 is mostly fewer episodes with no answer, 20% to 2%, and fewer in the wrong direction, 16% to 6%.
- In arm B's reward, IC is 3% of the spread within a group, so H6 was a weak test of whether a reward can teach IC at all.

The first reason suggests that online RL from S2 may keep S2's formulas and gain its reliability. The second suggests a reward whose IC can be heard. On val ideas alone S2 also has the highest IC among passes and the largest gain from first test to answer, so the choice of start doesn't rest on held-out ideas.

## Runs

| Run | Start | Reward | No answer |
|---|---|---|---|
| `factor-p1-s2-ic` | S2, `factor-s2` | Arm B's, unchanged, `configs/factor-idea/rl.toml` | Earns 0, as in prereg 03 |
| `factor-p2-s2-rank` | S2, `factor-s2` | A pass earns 0.1 + 0.9 x its percentile among passes on the same idea | Scored on the formula the episode tested best on the research window |

The percentile table comes from every passing rollout on training ideas in the six S4 runs, on 2025: `envs/factor_idea_rank/factor_idea_rank/ic_table.json`, sha256 `d2f4d8cd9bda07e87c0928dc6b7408bc019cb8f2acfe15526ad256a1fe1d00b1`, 17,683 passes on 69 ideas, at least 31 an idea. Tried on arm B's own rollouts with a table from arms A and C, it makes IC about 30% to 36% of the spread within a group. The rule for an episode with no answer is applied only in the training reward: the model sees the same prompts and tool results as before.

Everything else is `configs/factor-idea/rl.toml`: 35 steps, 128 rollouts a step in groups of 8, lr 3e-5, rank 32, the guards of prereg 03. Each run has one seed, 1. `scripts/factor/run_pilot.py` runs them in turn, and `configs/factor-idea/rl-rank.toml` differs from `rl.toml` only in the training source's taskset.

Each kept adapter then answers the val ideas at eval settings, 16 rollouts per idea, 240 episodes, under the unchanged environment, and the judge reads every pass.

## What decides

Each run is compared with S4 arm B, both seeds pooled. It passes if all three hold:

1. **Training ideas, steps 26 to 35, all finished rollouts that handed in an answer:** IC among passes minus S4 arm B's, paired per idea, 95% CI lower bound above 0.
2. **Val ideas, answers it handed in:** IC among passes, averaged per idea, above S4 arm B's.
3. **Val ideas:** pass rate no more than 5 points below S4 arm B's. For `factor-p2-s2-rank`, an episode with no answer is scored on its best test here, as in training, and the rate without that is reported beside it.

`scripts/factor/pilot_compare.py` applies these, and is committed before either run's results are read.

Also reported, deciding nothing: run 2 minus run 1 on rule 1, the change in training IC from steps 1-10 to 26-35, tests per episode, answers handed in, wrong directions, and the judge's faithful share on val.

## What follows

A run that passes joins the second exam, on ideas no model has read. If neither passes, online RL from S2 goes no further, and the second exam still runs on the models that exist. Either way all of this is reported.

A pass here is a reason to go on and nothing more: one seed, 35 steps, and 15 val ideas. A failure says this recipe didn't do it in 35 steps, not that RL can't.

## Deviations

- **Run 2 trained on run 1's reward.** The framework rebuilds every task it hands out as the type named in the taskset's generic base (`Taskset.task_type()`), not as the type `load` returns. `FactorIdeaRankTaskset` inherited factor-idea's base, so every rollout of `factor-p2-s2-rank` was scored by factor-idea's task: arm B's linear reward, and 0 for an episode with no answer. Its traces show it. The task type is `FactorIdeaTask` in every rollout, the reward equals `reward_ic` in all 128 rollouts of step 5, and `reward_rank` was never recorded. My check before the run loaded the taskset and found the new reward, but it didn't rebuild a task the way the server does.
- **The fix.** The taskset now names its own task type. A new test rebuilds a task as `serve/server.py` does; it fails on the old code and passes on the new. Four episodes through the real framework, with DeepSeek on two training ideas, were scored by the new reward. Arms A, B and C of prereg 03 set their reward with a field of the same task, not a new class, and their traces show the reward each arm meant.
- **Run 2 runs again** as `factor-p2b-s2-rank`, with the same start, seed, config, table and rules. Rule 3's best test applies to it in place of `factor-p2-s2-rank`. When it started, the rules had already been applied to the other two runs, and neither passes. Run 1's rule 1 is +0.0003 (−0.0003 to +0.0008). The run that went wrong gets +0.0004 (−0.0002 to +0.0011) on rule 1 and −0.0000 on rule 2. The rerun is judged against S4 arm B, as before, not against them. `factor-p2-s2-rank` is reported as a second run of run 1's recipe, and it decides nothing.
