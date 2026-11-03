# Held-out accuracy by the base model's score on each question

Base `prereg02-base-rescored` and adapter `prereg02-rl35-rescored`, judge deepseek-v4-pro, 4 rollouts per question. Written by `scripts/by_base_score.py`. Exploratory.

## heldout

| Base on the question | Questions | base | adapter | Contribution to adapter minus base |
|---|---|---|---|---|
| 0 of 4 | 394 | 0.000 | 0.100 | +0.080 |
| 1 to 3 of 4 | 93 | 0.387 | 0.505 | +0.022 |
| 4 of 4 | 3 | 1.000 | 0.667 | -0.002 |
| all | 490 | 0.080 | 0.180 | +0.101 |

## heldout grouped by the earlier base eval `olmo3-7b-base-renamed`

| Group | Questions | Base | Adapter | Gain | `train_mixed_olmo` gain minus this, 95% CI |
|---|---|---|---|---|---|
| `train_mixed_olmo`, trained | 489 | 0.263 | 0.540 | +0.277 | |
| 0 of 4 | 383 | 0.023 | 0.102 | +0.079 | +0.198 [+0.161, +0.235] |
| 1 to 3 of 4 | 97 | 0.245 | 0.433 | +0.188 | +0.089 [+0.019, +0.161] |
| 4 of 4 | 10 | 0.625 | 0.700 | +0.075 | +0.202 [-0.048, +0.465] |
