# Held-out accuracy by the base model's score on each question

Base `prereg02-base` and adapter `prereg02-rl35`, judge deepseek-v4-pro, 4 rollouts per question. Written by `scripts/by_base_score.py`. Exploratory.

## heldout

| Base on the question | Questions | base | adapter | Contribution to adapter minus base |
|---|---|---|---|---|
| 0 of 4 | 219 | 0.000 | 0.104 | +0.082 |
| 1 to 3 of 4 | 54 | 0.414 | 0.491 | +0.015 |
| 4 of 4 | 2 | 1.000 | 0.625 | -0.003 |
| all | 275 | 0.088 | 0.183 | +0.095 |

## heldout grouped by the earlier base eval `olmo3-7b-base-renamed`

| Group | Questions | Base | Adapter | Gain | `train_mixed_olmo` gain minus this, 95% CI |
|---|---|---|---|---|---|
| `train_mixed_olmo`, trained | 489 | 0.263 | 0.540 | +0.277 | |
| 0 of 4 | 209 | 0.027 | 0.091 | +0.064 | +0.213 [+0.173, +0.254] |
| 1 to 3 of 4 | 60 | 0.250 | 0.461 | +0.211 | +0.066 [-0.031, +0.162] |
| 4 of 4 | 6 | 0.611 | 0.625 | +0.014 | +0.263 [-0.091, +0.658] |
