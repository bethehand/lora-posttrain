# Held-out accuracy by the base model's score on each question

Base `prereg01-base` and adapter `prereg01-rl35`, judge deepseek-v4-pro, 4 rollouts per question. Written by `scripts/by_base_score.py`. Exploratory.

## heldout

| Base on the question | Questions | base | adapter | Contribution to adapter minus base |
|---|---|---|---|---|
| 0 of 4 | 196 | 0.000 | 0.158 | +0.063 |
| 1 to 3 of 4 | 32 | 0.500 | 0.695 | +0.013 |
| 4 of 4 | 262 | 1.000 | 0.981 | -0.010 |
| all | 490 | 0.567 | 0.633 | +0.066 |

## slide100 with the published rollouts

| Base on the question | Questions | base | adapter | published base | published adapter | Contribution to published adapter minus published base |
|---|---|---|---|---|---|---|
| 0 of 4 | 36 | 0.000 | 0.125 | 0.111 | 0.778 | +0.240 |
| 1 to 3 of 4 | 9 | 0.389 | 0.583 | 0.556 | 0.778 | +0.020 |
| 4 of 4 | 55 | 1.000 | 0.995 | 0.836 | 0.982 | +0.080 |
| all | 100 | 0.585 | 0.645 | 0.550 | 0.890 | +0.340 |

## heldout grouped by the earlier base eval `qwen3-4b-base`

| Group | Questions | Base | Adapter | Gain | `train_mixed` gain minus this, 95% CI |
|---|---|---|---|---|---|
| `train_mixed`, trained | 174 | 0.483 | 0.853 | +0.371 | |
| 0 of 4 | 196 | 0.013 | 0.161 | +0.148 | +0.223 [+0.150, +0.294] |
| 1 to 3 of 4 | 29 | 0.448 | 0.655 | +0.207 | +0.164 [+0.006, +0.316] |
| 4 of 4 | 265 | 0.991 | 0.980 | -0.010 | +0.381 [+0.324, +0.438] |
