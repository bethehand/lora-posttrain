# Prereg 01 tests

Base `prereg01-base`, trained `prereg01-rl35`; `python3 scripts/prereg01_tests.py`.

| ID | Quantity | Estimate | 95% CI | Criterion | Verdict |
|---|---|---|---|---|---|
| H1 | heldout gain | +0.066 | [+0.042, +0.090] | CI lower bound > 0 | pass |
| H1b | heldout gain | +0.066 | [+0.042, +0.090] | gain >= 0.20 | fail |
| H2 | base on slide100, this session | 0.585 |  | 0.45-0.65; passed before the freeze at 0.568 | pass |
| H3 | train_sample gain minus heldout gain | +0.027 | [-0.014, +0.069] | <= 0.10 | pass |
| H4 | trained section reads per episode | 0.985 |  | > 1.5 | fail |
| H4 | section reads gain | +0.048 | [+0.030, +0.068] | CI lower bound > 0 | pass |
| H4 | wrong-section failure share change | -0.042 | [-0.064, -0.020] | CI upper bound < 0 | pass |
| expl. | train_mixed gain (trained questions) | +0.371 | [+0.315, +0.427] |  |  |
| expl. | heldout-notools gain | +0.016 | [+0.002, +0.031] |  |  |
| expl. | heldout gain judged by deepseek-flash | +0.069 | [+0.046, +0.093] |  |  |
|  | base heldout accuracy | 0.567 |  |  |  |
|  | trained heldout accuracy | 0.633 |  |  |  |
