# Prereg 02 tests

Base `prereg02-base-rescored`, trained `prereg02-rl35-rescored`; `python3 scripts/prereg02_tests.py`.

| ID | Quantity | Estimate | 95% CI | Criterion | Verdict |
|---|---|---|---|---|---|
| H1 | heldout gain | +0.101 | [+0.079, +0.122] | CI lower bound > 0 | pass |
| H2 | base heldout accuracy, this session | 0.080 |  | 0.06-0.13; 0.094 before the freeze | pass |
| H3 | train_sample gain minus heldout gain | +0.008 | [-0.027, +0.043] | <= 0.10 | pass |
| H4 | call-failure share change | -0.182 | [-0.214, -0.151] | CI upper bound < 0 | pass |
| H5 | heldout_mixed_olmo gain (97 untrained questions of the trained kind) | +0.188 | [+0.126, +0.250] | CI lower bound > 0 | pass |
| expl. | train_mixed_olmo gain (trained questions) | +0.277 | [+0.246, +0.308] |  |  |
| expl. | heldout_mixed_olmo gain / train_mixed_olmo gain (Qwen 0.56) | +0.679 | [+0.448, +0.922] |  |  |
| H6 | heldout gain, section_id environment | +0.052 | [+0.038, +0.067] | CI lower bound > 0 | pass |
| expl. | heldout-notools gain | -0.015 | [-0.029, -0.002] |  |  |
| expl. | heldout gain judged by deepseek-flash | +0.107 | [+0.084, +0.130] |  |  |
|  | base heldout accuracy | 0.080 |  |  |  |
|  | trained heldout accuracy | 0.180 |  |  |  |
