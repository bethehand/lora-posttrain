# Prereg 02 tests

Base `prereg02-base`, trained `prereg02-rl35`; `python3 scripts/prereg02_tests.py`.

| ID | Quantity | Estimate | 95% CI | Criterion | Verdict |
|---|---|---|---|---|---|
| H1 | heldout gain | +0.095 | [+0.065, +0.125] | CI lower bound > 0 | pass |
| H2 | base heldout accuracy, this session | 0.086 |  | 0.06-0.13; 0.094 before the freeze | pass |
| H3 | train_sample gain minus heldout gain | +0.018 | [-0.025, +0.059] | <= 0.10 | pass |
| H4 | call-failure share change | -0.175 | [-0.222, -0.127] | CI upper bound < 0 | pass |
| H5 | heldout_mixed_olmo gain (63 untrained questions of the trained kind) | +0.211 | [+0.121, +0.301] | CI lower bound > 0 | pass |
| expl. | train_mixed_olmo gain (trained questions) | +0.277 | [+0.246, +0.308] |  |  |
| expl. | heldout_mixed_olmo gain / train_mixed_olmo gain (Qwen 0.56) | +0.762 | [+0.430, +1.108] |  |  |
| H6 | heldout gain, section_id environment | +0.052 | [+0.038, +0.067] | CI lower bound > 0 | pass |
| expl. | heldout-notools gain | -0.015 | [-0.029, -0.001] |  |  |
| expl. | heldout gain judged by deepseek-flash | +0.099 | [+0.071, +0.129] |  |  |
|  | base heldout accuracy | 0.086 |  |  |  |
|  | trained heldout accuracy | 0.182 |  |  |  |
