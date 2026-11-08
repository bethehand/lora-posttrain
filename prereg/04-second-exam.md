# Pre-registration 04: a second exam, on ideas no model has read

**Status: FROZEN on 2026-11-08, before any model answered these ideas.** The commit that adds this text is the freeze, and the sha256 values under Pinned files pin what the exam runs. Nothing has read these ideas but the gates and the judge, on the reference formulas only, and no IC of theirs has been looked at, on any period. Pilot 03b has decided: no run passed it, so none joins, and E5 has nothing to test.

## Question

Prereg 03's exam and the reading after it (`results/factor/week2.md`, "After the exam") suggest four things: S2 writes better formulas than online RL's models, online RL is more reliable than S2, keeping one of several answers by research IC recovers signal, and research IC is a good guide to later IC. Those came from looking at the answers after the exam. This tests them on new ideas in new families, with the rule written first, and with nothing new to train.

## Design

| Item | Value |
|---|---|
| Ideas | `task/ideas_fresh.jsonl`, 21 ideas in 7 families absent from `task/ideas_v2.jsonl`. Each reference formula passes the gates and the judge. No IC of them has been looked at, on any period |
| Models | S1, S2, S3 arm B, S4 arms A, B and C with both seeds, and DeepSeek with tools and no manual, as in prereg 03. All keep their weights: none is trained on these ideas |
| Sampling | 16 rollouts per idea per model, 336 episodes. A model with two seeds gives 8 from each. Otherwise `configs/factor-idea/eval.toml`, temperature 0.6, top_p 0.95 |
| Gates | As frozen in prereg 03, on data through 2024 |
| Scored on | 2025, by the environment as each answer is given, and 2026-01-01 to 2026-10-16, by `rescore_eval.py --period sealed`, once per answer file. `scripts/factor/second_exam.py score` opens 2026 only when every file is whole and gives back its own verdicts and ICs on 2025 |
| Failed episodes | An episode that fails for a technical reason has no idea on it, and the whole file runs again, at most twice, before anything is read from it. If a file is still not whole, 2026 stays closed until the gap is reported here |
| Judge | `deepseek-flash` on every pass of every student model, as in prereg 03 |
| Statistics | Per idea, then averaged over ideas. 95% CIs from a paired bootstrap over ideas, 10,000 resamples. IC among passes uses ideas with a pass on both sides and says how many |

**Keeping one of k answers.** Draw k of a model's episodes on an idea at random, 2,000 draws per idea. Of those whose answer was tested and passes the gates, keep the one with the highest incremental IC in its own test report. If none, keep an answer at random, and if there is no answer the episode fails. The gates read data through 2024 and the test report stops at 2024, so a real process could do this. Whether the kept answer passes, and its IC, then come from the year being scored. One episode takes each episode once. The rule has nothing to tune.

## Hypotheses

All on 2026 unless stated, and all reported whether or not they pass, with 2025 beside them.

| ID | Hypothesis | Passes if |
|---|---|---|
| E1 | S2 writes better formulas than online RL's model | S2 minus S4 arm B, IC among passes, CI lower bound above 0 |
| E2 | Online RL is more reliable than S2 | S4 arm B minus S2, pass rate, CI lower bound above 0 |
| E3 | Keeping one of four by research IC recovers signal | S4 arm B at k=4 minus one episode, IC among passes, CI lower bound above 0 |
| E4 | Research IC guides later IC | Among one model's tested passes on one idea, pooled over every model in the exam with DeepSeek included and a model's seeds counted together, correlation of research IC with 2026 IC has a CI lower bound above 0.3, with whole ideas resampled |
| E5 | A run from S2 keeps both | Not tested: no run passed pilot 03b |

Reported beside them, deciding nothing: every model's pass rate, reward, IC and raw IC on both years; S2 and S4 arm B minus DeepSeek on IC among passes, pass rate and reward; the same process at k=2 and k=8 for S4 arm B, at k=2 and k=4 for S2 and DeepSeek; S4 arm B at k=4 against DeepSeek at k=2, and S2 at k=2 against DeepSeek at k=2, which spend about the same number of tests; the judge's faithful share for each student; each hypothesis by family and outside mirrored pairs. All of it goes to `results/factor/exam2/comparisons.txt`.

The four hypotheses tested are not corrected for multiple testing, so each CI should be read on its own.

## Known limits, stated in advance

- **The same author, after seeing the results.** I wrote these ideas after reading prereg 03's exam, and I wrote the verifier. Nobody independent checked them. The owner of the research system was sent the ideas and a draft of this text before the freeze, consented to opening 2026 once more for this exam, and asked for no change. Whether they read every idea isn't recorded.
- **2026 is not untouched.** Prereg 03's exam read it once, and its 2026 scores were read again afterwards. These ideas have never been scored on it, but anything learned about that year in general carries over. Nothing has read 2026 for any of these formulas. The owner consented on 2026-11-08 to one more opening, for this exam's answers only, and `sealed_access.log` will show each opening.
- **New families are thin.** The bank already covers most ideas daily bars allow. These share data fields, and sometimes a good deal of form, with training families: `own_moves_*` sits close to the level of volatility, and `trades_own_high_up` is a training idea's form on another field. So "never read" holds more firmly than "unlike anything taught".
- **Mirrored pairs.** 14 of the 21 ideas form 7 pairs that want opposite signs, so a stronger formula raises one side's IC and lowers the other's. That leaves room mostly for finding a better form, as in prereg 03, and leaves E1 and E3 a high bar.
- **21 ideas.** Intervals will be about as wide as on the 31 held-out ideas. A gap like S2's +0.0027 on those ideas could still miss.
- **Gates that can't judge an idea.** Eight drafts were dropped because no honest formula could pass: a shock moves a correlation either way, for instance. So these ideas lean toward what the gates can test.

## Pinned files

| File | sha256 |
|---|---|
| `task/ideas_fresh.jsonl` | `b54e053e3f16c815cc7ea6a6d8624a26538aa98633beb9eaeac329e99c3e9f84` |
| `task/ideas_fresh.zh.md` | `014565d420cbb12f1eba9dba1b9877fdcd6793550c7cea8c9e9d7e315e4937f2` |
| `scripts/factor/second_exam.py` | `418671c7567c4d78d8aedefcb274a9e80a067b12d8af5d4d86a8874273ac5d2f` |
| `scripts/factor/eval_stage.sh` | `e1b20db5cf340750b17efd0ca8769264b65a8670058d23cde13eee186e9b39d7` |
| `scripts/factor/explore_after_exam.py` | `283d87b27c5137d0d96755eb2d3f5633dee9ee8bf32a575a4c52e9bd03fdb352` |
| `scripts/factor/heldout_online.py` | `d67e33c64974b4fd8ddc552163a27d84ac8da16c168124df761dc574bab91291` |
| `scripts/factor/sealed_exam.py` | `45d1a761a3b96d45dd1637771d42e834c25f7763fe99e9b687301da2d22ef87b` |
| `scripts/factor/summarize_agent_eval.py` | `f90848e008bed5f852e844b8a5029897fc1c1f78cf2c01b740e6dfb2e2200762` |
| `envs/factor_idea/tests/test_second_exam.py` | `dfedd17388319f796ec90cab8df775d26e778a2a0b1b67b51b5a96ebbbee5a88` |
| adapter `factor-s1` | `9bebbd843147f55b44d3bdce6d0a2d93737adbc98b5c7603c7d8b8dbbf8b0199` |
| adapter `factor-s2` | `e2af1b3f12f712967cbe631b8fe768edf3a606875ba9c614bb63de886e36e003` |
| adapter `factor-s3p-ic` | `8557a72135790ddea9b5deac5bca02d11c1a0344302351ff43b93bc887e1c98d` |
| adapter `factor-s4-gates-seed1` | `c53bba1775671610c7c738eb4848653187bf5254357b36638d5e7744411cd8c8` |
| adapter `factor-s4-gates-seed2` | `3ae83a15ae3ac22b8719b5acc37f831b74bcb133e5e168a14c9ec20b6ebe6f84` |
| adapter `factor-s4-ic-seed1` | `91bfe9cf10c0c74ab9d5dfa6db8bb6943e3ac03b41f328f5b8861f874d101889` |
| adapter `factor-s4-ic-seed2` | `f4eabafbfba62bbb3b26f86173f54bb8479f2b3e78ca38efc058e0e5ec5b1d22` |
| adapter `factor-s4-judged-seed1` | `172384477d2e925abc3207d678488096e30cde24fc3320b6e18e7b05df506e67` |
| adapter `factor-s4-judged-seed2` | `a71dfa361c8fa9143d854e6bca30f2389d484a2c846a8aeb75104e65371234e5` |

The data, verifier, prompts, judge, scoring and comparison scripts and eval settings are as pinned in prereg 03: on 2026-11-08 all 28 of those files still matched. `second_exam.py answers` refuses to start unless this prereg is frozen with the ideas file's hash and each adapter matches the last line for its name in `results/factor/adapters.md`.

## Deviations

None in how the exam ran. All ten answer files were whole on the first attempt, each gave back its verdicts and ICs on 2025 before 2026 was opened, and each was scored on 2026 once: the sealed panel's access log went from 8 lines to 18.

One check was added after the results:

- **A recount, not pinned.** `scripts/factor/recount_exam2.py` counts E1 to E4 again from the stored answer files and 2026 files, sharing no code with `second_exam.py` or what it imports, and it decides nothing. Its bootstrap and keep-one draws are its own, so its intervals and E3 can differ from `results/factor/exam2/comparisons.txt` in the last digit. Its output is `results/factor/exam2/recount.txt`.
