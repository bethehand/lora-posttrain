# lora-posttrain

Post-training small open models with LoRA (SFT and RL) on one 4× RTX 4090 machine, evaluated on clean held-out splits with paired statistics.

## Status

- [x] **Chapter 1: wiki-search with Qwen3-4B.** Prime Intellect's result (Qwen3-4B-Instruct-2507, LoRA RL, tool-use search over Wikipedia) rerun on a held-out split by source page, pre-registered in [prereg/01-wiki-search.md](prereg/01-wiki-search.md). On pages never trained on, accuracy rises from 0.567 to 0.633 (+0.066, 95% CI +0.042 to +0.090), from reading the right section; trained questions gain +0.371, held-out questions of the same kind +0.207.
- [x] **Chapter 2: wiki-search with OLMo 3 7B.** The same recipe on AI2's OLMo-3-7B-Instruct, pre-registered in [prereg/02-olmo-wiki-search.md](prereg/02-olmo-wiki-search.md). Held-out accuracy rises from 0.080 to 0.180 (+0.101, +0.079 to +0.122) and all six hypotheses pass, but most of the gain is answers from memory after failed tool calls, not retrieval.
- [x] **Chapter 3: from a one-sentence idea to a factor, on OLMo 3 7B.** The owner's quant research as a narrow task: from one investment idea, write a factor formula, test it up to five times on past data, and hand one in. Taught to OLMo-3-7B-Instruct in four stages (two of SFT, then offline RL, then online RL), pre-registered in [prereg/03-factor-idea.md](prereg/03-factor-idea.md) and scored on 2025, with a sealed 2026 period for the test out of time, then tested again on 21 new ideas, scored on 2026, in [prereg/04-second-exam.md](prereg/04-second-exam.md). SFT teaches the task (pass rate on unseen idea families 0% to 51.6%) and RL makes it reliable (on new ideas the online-RL model passes 19.0 points more than the SFT model S2), but no stage was shown to teach better formulas. Keeping one of four answers by research-window incremental IC adds +0.0041 (+0.0027 to +0.0055) of incremental IC on 2026, and at about the same number of tests the student with this process can't be told from the teacher model with it. The working log is [results/factor/week2.md](results/factor/week2.md).

See [REPORT.md](REPORT.md). The whole report, all three chapters, is also in Chinese, in [REPORT.zh.md](REPORT.zh.md).

## Hardware

4× RTX 4090 24 GB (PCIe, no peer-to-peer), 1 TiB RAM, 128 CPU threads, NVIDIA driver 550.144.03 (CUDA 12.4). Exact software versions are recorded in `results/versions.md`.

## Layout

| Path | Contents |
|---|---|
| `setup/` | Environment variables, install script, version recording |
| `envs/` | Environment variants (held-out wiki-search; the factor sandbox and its tests) |
| `task/` | Chapter 3's research manual, ideas and seed factors |
| `configs/` | Training and evaluation configs for 4× 4090 |
| `scripts/` | Smoke test, data preparation, training, evaluation, analysis |
| `prereg/` | Hypotheses and pass criteria, committed before each run |
| `results/` | Metrics, per-question results, figures |

## Reproduce

```bash
scripts/sync.sh push                      # from your machine; needs .env.local
# on the server
cd /mnt/nvme1/bruce/lora-posttrain
bash setup/install.sh                     # uv + prime-rl v0.9.0, pinned
bash scripts/00_smoke.sh                  # gate: full RL stack on 2 GPUs
bash setup/install_olmo3_parser.sh        # chapter 2: OLMo 3 tool parser for prime-rl's renderers
```

Chapter 3's code is in `scripts/factor/` and `envs/factor_idea/`. Its market data, a frozen snapshot of Binance USDT perpetual daily bars, is not in the repo; `scripts/factor/build_panel.py` turns such a snapshot into the panels the sandbox reads.

## License

[Apache-2.0](LICENSE).
