# lora-posttrain

Post-training small open models with LoRA (SFT and RL) on one 4× RTX 4090 machine, evaluated on clean held-out splits with paired statistics.

## Status

- [x] **Chapter 1: wiki-search with Qwen3-4B.** Prime Intellect's result (Qwen3-4B-Instruct-2507, LoRA RL, tool-use search over Wikipedia) rerun on a held-out split by source page, pre-registered in [prereg/01-wiki-search.md](prereg/01-wiki-search.md). On pages never trained on, accuracy rises from 0.567 to 0.633 (+0.066, 95% CI +0.042 to +0.090), from reading the right section; trained questions gain +0.371, held-out questions of the same kind +0.207.
- [x] **Chapter 2: wiki-search with OLMo 3 7B.** The same recipe on AI2's OLMo-3-7B-Instruct, pre-registered in [prereg/02-olmo-wiki-search.md](prereg/02-olmo-wiki-search.md). Held-out accuracy rises from 0.080 to 0.180 (+0.101, +0.079 to +0.122) and all six hypotheses pass, but most of the gain is answers from memory after failed tool calls, not retrieval.

See [REPORT.md](REPORT.md).

## Hardware

4× RTX 4090 24 GB (PCIe, no peer-to-peer), 1 TiB RAM, 128 CPU threads, NVIDIA driver 550.144.03 (CUDA 12.4). Exact software versions are recorded in `results/versions.md`.

## Layout

| Path | Contents |
|---|---|
| `setup/` | Environment variables, install script, version recording |
| `envs/` | Environment variants (held-out wiki-search) |
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
