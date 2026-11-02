# lora-posttrain

Post-training small open models with LoRA (SFT and RL) on one 4× RTX 4090 machine, evaluated on clean held-out splits with paired statistics.

## Status

- [ ] **Chapter 1: wiki-search.** Reproduce Prime Intellect's result (Qwen3-4B-Instruct-2507, LoRA RL, tool-use search over Wikipedia) on a held-out split by source page. The original numbers were measured on training questions; see [REPORT.md](REPORT.md).

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
```
