# Smoke test: prime-rl v0.9.0 on 4x RTX 4090, driver 550

2026-11-02, `scripts/00_smoke.sh`: `configs/basic/reverse-text/rl.toml` from prime-rl v0.9.0,
Qwen3-0.6B (reverse-text SFT checkpoint), full fine-tuning, inference on GPU 0, trainer on GPU 1.

**Result: passes.** 20 steps in about 4 minutes, 0% errored rollouts, NCCL weight broadcast
working with P2P disabled. The CUDA 12.8/12.9 wheels run on driver 550 (CUDA 12.4) through
minor-version compatibility, so no driver upgrade is needed.

| Step | Time (s) | Mean reward |
|---|---|---|
| 1 | 17.4 | 0.183 |
| 5 | 4.4 | 0.405 |
| 10 | 4.3 | 0.725 |
| 15 | 7.9 | 0.786 |
| 20 | 8.1 | 0.810 |

Sampler-trainer mismatch KL stayed between 0.0006 and 0.009.

## What broke on the way

1. **Stalled downloads from GitHub.** `uv sync --all-extras` hung for 15 minutes on a GitHub
   download, and later git fetches failed with TLS resets. Fix: install only the `gpu`,
   `flash-attn` and `dashboard` extras, and retry `uv sync` (the uv cache keeps finished downloads).
2. **`vllm-router` missing.** prime-rl's inference entrypoint starts it by default, but it ships
   in the skipped `disagg` extra. Fix: install that one wheel at the lock's URL and hash.
3. **Port 8000 taken.** Another user's service on the shared server holds port 8000, prime-rl's
   default inference port. The router could not bind, and the orchestrator sent its requests to
   that service and got 401s, so every rollout failed. Fix: inference on port 18000, with the
   orchestrator client URL set explicitly because it does not follow `inference.server.port`.
