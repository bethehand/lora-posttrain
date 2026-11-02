#!/usr/bin/env bash
# Gate for stage 1: does the pinned prime-rl stack run on this box (driver 550, CUDA 12.4)?
set -euo pipefail
source "$(dirname "$0")/../setup/env.sh"
cd "$PRIME_RL_DIR"

uv run python -V
uv run python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
uv run python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok", float((x @ x).float().abs().mean()))
EOF

# Full RL stack on two GPUs: vLLM inference, orchestrator and trainer, 20 steps on Qwen3-0.6B
# The orchestrator's client URL does not follow inference.server.port, so both are set
CUDA_VISIBLE_DEVICES=0,1 uv run rl @ configs/basic/reverse-text/rl.toml \
  --no-monitors.wandb --output-dir "$PRL_OUTPUT_DIR/smoke" \
  --inference.server.port "$INFERENCE_PORT" \
  --orchestrator.model.client.base-url "http://localhost:$INFERENCE_PORT/v1"
