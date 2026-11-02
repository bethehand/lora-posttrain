#!/usr/bin/env bash
# RL pilot: 10 steps of configs/wiki-search/rl.toml on 2 GPUs (1 inference, 1 trainer, half the
# batch), to check the stack end to end and measure step time and memory before the
# pre-registered run.
#   bash scripts/rl_pilot.sh [gpus] [port] [extra rl args...]
set -euo pipefail
source "$(dirname "$0")/../setup/env.sh"
gpus=${1:-0,1}
port=${2:-18400}
shift $(( $# < 2 ? $# : 2 ))
cd "$PRIME_RL_DIR"

CUDA_VISIBLE_DEVICES=$gpus uv run rl @ "$REPO/configs/wiki-search/rl.toml" \
  --max-steps 10 \
  --deployment.num-infer-gpus 1 --deployment.num-train-gpus 1 \
  --orchestrator.batch-size 128 \
  --inference.server.port "$port" --orchestrator.model.client.base-url "http://localhost:$port/v1" \
  --no-orchestrator.eval \
  --output-dir "$PRL_OUTPUT_DIR/pilot" "$@"
