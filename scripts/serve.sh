#!/usr/bin/env bash
# Serve a model for evals on one GPU, or on several with data parallelism through SERVE_EXTRA.
#   bash scripts/serve.sh <gpu> <port> <model dir> [tool parser] [max model len]
# With LoRA on 4 GPUs, then load the adapter with scripts/load_adapter.sh:
#   SERVE_EXTRA="--vllm.enable-lora --vllm.max-lora-rank 8 --vllm.data-parallel-size 4" \
#     bash scripts/serve.sh 0,1,2,3 18000 $ROOT/models/Qwen3-4B-Instruct-2507
set -euo pipefail
source "$(dirname "$0")/../setup/env.sh"
gpu=$1
port=$2
model=$3
parser=${4:-hermes}
max_len=${5:-8192}
read -ra extra <<< "${SERVE_EXTRA:-}"
cd "$PRIME_RL_DIR"
CUDA_VISIBLE_DEVICES=$gpu exec uv run inference --vllm.model "$model" --vllm.tool-call-parser "$parser" \
  --vllm.max-model-len "$max_len" --server.port "$port" "${extra[@]}"
