#!/usr/bin/env bash
# Serve a model for evals on one GPU.
#   bash scripts/serve.sh <gpu> <port> <model dir> [tool parser] [max model len]
set -euo pipefail
source "$(dirname "$0")/../setup/env.sh"
gpu=$1
port=$2
model=$3
parser=${4:-hermes}
max_len=${5:-8192}
cd "$PRIME_RL_DIR"
CUDA_VISIBLE_DEVICES=$gpu exec uv run inference --vllm.model "$model" --vllm.tool-call-parser "$parser" \
  --vllm.max-model-len "$max_len" --server.port "$port"
