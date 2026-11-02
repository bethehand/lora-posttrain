#!/usr/bin/env bash
# RL pilot: the first few steps of configs/wiki-search/rl.toml exactly as configured, without the
# online evals, to check memory and time a step before the pre-registered run.
#   STEPS=5 bash scripts/rl_pilot.sh [extra rl args...]
set -euo pipefail
source "$(dirname "$0")/../setup/env.sh"
cd "$PRIME_RL_DIR"

CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} uv run rl @ "$REPO/configs/wiki-search/rl.toml" \
  --max-steps "${STEPS:-5}" --no-orchestrator.eval --output-dir "$PRL_OUTPUT_DIR/pilot" "$@"
