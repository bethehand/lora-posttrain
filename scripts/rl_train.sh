#!/usr/bin/env bash
# The pre-registered wiki-search run (prereg/01-wiki-search.md): configs/wiki-search/rl.toml as is,
# on GPUs 0-3, into an output directory that must not exist yet.
#   bash scripts/rl_train.sh
#   RUN_NAME=wiki-search-lr1e-5 bash scripts/rl_train.sh --trainer.optim.lr 1e-5   # the guard's rerun
set -euo pipefail
source "$(dirname "$0")/../setup/env.sh"
out="$PRL_OUTPUT_DIR/${RUN_NAME:-wiki-search}"
if [ -e "$out" ]; then
  echo "$out exists; move it aside first so the run starts clean" >&2
  exit 1
fi
cd "$PRIME_RL_DIR"

CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} uv run rl @ "$REPO/configs/wiki-search/rl.toml" --output-dir "$out" "$@"
