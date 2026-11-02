#!/usr/bin/env bash
# The pre-registered eval suite for one served model on wiki-search-heldout, 4 rollouts per
# question, judge deepseek-v4-pro: heldout, train_sample, and heldout without tools.
# Needs an inference server on $INFERENCE_PORT (see configs/wiki-search/eval.toml).
#   bash scripts/eval_suite.sh <model path or adapter name> <results name>
set -euo pipefail
source "$(dirname "$0")/../setup/env.sh"
model=$1
name=$2
cd "$PRIME_RL_DIR"

run() {  # <tag> <extra eval args...>
  local tag=$1
  shift
  local dir="$ROOT/outputs/evals/$name/$tag"
  rm -rf "$dir"
  uv run eval @ "$REPO/configs/wiki-search/eval.toml" -m "$model" --output-dir "$dir" --no-rich "$@"
  uv run python "$REPO/scripts/summarize_eval.py" "$(ls -d "$dir"/*/ | head -1)" "$name/$tag"
}

run heldout --env.taskset.split heldout
run train_sample --env.taskset.split train_sample
run heldout-notools --env.taskset.split heldout --env.taskset.use-tools false
