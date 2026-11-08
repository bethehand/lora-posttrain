#!/usr/bin/env bash
# One online RL run on factor-idea from an adapter, on GPUs 0-3, into a directory that must not exist yet.
#   bash scripts/factor/train_rl.sh <name> <start adapter> <ic|gates|judged> [more rl options, such as --trainer.model.lora.rank 8]
set -euo pipefail
source "$(dirname "$0")/../../setup/env.sh"
name=$1 start=$2 arm=$3
shift 3
out=$PRL_OUTPUT_DIR/$name
if [ -e "$out" ]; then
  echo "$out exists; move it aside first so the run starts clean" >&2
  exit 1
fi
cd "$PRIME_RL_DIR"
export FACTOR_IDEA_WORKERS=32 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 LORA_POSTTRAIN_INIT_ADAPTER=$start
config=rl.toml
[ "$arm" = ic ] || config=rl-$arm.toml
CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3} uv run rl @ "$REPO/configs/factor-idea/$config" --output-dir "$out" "$@"
