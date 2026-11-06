#!/usr/bin/env bash
# One SFT run on the GPUs given, at least two. Adapters land in $ROOT/outputs/sft/<name>/<name>/broadcasts/step_N.
#   bash scripts/factor/train_sft.sh <name> <dataset dir> <steps> <rank> <gpus> [adapter every] [start adapter] [more sft options]
# The start adapter comes from the stage before; without one the adapter starts fresh.
set -u
source "$(dirname "$0")/../../setup/env.sh"
name=$1 data=$2 steps=$3 rank=$4 gpus=$5 every=${6:-71} start=${7:-}
shift "$(( $# < 7 ? $# : 7 ))"

cd "$PRIME_RL_DIR" || exit 1
export CUDA_VISIBLE_DEVICES=$gpus LORA_POSTTRAIN_ADAPTER_EVERY=$every
[ -z "$start" ] || export LORA_POSTTRAIN_INIT_ADAPTER=$start
uv run sft @ "$REPO/configs/factor-idea/sft.toml" --data.name "$data" --model.lora.rank "$rank" \
  --deployment.num-train-gpus "$(echo "$gpus" | tr ',' '\n' | wc -l)" \
  --max-steps "$steps" --output-dir "$ROOT/outputs/sft/$name" --run.name "$name" "$@"
status=$?
echo "train exit $status at $(date +%H:%M:%S)"
exit "$status"
