#!/usr/bin/env bash
# The difficulty-filter pass (configs/wiki-search/filter.toml) sharded over several inference
# servers, one per GPU, then merged and summarized as one run.
#   bash scripts/filter_parallel.sh <server base url> [<server base url> ...]
set -euo pipefail
source "$(dirname "$0")/../setup/env.sh"
cd "$PRIME_RL_DIR"
out="$ROOT/outputs/evals/qwen3-4b-base-filter/train"
rm -rf "$out"

shard=0
for url in "$@"; do
  uv run eval @ "$REPO/configs/wiki-search/filter.toml" --client.base-url "$url" \
    --env.taskset.num-shards "$#" --env.taskset.shard "$shard" \
    --output-dir "$out/shard$shard" --no-rich > "$ROOT/logs/filter-shard$shard.log" 2>&1 &
  shard=$((shard + 1))
done
wait

mkdir -p "$out/merged"
cat "$out"/shard*/*/traces.jsonl > "$out/merged/traces.jsonl"
uv run python "$REPO/scripts/summarize_eval.py" "$out/merged" qwen3-4b-base-filter/train
