#!/usr/bin/env bash
# The difficulty-filter pass (configs/wiki-search/filter.toml) sharded over several inference
# servers, one per GPU, then merged and summarized as one run.
#   bash scripts/filter_parallel.sh <server base url> [<server base url> ...]
# FILTER_NAME names the results (default qwen3-4b-base-filter) and FILTER_EXTRA adds eval
# arguments, e.g. for OLMo:
#   FILTER_NAME=olmo3-7b-base-filter FILTER_EXTRA="-m $ROOT/models/Olmo-3-7B-Instruct \
#     --env.taskset.read-arg section" bash scripts/filter_parallel.sh <urls...>
set -euo pipefail
source "$(dirname "$0")/../setup/env.sh"
cd "$PRIME_RL_DIR"
name=${FILTER_NAME:-qwen3-4b-base-filter}
read -ra extra <<< "${FILTER_EXTRA:-}"
out="$ROOT/outputs/evals/$name/train"
rm -rf "$out"

shard=0
for url in "$@"; do
  uv run eval @ "$REPO/configs/wiki-search/filter.toml" --client.base-url "$url" \
    --env.taskset.num-shards "$#" --env.taskset.shard "$shard" "${extra[@]}" \
    --output-dir "$out/shard$shard" --no-rich > "$ROOT/logs/$name-shard$shard.log" 2>&1 &
  shard=$((shard + 1))
done
wait

mkdir -p "$out/merged"
cat "$out"/shard*/*/traces.jsonl > "$out/merged/traces.jsonl"
uv run python "$REPO/scripts/summarize_eval.py" "$out/merged" "$name/train"
