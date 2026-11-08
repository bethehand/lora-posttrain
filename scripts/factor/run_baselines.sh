#!/usr/bin/env bash
# Untrained baselines on the idea bank, train and held-out families together: summarize_agent_eval.py
# and the ideas file tell them apart afterwards. Scores on 2025.
#   bash scripts/factor/run_baselines.sh [prefix]
set -u
source "$(dirname "$0")/../../setup/env.sh"
prefix=${1:-v2}
ideas=$REPO/task/ideas_v2.jsonl
out=$ROOT/outputs/evals/factor
export FACTOR_IDEA_WORKERS=16 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=$PRIME_RL_DIR/.venv/bin/python

run_eval() { # name model tools extra...
  local name=$prefix-$1 model=$2 tools=$3; shift 3
  echo "=== $(date +%H:%M:%S) $name"
  rm -rf "${out:?}/$name"
  (cd "$PRIME_RL_DIR" && uv run eval @ "$REPO/configs/factor-idea/eval.toml" -m "$model" -r 4 --output-dir "$out/$name" --no-rich \
     --env.taskset.ideas "$ideas" --env.taskset.use-tools "$tools" "$@") > "$ROOT/logs/$name.log" 2>&1
  echo "    exit $? at $(date +%H:%M:%S)"
  (cd "$REPO" && $py scripts/factor/summarize_agent_eval.py "$out/$name" "$name" > /dev/null 2>&1)
}

serve() { # gpus port model parser plugin
  SERVE_EXTRA="--vllm.data-parallel-size 2${5:+ --vllm.tool-parser-plugin $5}" \
    setsid bash "$REPO/scripts/serve.sh" "$1" "$2" "$3" "$4" 8192 > "$ROOT/logs/$prefix-serve-$2.log" 2>&1 &
  server=$!
  for _ in $(seq 1 120); do
    curl -sf --max-time 3 "http://localhost:$(($2 + 100))/v1/models" > /dev/null && break
    sleep 5
  done
  sleep 20
  curl -sf --max-time 3 "http://localhost:$(($2 + 100))/v1/models" > /dev/null && echo "    served on $2" || echo "    server on $2 did not come up"
}

stop_server() {
  kill -- -"$server" 2>/dev/null
  sleep 15
  pgrep -g "$server" > /dev/null && kill -9 -- -"$server" 2>/dev/null
}

deepseek=(--client.base-url https://api.deepseek.com/v1 --client.api-key-var DEEPSEEK_API_KEY --sampling.max-tokens 4000 -c 16)
run_eval deepseek-tools deepseek-flash true "${deepseek[@]}"
run_eval deepseek-oneshot deepseek-flash false "${deepseek[@]}"

serve 0,1 18000 "$ROOT/models/Olmo-3-7B-Instruct" olmo3_keep_text "$REPO/setup/vllm_olmo3_keep_text.py"
run_eval olmo-tools "$ROOT/models/Olmo-3-7B-Instruct" true -c 32
run_eval olmo-oneshot "$ROOT/models/Olmo-3-7B-Instruct" false -c 32
stop_server

serve 0,1 18000 "$ROOT/models/Qwen3-4B-Instruct-2507" hermes
run_eval qwen-tools "$ROOT/models/Qwen3-4B-Instruct-2507" true -c 32
run_eval qwen-oneshot "$ROOT/models/Qwen3-4B-Instruct-2507" false -c 32
stop_server

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '
echo
echo "=== all done at $(date +%H:%M:%S)"
