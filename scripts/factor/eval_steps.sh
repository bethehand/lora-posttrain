#!/usr/bin/env bash
# Evaluate a run's adapters from one server on a set of ideas, waiting for each one to appear.
# One server saves reloading the model for every adapter. A rank up to 128 fits. Scores on 2025.
#   bash scripts/factor/eval_steps.sh <prefix> <broadcast dir> <split> <gpus> <port> <step> [<step> ...]
set -u
source "$(dirname "$0")/../../setup/env.sh"
prefix=$1 broadcasts=$2 split=$3 gpus=$4 port=$5
shift 5
model=$ROOT/models/Olmo-3-7B-Instruct
out=$ROOT/outputs/evals/factor
export FACTOR_IDEA_WORKERS=16 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=$PRIME_RL_DIR/.venv/bin/python

SERVE_EXTRA="--vllm.data-parallel-size 2 --vllm.tool-parser-plugin $REPO/setup/vllm_olmo3_keep_text.py --vllm.enable-lora --vllm.max-lora-rank 128" \
  setsid bash "$REPO/scripts/serve.sh" "$gpus" "$port" "$model" olmo3_keep_text 8192 > "$ROOT/logs/$prefix-serve.log" 2>&1 &
server=$!
trap 'kill -- -"$server" 2>/dev/null' EXIT
for _ in $(seq 1 120); do
  curl -sf --max-time 3 "http://localhost:$((port + 100))/v1/models" > /dev/null && break
  sleep 5
done
curl -sf --max-time 3 "http://localhost:$((port + 100))/v1/models" > /dev/null || { echo "the server never came up, see $ROOT/logs/$prefix-serve.log"; exit 1; }
sleep 15  # the engine answers a little before the router in front of it does

for step in "$@"; do
  adapter=$broadcasts/step_$step
  waited=0
  until [ -e "$adapter/.finished" ] || [ "$waited" -ge "${ADAPTER_WAIT:-7200}" ]; do sleep 20; waited=$((waited + 20)); done
  [ -e "$adapter/.finished" ] || { echo "step $step: no adapter after ${waited}s"; continue; }
  name=$prefix-step$step
  bash "$REPO/scripts/load_adapter.sh" "$port" "$name" "$adapter" > /dev/null || { echo "$name: adapter failed to load"; continue; }
  rm -rf "${out:?}/$name"
  (cd "$PRIME_RL_DIR" && uv run eval @ "$REPO/configs/factor-idea/eval.toml" -m "$name" -r 4 -c 32 --output-dir "$out/$name" --no-rich \
     --client.base-url "http://localhost:$port/v1" --env.taskset.ideas "$REPO/task/ideas_v2.jsonl" --env.taskset.split "$split") > "$ROOT/logs/$name.log" 2>&1
  echo "$name: eval exit $? at $(date +%H:%M:%S)"
  rm -f "$REPO/results/factor/agent/$name.jsonl" "$REPO/results/factor/agent/$name.summary.json"
  (cd "$REPO" && $py scripts/factor/summarize_agent_eval.py "$out/$name" "$name" > /dev/null) || { echo "$name: no summary, see $ROOT/logs/$name.log"; continue; }
  $py - "$REPO/results/factor/agent/$name.summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
tools = dict(map(tuple, d["tool_results"]))
tests = sum(tools.values())
print(f"  passed {d['passed']:.1%}  answered {d['has_answer']:.1%}  parsed {d['parsed']:.1%}  reward_ic {d['reward_mean']:.3f}  tests {d['tests_per_episode']:.2f}  test errors {1 - tools.get('computed', 0) / max(tests, 1):.1%}  IC among passed {d['ic_mean_among_passed'] or 0:+.4f}")
print(f"  failures {d['failure_reasons'][:5]}  stops {d['stops']}")
PY
done
echo "all steps evaluated at $(date +%H:%M:%S)"
