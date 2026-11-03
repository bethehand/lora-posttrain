#!/usr/bin/env bash
# Everything after the prereg 02 training run, unattended on the server:
# 1. wait for the run to end; if it crashed before step 35, set it aside and restart it once from
#    scratch with the same config (prereg 02's crash rule), then wait again
# 2. keep the step-35 adapter, its sha256, the run's configs and the per-step table
# 3. serve base and adapter from one LoRA vLLM server on GPUs 0-3 (olmo3 tool parser)
# 4. run the pre-registered eval suite for both in the same session, in parallel
# 5. stop the server, then run the analysis scripts (no GPU needed)
# Progress goes to $ROOT/logs/prereg02-post.log; the server is stopped on any exit.
#   tmux new-session -d -s lp-olmo-post "bash $REPO/scripts/prereg02_after_training.sh"
set -uo pipefail
source "$(dirname "$0")/../setup/env.sh"
exec >> "$ROOT/logs/prereg02-post.log" 2>&1

MODEL="$ROOT/models/Olmo-3-7B-Instruct"
ADAPTER="$ROOT/adapters/wiki-search-olmo-rl35"
NAME=olmo-rl35
PORT=$INFERENCE_PORT
SERVE=lp-olmo-eval-serve
TRAIN=lp-olmo-train
OUT="$REPO/results/train/prereg02"
log() { echo "$(date '+%F %T') $*"; }

stop_server() {
  tmux kill-session -t "=$SERVE" 2>/dev/null && log "stopped $SERVE"
}
trap stop_server EXIT

wait_gpus_free() {  # until every GPU holds under 1 GB, at most 10 minutes
  for _ in $(seq 1 60); do
    if nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 1000 {busy=1} END {exit busy}'; then
      return 0
    fi
    sleep 10
  done
  return 1
}

run_dir() { ls -d "$PRL_OUTPUT_DIR"/wiki-search-olmo/*/ 2>/dev/null | head -1 | sed 's:/$::'; }

# 1. wait for training, with one restart from scratch if it crashed
for attempt in 1 2; do
  log "waiting for $TRAIN (attempt $attempt)"
  while tmux has-session -t "=$TRAIN" 2>/dev/null; do sleep 60; done
  RUN=$(run_dir)
  if [ -n "$RUN" ] && [ -f "$RUN/broadcasts/step_35/adapter_model.safetensors" ]; then
    log "training finished: $RUN"
    break
  fi
  log "training ended without a step-35 adapter (run dir: ${RUN:-none})"
  tail -n 20 "$ROOT/logs/olmo-train.log"
  if [ "$attempt" = 2 ]; then
    log "second attempt failed too; stopping"
    exit 1
  fi
  wait_gpus_free || { log "GPUs did not free up; stopping"; exit 1; }
  mv "$PRL_OUTPUT_DIR/wiki-search-olmo" "$PRL_OUTPUT_DIR/wiki-search-olmo-crash1"
  mv "$ROOT/logs/olmo-train.log" "$ROOT/logs/olmo-train-crash1.log"
  log "restarting training from scratch with the same config (prereg 02 crash rule)"
  tmux new-session -d -s "$TRAIN" "cd $REPO && CONFIG=rl-olmo.toml RUN_NAME=wiki-search-olmo bash scripts/rl_train.sh > $ROOT/logs/olmo-train.log 2>&1"
  sleep 120
done

# 2. keep the adapter and the run record
mkdir -p "$ADAPTER" "$OUT"
cp "$RUN"/broadcasts/step_35/adapter_config.json "$RUN"/broadcasts/step_35/adapter_model.safetensors "$ADAPTER"/
(cd "$ADAPTER" && sha256sum adapter_model.safetensors adapter_config.json) > "$OUT/adapter.sha256"
cp -r "$RUN/configs" "$OUT/"
python3 "$REPO/scripts/run_steps.py" "$RUN/metrics.jsonl" 0.3 6.0 > "$OUT/steps.txt"
log "adapter kept at $ADAPTER"; cat "$OUT/adapter.sha256"; tail -n 1 "$OUT/steps.txt"

# 3. one LoRA server for both models
wait_gpus_free || { log "GPUs did not free up after training; stopping"; exit 1; }
tmux new-session -d -s "$SERVE" "SERVE_EXTRA='--vllm.enable-lora --vllm.max-lora-rank 8 --vllm.data-parallel-size 4' bash $REPO/scripts/serve.sh 0,1,2,3 $PORT $MODEL olmo3 8192 > $ROOT/logs/prereg02-serve.log 2>&1"
ready=0
for _ in $(seq 1 90); do
  if curl -sf -m 5 "http://localhost:$((PORT + 100))/v1/models" > /dev/null; then ready=1; break; fi
  sleep 10
done
[ "$ready" = 1 ] || { log "server not ready after 15 minutes"; tail -n 30 "$ROOT/logs/prereg02-serve.log"; exit 1; }
bash "$REPO/scripts/load_adapter.sh" "$PORT" "$NAME" "$ADAPTER" || { log "adapter failed to load"; exit 1; }
log "server ready with $NAME"

# 4. the pre-registered suite, base and adapter in the same session
export EVAL_EXTRA="--env.taskset.read-arg section"
export TAGS="heldout train_sample train_mixed_olmo heldout-notools heldout-sectionid"
bash "$REPO/scripts/eval_suite.sh" "$MODEL" prereg02-base > "$ROOT/logs/prereg02-eval-base.log" 2>&1 &
base_pid=$!
bash "$REPO/scripts/eval_suite.sh" "$NAME" prereg02-rl35 > "$ROOT/logs/prereg02-eval-rl35.log" 2>&1 &
rl_pid=$!
wait "$base_pid"; base_rc=$?
wait "$rl_pid"; rl_rc=$?
log "eval suites finished: base exit $base_rc, adapter exit $rl_rc"

# 5. free the GPUs, then analyse
stop_server
cd "$PRIME_RL_DIR"
for run in prereg02-base prereg02-rl35; do
  for tag in heldout heldout-sectionid; do
    dir=$(ls -d "$ROOT/outputs/evals/$run/$tag"/*/ 2>/dev/null | head -1)
    [ -n "$dir" ] && python3 "$REPO/scripts/call_failures.py" "$dir" "$run/$tag"
  done
  dir=$(ls -d "$ROOT/outputs/evals/$run/heldout"/*/ 2>/dev/null | head -1)
  if [ -n "$dir" ]; then
    uv run python "$REPO/scripts/navigation_metrics.py" "$dir" "$run/heldout"
    uv run python "$REPO/scripts/rejudge.py" "$dir" "$run/heldout-flash"
  fi
done
python3 "$REPO/scripts/navigation_metrics.py" --compare prereg02-base/heldout prereg02-rl35/heldout
python3 "$REPO/scripts/call_failures.py" --compare prereg02-base/heldout prereg02-rl35/heldout
python3 "$REPO/scripts/prereg02_tests.py"
python3 "$REPO/scripts/by_base_score.py" prereg02-base prereg02-rl35 olmo3-7b-base-renamed train_mixed_olmo
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
log "all done"
