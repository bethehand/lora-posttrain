#!/usr/bin/env bash
# Evaluate one adapter (or the base model, with "base") on a set of ideas, on two GPUs. Scores on 2025.
#   bash scripts/factor/eval_stage.sh <name> <adapter dir|base> <split> [gpus] [port] [rank]
# ROLLOUTS sets the rollouts per idea, 4 unless a pilot asks for more. IDEAS sets the ideas file, ideas_v2.jsonl unless an exam brings its own.
set -u
source "$(dirname "$0")/../../setup/env.sh"
name=$1 adapter=$2 split=$3 gpus=${4:-2,3} port=${5:-18200} rank=${6:-128}
ideas=${IDEAS:-$REPO/task/ideas_v2.jsonl}
model=$ROOT/models/Olmo-3-7B-Instruct
out=$ROOT/outputs/evals/factor
export FACTOR_IDEA_WORKERS=16 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py=$PRIME_RL_DIR/.venv/bin/python

extra="--vllm.data-parallel-size 2 --vllm.tool-parser-plugin $REPO/setup/vllm_olmo3_keep_text.py"
[ "$adapter" = base ] || extra="$extra --vllm.enable-lora --vllm.max-lora-rank $rank"
SERVE_EXTRA="$extra" setsid bash "$REPO/scripts/serve.sh" "$gpus" "$port" "$model" olmo3_keep_text 8192 > "$ROOT/logs/$name-serve.log" 2>&1 &
server=$!
for _ in $(seq 1 120); do
  curl -sf --max-time 3 "http://localhost:$((port + 100))/v1/models" > /dev/null && break
  sleep 5
done
curl -sf --max-time 3 "http://localhost:$((port + 100))/v1/models" > /dev/null || { echo "the server never came up, see $ROOT/logs/$name-serve.log"; kill -- -"$server" 2>/dev/null; exit 1; }
sleep 15  # the engine answers a little before the router in front of it does
served=$model
if [ "$adapter" != base ]; then
  bash "$REPO/scripts/load_adapter.sh" "$port" "$name" "$adapter" > /dev/null || { echo "adapter failed to load"; kill -- -"$server"; exit 1; }
  served=$name
fi
rm -rf "${out:?}/$name"
(cd "$PRIME_RL_DIR" && uv run eval @ "$REPO/configs/factor-idea/eval.toml" -m "$served" -r "${ROLLOUTS:-4}" -c 32 --output-dir "$out/$name" --no-rich \
   --client.base-url "http://localhost:$port/v1" --env.taskset.ideas "$ideas" --env.taskset.split "$split") > "$ROOT/logs/$name.log" 2>&1
echo "eval exit $?"
rm -f "$REPO/results/factor/agent/$name.jsonl" "$REPO/results/factor/agent/$name.summary.json"
(cd "$REPO" && $py scripts/factor/summarize_agent_eval.py "$out/$name" "$name" > /dev/null)
summarized=$?
kill -- -"$server" 2>/dev/null
sleep 15
pgrep -g "$server" > /dev/null && kill -9 -- -"$server" 2>/dev/null
[ "$summarized" = 0 ] || { echo "no summary, see $ROOT/logs/$name.log"; exit 1; }
$py - "$REPO/results/factor/agent/$name.jsonl" <<'PY'
import collections, json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
passed = [r for r in rows if r.get("matches_idea")]
reasons = collections.Counter("passed" if r.get("matches_idea") else ("no answer" if not r.get("factor") else (r.get("reason") or "?")[:40]) for r in rows)
ics = [r["ic"] for r in passed if r.get("ic") is not None]
tool = [m for r in rows for m in r["steps"] if m["role"] == "tool"]
errors = sum(not m["content"].startswith("computes") for m in tool)
print(f"{len(rows)} episodes, passed {len(passed)} ({len(passed) / len(rows):.1%}), mean reward_ic {sum(r.get('reward_ic') or 0 for r in rows) / len(rows):.3f}, IC among passes {sum(ics) / max(len(ics), 1):+.4f}, tests {sum(r.get('tests') or 0 for r in rows) / len(rows):.2f}, test errors {errors}/{len(tool)}")
print(dict(reasons.most_common(8)))
PY
