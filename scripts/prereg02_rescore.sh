#!/usr/bin/env bash
# Prereg 02 deviation: the DeepSeek API timed out on most judge calls between 16:50 and 18:20 on
# 2026-11-03, during the evals. Retry those calls with the same judge (scripts/rescore_failed.py),
# keep every verdict the run did give, and rerun the analysis on the filled-in runs, named
# <run>-rescored. The runs as scored, with failed episodes dropped, stay under their own names.
#   bash scripts/prereg02_rescore.sh
set -euo pipefail
source "$(dirname "$0")/../setup/env.sh"
cd "$PRIME_RL_DIR"
for run in prereg02-base prereg02-rl35; do
  for tag in heldout train_sample train_mixed_olmo heldout-notools heldout-sectionid; do
    src=$(ls -d "$ROOT/outputs/evals/$run/$tag"/*/ | head -1)
    src=${src%/}
    dst="$ROOT/outputs/evals/$run-rescored/$tag/$(basename "$src")"
    check=0
    case $tag in heldout|train_sample) check=200 ;; esac
    uv run python "$REPO/scripts/rescore_failed.py" "$src" "$dst" --check "$check"
    uv run python "$REPO/scripts/summarize_eval.py" "$dst" "$run-rescored/$tag"
  done
  for tag in heldout heldout-sectionid; do
    dst=$(ls -d "$ROOT/outputs/evals/$run-rescored/$tag"/*/ | head -1)
    python3 "$REPO/scripts/call_failures.py" "$dst" "$run-rescored/$tag"
  done
  dst=$(ls -d "$ROOT/outputs/evals/$run-rescored/heldout"/*/ | head -1)
  uv run python "$REPO/scripts/navigation_metrics.py" "$dst" "$run-rescored/heldout"
  uv run python "$REPO/scripts/rejudge.py" "$dst" "$run-rescored/heldout-flash"
done
python3 "$REPO/scripts/navigation_metrics.py" --compare prereg02-base-rescored/heldout prereg02-rl35-rescored/heldout
python3 "$REPO/scripts/call_failures.py" --compare prereg02-base-rescored/heldout prereg02-rl35-rescored/heldout
python3 "$REPO/scripts/prereg02_tests.py" prereg02-base-rescored prereg02-rl35-rescored
python3 "$REPO/scripts/by_base_score.py" prereg02-base-rescored prereg02-rl35-rescored olmo3-7b-base-renamed train_mixed_olmo
python3 "$REPO/scripts/by_base_score.py" prereg02-base prereg02-rl35 olmo3-7b-base-renamed train_mixed_olmo
echo "rescore done $(date '+%F %T')"
