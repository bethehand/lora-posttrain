#!/usr/bin/env bash
# Install uv and prime-rl at a pinned commit under $ROOT. Safe to re-run.
set -euo pipefail
source "$(dirname "$0")/env.sh"

UV_VERSION=0.13.6
# prime-rl v0.9.0 (2026-08-25): the last release on CUDA 12 wheels. v0.10.0 moved to CUDA 13,
# which needs driver >= 580; this server runs 550.
PRIME_RL_COMMIT=ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1

if ! uv --version 2>/dev/null | grep -q "$UV_VERSION"; then
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" \
    | env UV_INSTALL_DIR="$ROOT/bin" UV_NO_MODIFY_PATH=1 sh
fi
uv --version

if [ ! -d "$PRIME_RL_DIR/.git" ]; then
  git clone --filter=blob:none https://github.com/PrimeIntellect-ai/prime-rl.git "$PRIME_RL_DIR"
fi
cd "$PRIME_RL_DIR"
git fetch --tags origin
git checkout --detach "$PRIME_RL_COMMIT"
# Submodules are declared with SSH URLs; fetch them over HTTPS so no GitHub key is needed here
git -c url."https://github.com/".insteadOf="git@github.com:" \
  submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config

# Locked dependencies plus the two upstream environments we use. Only the extras a 4090 box
# needs: --all-extras would add Hopper/Blackwell kernels and multi-node disaggregation, which
# are useless here and mostly hosted on GitHub, where large downloads stall from this server.
# Git fetches from GitHub also drop at random (TLS resets); downloads already in the uv cache
# survive, so each retry only redoes what failed.
export UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=8
for attempt in 1 2 3 4 5; do
  if uv sync --locked --extra gpu --extra flash-attn --extra dashboard \
      --package prime-rl --package reverse-text --package wiki-search; then
    break
  fi
  if [ "$attempt" = 5 ]; then
    echo "uv sync failed 5 times" >&2
    exit 1
  fi
  echo "uv sync failed (attempt $attempt), retrying in 30s" >&2
  sleep 30
done
# Installed after `uv sync`, which is exact and would remove them:
# - our environment variants, editable so a `scripts/sync.sh push` takes effect without reinstalling;
# - vllm-router, which `uv run inference` puts in front of vLLM to keep a multi-turn session on one
#   engine (prefix-cache hits). It ships in the `disagg` extra, whose other packages are for Hopper
#   or multi-node setups, so only this wheel is installed, at the lock's URL and hash.
uv pip install --no-deps -e "$REPO/envs/wiki_search_heldout"
uv pip install --no-deps --require-hashes -r "$REPO/setup/vllm-router.txt"

bash "$REPO/setup/record_versions.sh"
