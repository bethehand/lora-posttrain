# Source on the server before running anything: `source setup/env.sh`.
# Every download, cache and output goes to the data disk; the system disk is shared and nearly full.

export ROOT="${ROOT:-/mnt/nvme1/bruce}"
export REPO="$ROOT/lora-posttrain"
export PRIME_RL_DIR="$ROOT/prime-rl"

export PATH="$ROOT/bin:$PATH"
export UV_CACHE_DIR="$ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOT/python"
# `uv run` would otherwise re-sync and drop the extras and env packages installed by setup/install.sh
export UV_NO_SYNC=1

export XDG_CACHE_HOME="$ROOT/cache"
export HF_HOME="$ROOT/cache/huggingface"
export MODELSCOPE_CACHE="$ROOT/cache/modelscope"
export VLLM_CACHE_ROOT="$ROOT/cache/vllm"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export FLASHINFER_WORKSPACE_BASE="$ROOT/cache"
export CUDA_CACHE_PATH="$ROOT/cache/nv"
export WIKI_SEARCH_CACHE="$ROOT/cache/wiki_search"
export TMPDIR="$ROOT/tmp"
export PRL_OUTPUT_DIR="$ROOT/outputs"

# RTX 4090s have no peer-to-peer over PCIe
export NCCL_P2P_DISABLE=1
# W&B is unreachable from this server; prime-rl's file monitor writes metrics.jsonl instead
export WANDB_MODE=disabled

# vLLM and the orchestrator keep many sockets open; the default soft limit here is 1024
ulimit -n 65536

mkdir -p "$ROOT"/{bin,cache,python,tmp,outputs,models,logs}

# API keys (DEEPSEEK_API_KEY) live outside the repo, readable by this user only
if [ -f "$ROOT/.secrets.env" ]; then
  source "$ROOT/.secrets.env"
fi
