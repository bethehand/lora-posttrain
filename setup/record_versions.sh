#!/usr/bin/env bash
# Write the exact software and hardware versions to results/versions.md.
set -euo pipefail
source "$(dirname "$0")/env.sh"
out="$REPO/results/versions.md"
mkdir -p "$(dirname "$out")"

{
  echo "# Versions"
  echo
  echo "Recorded $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)."
  echo
  echo '```'
  grep PRETTY_NAME /etc/os-release
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  uv --version
  echo "prime-rl $(git -C "$PRIME_RL_DIR" rev-parse HEAD) ($(git -C "$PRIME_RL_DIR" describe --tags --always))"
  git -C "$PRIME_RL_DIR" submodule status
  cd "$PRIME_RL_DIR"
  uv run python - <<'EOF'
import importlib.metadata as md
import platform
print("python", platform.python_version())
for name in ["torch", "vllm", "flash-attn", "flashinfer-python", "triton", "transformers", "nvidia-nccl-cu12", "verifiers"]:
    try:
        print(name, md.version(name))
    except md.PackageNotFoundError:
        print(name, "not installed")
EOF
  echo '```'
} > "$out"
cat "$out"
