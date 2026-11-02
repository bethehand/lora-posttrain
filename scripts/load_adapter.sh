#!/usr/bin/env bash
# Load a LoRA adapter into a running scripts/serve.sh server started with --vllm.enable-lora.
# The admin route lives on the engine, which listens on the server port + 100 behind the router.
#   bash scripts/load_adapter.sh <server port> <adapter name> <adapter dir>
set -euo pipefail
port=$1
name=$2
dir=$3
curl -sf -X POST "http://localhost:$((port + 100))/load_lora_adapter" -H "Content-Type: application/json" \
  -d "{\"lora_name\": \"$name\", \"lora_path\": \"$dir\"}"
echo
# The router answers /v1/models with a 500 under data parallelism, so ask the engine
curl -sf "http://localhost:$((port + 100))/v1/models" | python3 -c "import json, sys; print([m['id'] for m in json.load(sys.stdin)['data']])"
