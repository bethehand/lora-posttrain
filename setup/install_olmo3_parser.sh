#!/usr/bin/env bash
# Register the OLMo 3 tool parser (setup/renderers_olmo3.py) with prime-rl's renderers package as
# tool_parser "olmo3", for `[orchestrator.renderer] name = "default"` and `tool_parser = "olmo3"`.
# renderers 0.1.10 (deps/renderers @ cb82439) ships no OLMo renderer or parser. Idempotent; rerun
# after editing the parser.
#   bash setup/install_olmo3_parser.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
pkg="$PRIME_RL_DIR/deps/renderers/renderers"
cp "$REPO/setup/renderers_olmo3.py" "$pkg/olmo3_tool_parser.py"
marker="# lora-posttrain: OLMo 3 tool parser"
if ! grep -qF "$marker" "$pkg/parsers.py"; then
  cat >> "$pkg/parsers.py" <<EOF

$marker
from renderers.olmo3_tool_parser import Olmo3ToolParser  # noqa: E402

TOOL_PARSERS["olmo3"] = Olmo3ToolParser
EOF
fi
cd "$PRIME_RL_DIR"
uv run python -c "from renderers.parsers import TOOL_PARSERS; assert 'olmo3' in TOOL_PARSERS; print('olmo3 tool parser registered')"
