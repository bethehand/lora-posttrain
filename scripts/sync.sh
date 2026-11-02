#!/usr/bin/env bash
# Mirror this repo to the server (push) or bring results back (pull).
# The server address lives in .env.local (git-ignored): REMOTE_HOST, REMOTE_PORT, REMOTE_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."
source .env.local
ssh_cmd="ssh -p $REMOTE_PORT"

case "${1:-push}" in
  push) rsync -az --delete -e "$ssh_cmd" --exclude .git --exclude .env.local --exclude results/ \
          ./ "$REMOTE_HOST:$REMOTE_DIR/" ;;
  pull) rsync -az -e "$ssh_cmd" "$REMOTE_HOST:$REMOTE_DIR/results/" results/ ;;
  *) echo "usage: $0 [push|pull]" >&2; exit 1 ;;
esac
