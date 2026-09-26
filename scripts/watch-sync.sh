#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
if ! command -v fswatch >/dev/null 2>&1; then
  echo "Install fswatch on the Mac first: brew install fswatch" >&2
  exit 2
fi

"$repo/scripts/sync.sh"
echo "Watching $repo for source changes. Stop with Ctrl-C."
ignore_re='/(\.git|\.venv|venv|env|data|datasets|models|checkpoints|weights|outputs|runs|wandb|logs|\.nano-sync-baseline|\.nano-sync-backups|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache)(/|$)'
fswatch -m kqueue_monitor -E -r -o -l 1 \
  --prune="$ignore_re" --exclude="$ignore_re" \
  "$repo" | while IFS= read -r _; do
    "$repo/scripts/sync.sh"
  done
