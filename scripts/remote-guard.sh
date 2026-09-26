#!/usr/bin/env bash
set -euo pipefail

mode=${1:?expected check or snapshot}
repo=${2:?expected absolute project path}
baseline="$repo/.nano-sync-baseline"
exclude="$repo/.rsync-exclude"

if [[ ! -f "$exclude" ]]; then
  echo "Missing $exclude" >&2
  exit 2
fi

case "$mode" in
  check)
    if [[ ! -d "$baseline" ]]; then
      echo "Sync baseline is missing. Inspect the server, then run sync.sh --force." >&2
      exit 2
    fi
    # Checks file contents and path additions/deletions; ignored data and runtime files stay out.
    drift=$(rsync -rcni --delete --exclude-from="$exclude" "$repo/" "$baseline/")
    if [[ -n "$drift" ]]; then
      echo "Remote source changed since the previous sync:" >&2
      printf '%s\n' "$drift" >&2
      echo "Move changes to the Mac checkout, or review sync.sh --dry-run --force before using --force." >&2
      exit 3
    fi
    ;;
  snapshot)
    mkdir -p "$baseline"
    rsync -rc --delete --exclude-from="$exclude" "$repo/" "$baseline/"
    ;;
  *) echo "Expected check or snapshot" >&2; exit 2 ;;
esac
