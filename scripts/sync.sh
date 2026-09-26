#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$repo/scripts/ssh-agent.sh"
host=${NANO_SSH_HOST:-nano-gpu}
remote_input=${NANO_REMOTE_DIR:-'~/nano-vllm'}
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
dry_run=false
initialize=false
force=false

usage() {
  cat <<'EOF'
Usage: scripts/sync.sh [--dry-run] [--init] [--force]
  --dry-run  Preview changes without modifying the server.
  --init     Adopt an existing remote directory for the first sync.
  --force    Bypass the remote-edit guard after reviewing the preview.

Overrides: NANO_SSH_HOST (default nano-gpu), NANO_REMOTE_DIR (default ~/nano-vllm).
EOF
}

while (($#)); do
  case "$1" in
    --dry-run) dry_run=true ;;
    --init) initialize=true ;;
    --force) force=true ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

if [[ ! "$host" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "NANO_SSH_HOST must be a plain SSH alias." >&2
  exit 2
fi

remote_home=$(ssh "${ssh_opts[@]}" "$host" 'printf %s "$HOME"')
case "$remote_input" in
  '~/'*) remote_dir="$remote_home/${remote_input#\~/}" ;;
  /*) remote_dir=$remote_input ;;
  *) echo "NANO_REMOTE_DIR must be absolute or start with ~/" >&2; exit 2 ;;
esac
if [[ ! "$remote_dir" =~ ^/[A-Za-z0-9._/-]+$ || "$remote_dir" == / || "$remote_dir" == "$remote_home" || "$remote_dir" == *"/../"* || "$remote_dir" == */.. ]]; then
  echo "Unsafe remote directory: $remote_dir" >&2
  exit 2
fi

if ! ssh "${ssh_opts[@]}" "$host" 'command -v rsync >/dev/null'; then
  echo "rsync is required on the GPU server." >&2
  exit 2
fi

marker="$remote_dir/.nano-sync-owner"
guard="$remote_dir/scripts/remote-guard.sh"
owned=false
if ssh "${ssh_opts[@]}" "$host" "test -f '$marker'"; then
  owned=true
  if [[ "$force" == false ]]; then
    ssh "${ssh_opts[@]}" "$host" "bash '$guard' check '$remote_dir'"
  fi
elif [[ "$initialize" == false && "$dry_run" == false ]]; then
  echo "First sync requires --init. Review scripts/sync.sh --dry-run first." >&2
  exit 2
fi

# Preserve code permissions and timestamps, but never copy macOS owner/group IDs.
rsync_opts=(-rlptz --delete --itemize-changes --exclude-from="$repo/.rsync-exclude")
transport='ssh -o BatchMode=yes -o ConnectTimeout=10'
if [[ "$dry_run" == false && "$owned" == true ]]; then
  preview=$(rsync "${rsync_opts[@]}" --dry-run -e "$transport" "$repo/" "$host:$remote_dir/")
  # Sync state directories may change only the project root's directory timestamp.
  meaningful=$(printf '%s\n' "$preview" | sed '/^\.d..t.... \.\/$/d')
  if [[ -z "$meaningful" ]]; then
    echo "Source already synchronized."
    exit 0
  fi
fi
if [[ "$dry_run" == true ]]; then
  rsync_opts+=(--dry-run)
else
  ssh "${ssh_opts[@]}" "$host" "mkdir -p '$remote_dir'"
  backup_dir="$remote_dir/.nano-sync-backups/$(date -u +%Y%m%dT%H%M%SZ)-$$"
  # This rsync version does not reliably combine --backup-dir with --delete.
  # Copy the previous source first, then let the actual sync propagate deletions.
  ssh "${ssh_opts[@]}" "$host" "mkdir -p '$backup_dir/source' && cat > '$backup_dir/exclude' && rsync -a --exclude-from='$backup_dir/exclude' '$remote_dir/' '$backup_dir/source/'" < "$repo/.rsync-exclude"
fi

echo "Mac source: $repo/"
echo "GPU target: $host:$remote_dir/"
rsync "${rsync_opts[@]}" -e "$transport" "$repo/" "$host:$remote_dir/"

if [[ "$dry_run" == false ]]; then
  ssh "${ssh_opts[@]}" "$host" "printf 'nano-vllm mac-source v1\n' > '$marker' && bash '$guard' snapshot '$remote_dir'"
  echo "Sync complete. Replaced or deleted remote files, if any, are saved in $backup_dir"
fi
