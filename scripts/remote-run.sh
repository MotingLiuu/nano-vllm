#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$repo/scripts/ssh-agent.sh"
host=${NANO_SSH_HOST:-nano-gpu}
remote_input=${NANO_REMOTE_DIR:-'~/nano-vllm'}
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10)
sync_first=true

if [[ "${1:-}" == --no-sync ]]; then
  sync_first=false
  shift
fi
if [[ "${1:-}" == -- ]]; then shift; fi
if (($# == 0)); then
  echo "Usage: scripts/remote-run.sh [--no-sync] [--] command [arguments...]" >&2
  exit 2
fi

if [[ "$sync_first" == true ]]; then
  "$repo/scripts/sync.sh"
fi
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

quote_arg() {
  local value=$1
  value=${value//\'/\'\\\'\'}
  printf "'%s'" "$value"
}

command_line=''
for arg in "$@"; do
  command_line+=" $(quote_arg "$arg")"
done
remote_command="cd '$remote_dir' && exec$command_line"
# A login shell provides the same PATH as the interactive server shell (uv is in ~/.local/bin).
ssh "${ssh_opts[@]}" "$host" "bash -lc $(quote_arg "$remote_command")"
