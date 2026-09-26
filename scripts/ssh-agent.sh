#!/usr/bin/env bash
# Sourced by local scripts. GUI agents can inherit an expired SSH_AUTH_SOCK;
# WezTerm keeps the active agent socket under ~/.local/share/wezterm/.
if [[ ! -S "${SSH_AUTH_SOCK:-}" ]]; then
  newest=''
  for candidate in "$HOME"/.local/share/wezterm/agent.*; do
    if [[ -S "$candidate" && ( -z "$newest" || "$candidate" -nt "$newest" ) ]]; then
      newest=$candidate
    fi
  done
  if [[ -n "$newest" ]]; then
    export SSH_AUTH_SOCK=$newest
  fi
fi
