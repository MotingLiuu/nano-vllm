# nano-vllm working rules

- This Mac checkout is the source of truth for project source and configuration. Edit code, tests, docs, `pyproject.toml`, and `.rsync-exclude` here.
- The GPU server at SSH alias `nano-gpu` is for execution. Do not edit, patch, format, commit, pull, or generate source files inside `~/nano-vllm` on the server.
- Before remote tests or benchmarks, run `scripts/remote-run.sh <program> [args...]` from this checkout. It syncs first, checks for remote source edits, then runs in the server project directory. Use `--no-sync` only after an explicit successful sync when repeatedly running commands.
- Keep data, downloaded model weights, checkpoints, caches, and the Python environment on the server. They are excluded from rsync and protected from `--delete`.
- If sync reports remote source changes, stop and inspect them. Move useful changes into this Mac checkout, then review `scripts/sync.sh --dry-run --force` before using `--force`. Never silently overwrite remote edits.
- Avoid running tools on the server that rewrite tracked source or create new project-root source files. For `uv`, use the existing environment with `uv run --no-sync ...` when appropriate; prepare or update dependencies deliberately.
