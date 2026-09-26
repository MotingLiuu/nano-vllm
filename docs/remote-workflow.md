# Mac 编辑、GPU 服务器运行

本地仓库是唯一源码源头：`~/workspace/nano-vllm`。`scripts/sync.sh` 单向同步到服务器 `~/nano-vllm`；服务器上的 `data/`、`.venv/`、模型权重和缓存留在服务器。测试与 benchmark 用 `scripts/remote-run.sh` 经 SSH 执行。

## 一次性准备

`~/.ssh/config` 中使用以下配置。私钥有口令；先在 Mac 终端运行 `ssh-add --apple-use-keychain ~/.ssh/id_rsa`，输入口令，再试 `ssh nano-gpu 'printf ok'`。不要把口令写进脚本。

```sshconfig
Host nano-gpu
    HostName connect.westd.seetacloud.com
    User root
    Port 10196
    IdentityFile ~/.ssh/id_rsa
    IdentitiesOnly yes
    AddKeysToAgent yes
    UseKeychain yes
    ServerAliveInterval 30
```

已确认 Mac 的 `main` 是目标源码，并于 2026-09-26 用 `./scripts/sync.sh --init` 完成首次同步。今后在本地仓库运行：

```bash
./scripts/sync.sh --dry-run   # 预览将要变更的远程源码
./scripts/sync.sh             # 手动同步
./scripts/remote-run.sh uv --version  # 自动先同步，再在服务器执行
```

之后改完本地代码，直接运行 `./scripts/remote-run.sh uv run --no-sync python -m pytest`；它会先同步。服务器已有的 `.venv` 可以继续使用。若需要安装依赖，在服务器环境中单独处理，并留意 `uv sync` 可能生成 `uv.lock`：这属于源码文件，应放到 Mac 仓库后再同步。日常运行用 `uv run --no-sync` 避免它自动改动项目文件。目前服务器的 `.venv` 还没有安装 PyTorch；待 GPU 可用后，可在服务器环境中运行 `./scripts/remote-run.sh --no-sync uv pip install --python .venv/bin/python -e .` 安装项目依赖。

`scripts/sync.sh` 默认启用 `--delete`，所以本地删除的源码也会从服务器删除。`.rsync-exclude` 中的路径受保护，脚本没有使用 `--delete-excluded`。每次正式同步前会将服务器原有源码存入 `.nano-sync-backups/`，包括即将被替换或删除的文件。它还会将同步后的源码存为 `.nano-sync-baseline/`，下一次同步前核对服务器源码；如果服务器端有人直接改了源码，同步会停止。检查后可以用 `./scripts/sync.sh --dry-run --force` 预览，再用 `./scripts/sync.sh --force` 覆盖。

`.git/` 不同步，所以服务器仍显示为原来的 `hybrid` 分支，`git status` 可能显示工作树改动。源码版本以 Mac 的 `main` 为准，不在服务器提交或拉取代码。

## 可选自动同步

Mac 上安装 `fswatch` 后运行 `./scripts/watch-sync.sh`。它先同步一次，然后在本地文件变化时再次同步。运行前必须完成首次 `--init`。如果更喜欢手动控制，只用 `sync.sh` 和 `remote-run.sh` 即可。

脚本默认使用 `nano-gpu` 和 `~/nano-vllm`；如服务器变化，可通过 `NANO_SSH_HOST`、`NANO_REMOTE_DIR` 临时覆盖。不要在服务器直接编辑项目源码；服务器上的数据、权重、运行环境和生成日志可单独维护。

## 在 Antigravity 中使用

把 Mac 上的 `~/workspace/nano-vllm` 作为 Antigravity 工作区打开。项目根目录的 `AGENTS.md` 会作为工作区规则自动读取，指示 Agent 在 Mac 修改代码，并用 `./scripts/remote-run.sh ...` 在服务器执行。Antigravity 终端可以先运行 `./scripts/remote-run.sh --no-sync uv --version` 验证连接；首次同步已经完成，所以此命令可跳过同步。脚本会在 GUI 进程继承到失效的 `SSH_AUTH_SOCK` 时查找当前 WezTerm 代理。若 Antigravity 的终端沙箱拦截 SSH，按其权限提示批准该命令的网络访问即可，不需要关闭整个沙箱。当前服务器没有可用的 `/dev/nvidia*` 设备，GPU 测试要等实例提供 GPU 后再运行。
