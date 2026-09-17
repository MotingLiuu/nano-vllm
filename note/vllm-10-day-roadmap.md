# 从手写 LLM 到推理引擎：适合 MotingLiuu 的 10 天项目路线

> 更新日期：2026-09-17。当前默认主模型为 Qwen3.5-2B，参数量允许降低，9B 已移出必做范围。本版替代之前的零基础路线，根据你补充的课程、CS336 实践及公开 GitHub 内容重新安排。本文是执行计划，未替你安装环境、运行参考 PR 或验证性能。
>
> 已确认：学完 CS224N、CS224R、CS61A、CS61C；正在学 6.1810；完成 CS336 第一个手写 LLM lab；熟练 Python/PyTorch，熟悉 tokenizer/embedding，了解 KV cache；每天 8–10 小时；可用 3090 级 GPU。按单卡约 24 GB 显存规划，以实际设备为准。

## 1. 从哪里开始

**你的起点：直接读 `Sequence → Scheduler → BlockManager → ModelRunner`，用一次真实运行追踪请求和状态。** 小模型 smoke test 只负责检查环境，不再占据一天的学习目标。把你 CS336 中的模型 forward/generate 与这条调用链对照：模型计算你已有实践，新增内容是请求在多轮执行中的身份、资源和进度管理。

你的项目是：**在 nano-vLLM 上集成 Qwen3.5-2B，并实现两种状态的联合管理、多模态输入、联合前缀缓存和 decode CUDA Graph。**

### 1.1 对你当前能力的判断

你可以直接进入源码驱动的推理系统项目。课程与 Lab 经历支持跳过基础教学，但不能直接推出你已掌握 GPU profiling、异步生命周期或 GDN 状态恢复；用下面的针对性实验补齐即可。

| 你已有的基础 | 可以直接复用 | 本项目新增的能力 |
|---|---|---|
| CS336 Lab 1 + PyTorch | 模型 forward、张量 shape、数值对照和测试 | 把一次 forward 放进有状态、多轮、多请求的 runtime |
| CS224N | 阅读 attention 与模型结构 | GDN 的 recurrent/conv state、视觉位置语义 |
| CS61A | 模块分解、数据结构、接口 | 明确请求、调度器、内存管理器的职责 |
| CS61C | 用内存布局和访存成本理解代码 | GPU 上的 paged KV、metadata、launch overhead |
| 6.1810 正在学习 | 用资源所有权、间接寻址的思路分析 runtime | 引用计数、复用、抢占、恢复的不变量；不预设你已完成相关 xv6 lab |
| CS224R | 研究与实验背景 | 本项目直接用得较少，不额外安排 RL 学习 |

我实际查看了你的 [CS336 仓库](https://github.com/MotingLiuu/CS336)，以及其指向的 [Lab 1 model.py（固定提交）](https://github.com/MotingLiuu/cs336_assignment1-basics/blob/33b76a799a0cc5847c42eb5680f3c893fa8c98be/src/cs336_basics/model.py)，看到了 attention、优化器、checkpoint 等实现。也确认已有 [nano-vLLM 仓库](https://github.com/MotingLiuu/nano-vllm)。这些用于选择学习起点；本次没有运行你的测试、完整审计所有作业，或认定仓库内全部功能都已完成。CS336 中存在 assignment2 子模块，不等于你已做完 Assignment 2。

### 1.2 与上一版相比，具体改什么

- 删除 token、embedding、张量操作、基础 Transformer 的学习和自测。
- Day 1 从“跑出文本”升级为“解释调度状态机、物理块映射，完成一次资源不足实验”。
- Day 2 除了 2B 对齐，还完成 GDN 分块等价实验和稳定 slot 的最小接入，降低 Day 3 工作量。
- Day 5 提前写联合缓存的恢复 contract，完成一条 FP32 恢复实验；Day 6 专门做查找、共享和生命周期。
- 预留 10–20 小时浮动时间给依赖、数值和显存排错，不把省下的基础课时间换成更多新 feature。

三个代码来源的分工：

| 名称 | 它是什么 | 在项目中的用途 |
|---|---|---|
| Qwen3.5 | 模型结构与权重，决定输入如何变成下一个 token | 你要支持的模型 |
| Transformers | 模型加载、Processor 和模型执行参考实现 | 正确性参照 |
| vLLM / nano-vLLM | 管理请求、显存和批处理的推理引擎；nano-vLLM 是独立的轻量实现 | 学习原理，并以 nano-vLLM 为改造底座 |

**10 天可作为完成整合原型的冲刺目标，前提是参考实现能在前两天跑通。** 对你来说，主要风险已经是依赖适配、跨模块状态正确性与显存，而非理解基本 LLM。保留全部功能目标，每一步按实验验收；十天不等于生产级稳定性，也不等于必然跑完超出硬件容量的性能矩阵。

### 原材料中已经核对与需要修正的地方

- PR #232 存在，本次查阅仍为 Open；描述包括多模态、视觉缓存及 recurrent slot pool。它是候选起点，不是经过本项目验证的完整底座。[PR #232](https://github.com/GeeeekExplorer/nano-vllm/pull/232)
- 本版主模型 2B 为 24 层：18 层 linear attention、6 层 full attention。原材料中“24 GDN + 8 Full Attention”属于 9B，不再作为本版要求。[2B 配置](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)
- qLLM 的 StateManager 思路可参考，但它的目标是 35B-A3B MoE，不能原样搬到 2B。其 README 明确说明启用 StateManager 时禁用了 prefix cache，所以不能把它当成联合缓存的现成实现。[qLLM](https://github.com/RLS-ResearchLab/qLLM)
- 当前 nano-vLLM 主线 scheduler 已有 chunked prefill 逻辑，但一轮返回 prefill 或 decode；“支持 continuous batching”和“同一轮混合 prefill/decode”不是同一个承诺。先检查你锁定的版本再决定改什么。[scheduler 源码](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/engine/scheduler.py)
- “快照是冷数据，所以 BF16 安全”不成立。恢复后的状态仍会参与后续计算，必须实测误差。
- “前缀命中”必须同时恢复同一 token 边界的 KV、recurrent state、conv state，以及必要的位置元数据；只有 KV 命中不够。

## 2. 保留什么，怎样控制工程量

| 最终模块 | 必须交付 | 控制实现成本的方法 |
|---|---|---|
| Hybrid runtime | 2B 混合层；Paged KV；recurrent/conv state；变长 packed prefill；chunked prefill；continuous batching；双状态抢占 | 复用模型与算子，只实现状态和调度适配 |
| 多模态 | 文本+单图；Processor；placeholder 对齐；vision encoder/merger；三轴 mRoPE；视觉 embedding cache | 复用 HF 语义与 PR；图像编码先在 prefill 前完成 |
| 联合前缀缓存 | 最长可恢复前缀；KV+GDN/conv snapshot；FP32 对照与 BF16 模式；双引用计数；按容量 LRU；频次准入 | 先支持完整 block 边界，快照边界与 prefill 切分对齐 |
| CUDA Graph | decode 的 1/2/4/8/16 桶；固定 workspace；动态 metadata；请求进出时 replay | 先只捕获 decode，prefill 和 vision 保持 eager |
| 验证 | HF greedy 对齐；机制回归；TTFT/TPOT/吞吐/缓存命中指标 | 测试从第一天累计，不等第九天才开始 |

本次不增加：自己写 GDN CUDA kernel、多 GPU 并行、视频、多图、训练、量化、完整 HTTP 服务或 Web UI。这些不属于你给出的核心验收目标。

抢占先选一种完整策略：**释放两类运行状态，保留逻辑 token 历史，之后从有效联合前缀恢复或重算。** 不额外做 CPU swap。重算必须包含已经生成、但仍需成为上下文的 token，并明确最后一个 token 是否已经被模型消费。

## 3. 开始前的资源与环境检查

### 3.1 GPU 决策

Mac 可以编辑代码、读源码和跑部分 CPU 单元测试；本路线的 CUDA 算子与 CUDA Graph 验收需要远程或本地 NVIDIA GPU。

**主模型改为 Qwen3.5-2B（BF16）；Qwen3.5-0.8B 用于快速调试，4B 仅作可选扩展。9B 不再是交付要求。** 这是针对十天周期与单张 3090 的工程选择，不是已实测的性能结论。

| 模型 | 在这十天里的用途 | 是否必须 |
|---|---|---|
| Qwen3-0.6B | Day 1 检查未改动的 nano-vLLM 底座；不能验证 GDN/多模态 | 底座尚未跑通时使用 |
| Qwen3.5-0.8B | 重复跑短 fixture、诊断 loader/状态/位置问题 | 可选；不要为它维护另一套引擎 |
| Qwen3.5-2B | 全部功能、HF 对齐、最终演示与 benchmark | 默认主模型 |
| Qwen3.5-4B | 功能冻结后有余量再测配置通用性 | 可选，不挤占 Day 9–10 |

已核对 [0.8B 配置](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json) 和 [2B 配置](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)：两者均有 vision 配置、三轴 interleaved mRoPE，以及 24 层文本 decoder（18 层 linear attention + 6 层 full attention）。因此降低参数量仍可覆盖本项目的 hybrid、多模态和缓存机制。不要改用普通纯 attention 小模型替代最终验证。

**统一使用配置驱动的实现。** 从 `layer_types` 计算哪些层分配 KV、哪些层分配 GDN/conv state，从配置读取各自 head 数与维度。特别核对小模型的 `tie_word_embeddings=true`：checkpoint 可能不单独存储输出 head，loader 需要按共享权重语义处理。PR 能跑某个尺寸，不代表自动兼容所有尺寸；Day 2 的第一项工作就是核查这些差异。

### 2B 在 3090 上的预算

标准 RTX 3090 配备 24 GB 显存。[NVIDIA 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/)

按约 2B 参数估算，BF16 权重数量级约 4 GB（十进制）；准确占用以 checkpoint、实际加载和 allocator 测量为准。依据上述 2B 配置、单 GPU、BF16 KV 及 FP32 recurrent state 推算：

```text
KV 每 token = 6 层 × 2(K/V) × 2 KV heads × 256 head_dim × 2 bytes
             = 12 KiB
单请求 16,384-token 输入 KV ≈ 192 MiB
16 个互不共享请求的输入 KV ≈ 3 GiB

recurrent state 每请求 = 18 层 × 16 value heads × 128 × 128 × 4 bytes
                       ≈ 18 MiB
16 个活动请求的 recurrent state ≈ 288 MiB
单份 BF16 recurrent snapshot ≈ 9 MiB
```

这是基于配置和上述状态布局的估算，不含 conv state、生成 token 的额外 KV、block padding、vision 中间张量、Graph 内存池、算子 workspace 和缓存条目。需核对实际实现的分配布局。100 份这样的 BF16 recurrent 快照就约 900 MiB，因此小模型仍需要容量管理。

另一个有用的结论：0.8B 与 2B 在上述 KV/GDN 维度上相同，减小权重不意味着这些状态也按参数量同比例缩小。用 0.8B 可以缩短部分计算和加载时间，但不能免除状态容量测试。

**现在可以把单卡 2B 的 16K、B=1/2/4/8/16 作为要尝试完成的正式矩阵，无需预设换大卡。** 先验证短输入，再逐步加长；上述估算表明预算明显宽松，但未运行前不保证全部组合成功。

| 验收线 | 执行方式 | 如何记录 |
|---|---|---|
| 功能机制 | 2B：先 128–512 输入 token、16–32 输出 token、B=1；再扩展 B=2/4/8/16 | 逐项确认正确性；0.8B 结果单独标注 |
| 性能负载 | 2B：2K → 8K → 16K；12K shared prefix 与 50% trace 单独测 | 记录实际峰值显存、OOM、吞吐和延迟 |

3090 开发纪律：

1. HF 与引擎默认分进程依次运行，保存参考结果到 CPU 文件，减少对显存预算和 benchmark 的干扰；不再因为 9B 双模型容量而受限。
2. 初始上下文和 batch 仍设小值，并给 KV、state pool、snapshot、视觉缓存分别设容量。不要因为权重变小就无限缓存。
3. prefill 只计算/保留所需位置的 logits；小模型也有大词表，整段 16K 的词表投影可能成为临时显存峰值。
4. 图文 fixture 固定图像尺寸和 Processor 配置；扩大分辨率单独测。
5. Graph 逐桶 capture 并测内存，不预设所有 workspace 零成本。
6. 出现 OOM 先定位池预分配、长 prefill、视觉 token 和临时张量，不增加量化/offload 主线。

若 2B 有短期无法解决且与尺寸相关的兼容问题，可以在 Day 2 明确将 0.8B 升为最终模型，并在该尺寸重建全部 HF 基线与 benchmark 配置。0.8B 同样可以是有效交付；不能混用两个尺寸的正确性与性能数据。无须为了参数量返回 9B。

首日记录：GPU 型号/显存、驱动、Python、PyTorch、CUDA runtime、Transformers、flash-attn/FLA/相关算子版本、仓库 SHA、模型 revision、空闲磁盘和下载耗时。

### 3.2 固定开发基线

以下命令是在你选定的 Linux 开发机上执行的起步操作；安装依赖以锁定版本的 README/项目配置为准，不承诺任意版本组合兼容。

```bash
nvidia-smi
python --version
git clone https://github.com/MotingLiuu/nano-vllm.git
cd nano-vllm
git remote add upstream https://github.com/GeeeekExplorer/nano-vllm.git
git rev-parse HEAD
git switch -c project/qwen35-hybrid
git fetch upstream pull/232/head:reference/qwen35-pr232
git rev-parse reference/qwen35-pr232
git diff --stat HEAD...reference/qwen35-pr232
```

已有本地 checkout 就复用，不重复 clone；已有 upstream 就先检查地址，不重复 add。先用 `git status` 确认现有修改，不覆盖你的工作。把两条 SHA 写入 `docs/environment.md`。需要运行 PR 示例时可为参考分支建立单独 worktree/环境，先验证它，避免直接把整个 PR 合进学习底座后无法定位问题。

依赖就绪后检查：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())'
```

下载小模型后，按上游 `example.py` 运行。首个 smoke test 用 `enforce_eager=True`、单 GPU、短输入和短输出；目标是证明推理能工作。使用底座支持的 greedy 路径时确认 `temperature=0` 的实际实现。

**锁版本规则：** 保存成功运行时的包版本；固定 nano-vLLM、参考 PR、Transformers 和模型 revision。十天中不要随手 `pip install -U` 全量升级。

### 3.3 从你的现有代码迁移理解

取自己的 CS336 模型，只画接口对应关系，不要求把它移植进 nano-vLLM，也不重新训练：

```text
你熟悉的：token IDs → model.forward → logits → sampling
新增外层：request → scheduler → memory allocation → packed metadata
                                  ↓
                             model.forward
                                  ↓
                  update progress / keep state / release resources
```

模型权重跨请求共享；KV/GDN 是请求相关状态；batch metadata 是本轮执行数据。这三类生命周期是第一天要分清的东西。

## 4. 只补四块知识，每块用一个实验学会

| 缺口 | 你要掌握到什么程度 | 最小实验与时间上限 |
|---|---|---|
| Paged KV 与调度 | 从逻辑 token 位置查到物理 slot，解释分配、共享、回收 | Day 1：45 分钟 CPU 小实验，非第二套引擎 |
| GDN 状态语义 | 看懂输入/输出/recurrent/conv shape，理解跨 chunk 的状态传递 | Day 2：60–90 分钟调用现成算子的整段/分段对比 |
| 多模态位置 | 区分 request offset、chunk offset、packed offset 与三轴位置 | Day 4–5：用两请求和跨图片边界的 chunk 核对索引 |
| GPU 执行与测量 | 区分 CPU 提交与 GPU 完成、capture 与 replay，找到 launch/访存瓶颈 | Day 1 做短 profile，Day 7–8 做固定 buffer/replay 实验 |

不复习 tokenizer、embedding、基础 attention 和反向传播。只有看到陌生机制才定向阅读，单次资料学习限定 30–60 分钟，之后必须回到一个具体测试。

### 实验 A：Paged KV，用你已有的内存知识理解

只在 CPU 上模拟 block size=4，总共 5 个 block：A 长度为 6，B 长度为 3。写出分配表、空闲表，以及 A 的逻辑位置 5 对应的物理 slot：

```text
logical_block = token_position // block_size
physical_block = block_table[logical_block]
physical_slot = physical_block * block_size + token_position % block_size
```

追加 token，直到需要新 block；释放 B，再给 C 分配。检查无重复分配、无释放后引用、全部结束后资源归还。不用写 GPU kernel。

可以借用页表的间接寻址思路，但不要把 GPU KV block 当成硬件虚拟内存页：这里通常由引擎管理映射，并由 attention 算子消费 block table。前缀共享也必须遵守块是否完整、是否可写的规则，不能无条件共享尾块。

### 实验 B：GDN，用状态转移理解

将官方/现成算子的状态转移抽象为 `F(tokens, initial_state) → outputs, final_state`，其中 state 包含 recurrent 和 conv 两部分。

验证 `F(A+B, s0)` 与先 `F(A, s0)` 再 `F(B, sA)` 的对应输出和最终状态。使用锁定实现支持的 dtype，记录数值容差。先在小张量上验证，再进入真实 2B。这比手写一个简化但不等价的 GDN 公式更直接。

## 5. 十天总览：80 小时核心任务，另留 10–20 小时缓冲

每行按 8 小时核心任务安排，正常休息另计。额外时间优先给前一日未解决的问题，不增加新模块。资料学习融入任务，不另开一整套课程。

| 天 | 前半天：4 小时 | 后半天：4 小时 | 必须留下的证据 |
|---|---|---|---|
| 1 | 底座 smoke test、请求/内存主链、45 分钟分页实验 | 调度 trace、强制资源不足、短 profile、PR 差距表 | 状态转移表、运行日志、模块差距表 |
| 2 | HF 参考、GDN 分段等价、状态接口 | PR 文本适配、单请求稳定 slot、2B 对齐 | 参考输出、状态 shape/dtype、文本对比 |
| 3 | 变长 packed/chunked prefill、双资源 admission | 请求重排、slot 复用、抢占重算与回归 | 调度状态测试 |
| 4 | 复用 vision/Processor，单图 eager | 三轴位置、chunk/packed 索引核对 | 单图对齐、位置追踪表 |
| 5 | 视觉缓存、多模态与 chunked 联合回归 | prefix contract、单前缀 FP32 snapshot/restore 实验 | 多模态门槛 + 状态恢复证据 |
| 6 | 最长可恢复前缀、联合 entry 与恢复路径 | 引用计数、共享快照隔离、抢占与缓存联动 | 完整联合缓存正确性 |
| 7 | BF16 对照、容量 LRU、频次准入 | 内存压力回归、Graph workspace contract 与小实验 | 精度/容量报告、Graph 接口约束 |
| 8 | 接入 decode Graph，逐桶捕获 | 动态 batch、dummy lane、slot 复用回归 | 1/2/4/8/16 桶结果；冻结功能 |
| 9 | 全链路正确性与负例测试 | 固定 trace benchmark，保存原始数据 | 正确性报告、实测性能 |
| 10 | 干净环境复现、README 与贡献说明 | 演示与独立讲解；仅修复现问题 | 可复现交付 |

依赖顺序：`文本正确 → 调度状态正确 → 多模态正确 → 联合缓存正确 → Graph 正确 → 性能测量`。Day 5 后半天的恢复实验以前半天正确性过关为条件；未通过就用这段时间排错，Day 6 再接缓存。

## 6. Day 1：从模型生成循环走到请求调度器

**今天要回答：请求为什么能在不同 batch 中进出，历史状态仍然正确？**

### 具体操作

1. **0–1 小时：** 在已有 nano-vLLM 仓库建开发分支，锁定环境和 SHA，启动权重下载，用 0.6B 做 eager smoke test。依赖未就绪则先读代码，不等待下载空转。
2. **1–3 小时：** 读 Sequence、Scheduler、BlockManager、ModelRunner；结合实验 A 写出分配/追加/释放过程。模型层只看与已有 CS336 实现不同的接口。
3. **3–4 小时：** 手推 A/B/C 三个请求的状态变化，分别记录 waiting、部分 prefill、decode、finish/preempt 的资源。
4. **4–6 小时：** 加可关闭的调度 trace，主动限制 token budget/KV 容量，使日志出现分块或资源不足后的恢复。查看你的锁定版本是否真有相应行为；缺失记入差距表。
5. **6–7 小时：** 用短输入做一次 `torch.profiler`，区分 prefill 与 decode 的 CPU/GPU 时间。只定位阶段，不调 kernel；输出几行结论，不追求漂亮图表。
6. **7–8 小时：** 查看 PR #232 的模型、runner、scheduler diff，制作“上游已有/PR 提供/需要新增/待验证”表；标出 GDN 算子和依赖要求，给 Day 2 排雷。

推荐阅读顺序，路径以锁定版本为准：

```text
engine/sequence.py       → 请求状态、逻辑长度与进度
engine/scheduler.py      → admission、prefill/decode、抢占和完成
engine/block_manager.py  → 分配、追加、引用与回收
engine/model_runner.py   → prepare_prefill / prepare_decode / run
engine/llm_engine.py     → 串起完整循环
models/qwen3.py / layers/attention.py → 只看模型与 runtime 的边界
```

日志至少包含：`step, request_id, phase, num_computed, num_scheduled, block_table, free_blocks`。以后加 `state_slot_id`。性能测量时关闭详细 trace。

### 验收

- [ ] 能逐行解释一轮 `schedule → run → postprocess`。
- [ ] 区分序列总长度、已计算长度、本轮计算长度，知道三者不总是相同。
- [ ] 能从逻辑位置手算 KV 物理 slot。
- [ ] 能说明请求结束和资源不足时，何处更新状态与释放资源。
- [ ] 已拿到真实调度日志及一份实现差距表，而不只是架构图。

当天的目标不是“吃透所有源码”，而是能沿状态变化定位代码。依赖问题超过 2 小时就保存完整错误，使用缓冲时间定向解决；不要把安装卡顿误判为能力不足。

## 7. Day 2：GDN 状态语义、2B 文本基线与最小 slot 接入

**今天只做单请求、纯文本、eager、禁用 prefix cache。**

### 具体操作

1. 阅读 2B 的 `config.json`，写下 layer types、attention heads、状态维度及 dtype。不要把 Qwen3 的普通 attention 参数直接套过来。
2. 依据 [HF Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5) 和模型卡，运行同一个 2B checkpoint 的 HF 文本生成。
3. 保存处理后的 `input_ids`、模型 revision、模板设置、EOS 配置、生成 token IDs。禁用随机采样，明确 thinking/chat template 设置。3090 上将参考结果存到 CPU 文件后退出 HF 进程，再运行引擎。
4. 在独立参考环境运行 PR #232 的文本路径，先读 diff 再复用；保留来源和许可证说明。
5. 对照 HF，逐项核对权重加载、层类型、norm、GDN 算子输入输出和 recurrent/conv state。把工作范围限制在已有实现的适配。
6. 用 5 个短 prompt，每个生成 16–32 个 token。逐步比较输出；出现首次分歧时，比较同一历史输入下的 logits。
7. 完成实验 B，核对 recurrent 与 conv 都能跨分段连续传递。对照实际实现识别 decay/gate、状态更新和输出读取，不要求推导训练算法或优化 kernel。
8. 接入/确认单请求稳定 `state_slot_id`，明确 allocate/reset/release。复用 PR 中已有的 slot pool 时，先写测试验证其行为，避免再造第二套状态管理器；多请求压力放在明天。

必须写出的状态关系：

```text
Full Attention：历史 KV 随上下文增长
GDN：每层持有 recurrent state + conv state
prefill(x[0:n]) → state_after_n
decode(state_after_n, x[n]) → state_after_n_plus_1
```

### 验收

- [ ] HF 与引擎使用完全相同的模型和输入 IDs。
- [ ] 5 个短样例完成逐 token 比较；任何分歧都有定位记录。
- [ ] 能说出模型每层状态的 shape、dtype 和所有者。
- [ ] 小规模整段/分段 GDN 实验通过，并能指出 conv 缓冲的传递位置。
- [ ] 单请求 slot 分配、初始化、释放已走通。
- [ ] 保存 `tests/fixtures/`、`docs/model-contract.md`。

**卡住处理：** 一旦输出不同，先关 Graph/缓存、缩到 batch=1，不要同时改模型、调度器和 kernel。当天仍跑不通 2B，先判断是共同的 runtime 问题还是尺寸适配问题。前者必须修复；后者可按第 3 节将 0.8B 明确设为最终模型并完整验收，不能只换模型掩盖状态错误。

## 8. Day 3：请求重排时，状态仍然属于原请求

**今天的核心：`batch_index != request_identity`。**

承接 Day 2 的单请求 slot，把它扩展成多请求资源管理。你正在学习的操作系统内容可以提供类比，但判断依据是这里的实际状态转移与测试，不必等完整学完 6.1810 才开工。

### 先写接口，再改代码

以下是建议接口，不是上游现成 API：

```python
allocate(request_id) -> slot_id
release(request_id)
reset(slot_id)
snapshot(slot_id, boundary) -> Snapshot
restore(snapshot, destination_slot)
```

Sequence 保存稳定的 `state_slot_id`，ModelRunner 只把本轮请求映射转成 GPU metadata。Snapshot 要包含 recurrent 和 conv 两类状态，后续才可用于缓存。

### 具体操作

1. 为每个正在计算或保留计算进度的请求分配一个 slot。注意：部分 prefill 请求可能仍在 waiting 队列，也需要保留 slot，不能只看 RUNNING 标签。
2. 明确 slot 的初始化、完成释放、取消释放与重用规则；用小容量池强制触发复用。
3. 用长度 3、5、9 的输入验证 packed prefill。检查每个请求的 cumulative offsets、状态与最后有效 token 的 logits。
4. 用相同输入比较 full prefill 与分块处理。第 k 个 chunk 的 KV 可见范围到当前已计算末尾，不能包括未来未写入区域。
5. 加 token budget，把长输入分成多轮；只有输入处理完成才能采样下一个输出 token。
6. 强制资源不足，触发释放 KV+slot 后重算。两种资源必须一起检查、一起提交分配；分配失败不能泄漏另一种资源。

### 必做测试

| 场景 | 应成立的结果 |
|---|---|
| A/B/C → C/D，batch 顺序变化 | C 的状态没有变成 A 的状态 |
| A 完成后 D 复用其 slot | D 等价于独立 fresh run |
| full prefill vs 3 个 chunk | 同一输入历史下终态和输出在预定容差内一致 |
| 被抢占后恢复/重算 | 生成序列与无抢占基线一致或分歧已定位 |
| KV 有空位但 state slot 不够 | 请求等待或合法抢占，没有半分配 |
| 请求全部结束/取消 | 活动引用与 slot 回到初始计数 |

交付：`docs/state-lifecycle.md`、状态管理实现和上述测试。先用人工限制资源触发抢占，不必等真的 OOM。

## 9. Day 4：把一张图送进 decoder

先读 Processor 和 HF forward 的接口，再参考 PR。不要从重写 vision tower 开始。

```text
text + image
  → Processor
  → input_ids + pixel_values + image_grid_thw
  → vision encoder + patch merger
  → visual embeddings
  → 在对应 image token 位置替换 text embeddings
  → hybrid decoder
```

### 具体操作

1. 固定一张本地测试图和一句简单问题，用 HF 保存基线。
2. 打印 Processor 输出的字段和 shape；明确 image placeholder 是否已展开，防止 Processor 与 runtime 重复展开。
3. 对齐 image token 数量与 merger 输出 embedding 数量，数量不符立即报错。
4. batch=1、完整 prefill 下完成替换。记录视觉位置范围以及前后普通文本的位置。
5. 文本和图文各跑一次，确保新增路径未破坏文本生成。
6. 当天就记录一份三轴 positions，并用两条不同长度请求手算 packed offsets。先提前识别位置问题，Day 5 再覆盖 chunk 切过图片区域的完整组合。

验收：能画出每个张量的流向；真实单图 prompt 能生成；与 HF 在相同输入下比较输出。今天先不打开视觉缓存，不引入多图或视频。

## 10. Day 5：位置、分块和图片缓存的正确性

### 具体操作

1. 从锁定 HF 实现理解三轴 positions 和位置增量。文本位置通常在三轴上相同，但图片后文本的续接规则必须沿用参考实现，不能简单把整个序列设成 `arange`。
2. 保存每个请求的逻辑位置，映射到本 chunk，再映射到 packed tensor。把三种坐标分别命名，避免混用。
3. 故意让 chunk 边界切过 image token 区域，只回填当前 chunk 对应的 visual embedding 切片。vision 可先整图计算一次。
4. 加视觉 embedding cache：键包含图像内容、模型 revision、Processor/resize 配置等会影响 embedding 的信息。容量按字节限制，先用普通 LRU。
5. 相同图片重复请求验证 hit；更换图片、缩放设置或模型版本验证 miss。不要用文件名作为唯一键。
6. 固定一个小型 fixture 集：建议 10 条文本 + 10 条单图，涵盖两种图像尺寸、不同 prompt 长度、不同 chunk 边界。

### 验收

- [ ] 图像 token 与 embedding 一一对应。
- [ ] 单独运行与 packed batch 的结果可对齐。
- [ ] 切分 image token 的 chunk 不会错位或重复注入。
- [ ] 视觉 cache hit/miss 与关闭缓存的结果一致。
- [ ] 图片相同/不同、预处理相同/不同的隔离测试通过。

**Day 5 是第一道硬门槛：文本、单图、调度状态仍有未解释分歧，就继续修正确性，暂停联合缓存与 Graph。**

### 通过门槛后：下午提前做联合恢复实验

用一个文本请求、一个完整 block 边界 N，不做 hash 查找、不做淘汰策略：保存该处 KV 引用、FP32 recurrent 和 conv 快照、位置与已消费长度；在独立 slot 恢复后继续处理后缀，和从头运行比较。再让两个独立请求从同一快照分叉，验证它们不会互相改写状态。

写出 `docs/prefix-contract.md` 初稿，尤其明确“state_after_N 表示已经消费多少 token”。这一步提前消除 Day 6 最大的语义风险；若多模态对齐耗时超出预算，把实验顺延，不能省略前置门槛。

## 11. Day 6：实现能真正恢复计算的联合前缀缓存

先做 FP32 snapshot。关键不是 hash 表，而是**所有状态都停在同一个已消费 token 边界**。

若 Day 5 已通过单前缀恢复实验，今天直接把它接入最长前缀查找和生命周期；先建立 KV 与 snapshot 的基础引用计数，Day 7 主要做容量淘汰、精度和压力测试。

建议 entry 保存：

```text
prefix identity / parent hash / token 验证信息
consumed_token_count
KV block references
每个 GDN 层的 recurrent snapshot + conv snapshot
继续执行所需的位置元数据
模型/处理配置身份、视觉内容身份
字节数、引用计数、最近使用时间
```

### 具体操作

1. 缓存完整 KV block 边界；hash 作为候选索引，命中后核对 token 与输入身份。
2. 多模态前缀键必须包含视觉内容与位置语义。两张不同图片可以产生相同 placeholder IDs，不能只比较 IDs。
3. 在准备缓存的 block 边界切分 prefill 并抓取状态。大 chunk 一次走到末尾，不能凭最终 GDN state 反推所有中间边界。
4. 查找“同时拥有有效 KV 和状态快照”的最长匹配前缀。只有 KV 的条目按 miss 或更短联合前缀处理。
5. 命中后 pin KV、分配独立活动 slot，把 immutable snapshot 拷入活动状态。请求不能原地修改共享快照。
6. 更新 computed length、positions 和下一个待消费 token，继续处理未缓存后缀。
7. 完整 prompt 命中时明确 logits 策略：若未保存终点 logits，退到更早有效边界，重算尾部以得到正确采样 logits。

### 必做测试

- 同一前缀、不同后缀：hit 与关闭缓存逐 token 对比。
- 只有 KV、没有 GDN snapshot：不得错误跳过 prefill。
- recurrent 正确但 conv 故意缺失：测试应能发现错误。
- 恢复边界偏移一 token：测试应失败。
- 两请求恢复同一 snapshot 后继续生成：互不污染。
- 图片不同、文本和 placeholder IDs 相同：不得误命中。

交付：`docs/prefix-contract.md`，说明边界、输入身份、恢复流程和失败处理。

## 12. Day 7：控制快照精度和显存占用

### BF16 模式

1. 保留 FP32 snapshot 作为对照，增加显式 BF16 snapshot 配置。
2. 只压缩适合压缩的浮点状态；整数 metadata 不转换。恢复时转回活动状态所需 dtype。
3. 在相同输入历史下比较 FP32 与 BF16 的 logits，再比较自由生成的 token IDs。两者分别反映局部误差和累积结果。
4. 记录最大/平均误差、top-1 匹配率、首个分歧位置及快照字节节省。不要只报告 cosine similarity。
5. BF16 不满足既定要求时默认保持 FP32，并将 BF16 标为实验功能；这不算 BF16 验收通过。

### 双引用计数与容量

明确两类物理资源的所有者：KV block 可能被活动请求和缓存条目引用；snapshot 被缓存条目及正在恢复它的操作持有。异步拷贝未完成前不能回收源快照。

容量按实际字节记账，分别记录 KV、运行状态、snapshot、视觉 embedding 与 Graph 内存。先实现简单的按字节容量 LRU，再做频次准入：维护有界频次表，达到阈值后才保留昂贵快照。新前缀即使没进缓存，也应能积累后续观察次数。

无法淘汰被 pin 的资源时，跳过新条目准入或回退到正常计算，不能超额分配或死循环。

验收：

- [ ] 活动请求持有的资源不会被淘汰。
- [ ] 短寿命请求反复进入/退出，引用计数不泄漏、不变负。
- [ ] cache 满时仍能完成请求；超大条目不被错误接纳。
- [ ] 高频前缀可准入，低频前缀不会挤满 snapshot 区。
- [ ] FP32/BF16 报告写明样例数、生成长度和限制。

### 为 Day 8 准备 30–45 分钟的小实验

用一个简单、无模型权重的 CUDA 张量操作练习 warmup/capture/replay：保持输入 buffer 地址不变，更新内容后 replay，检查输出跟着变化。然后列出真实 decode 需要固定的所有 buffer 及最大 shape。若今天精度/容量测试没过，小实验顺延；不要把玩具 Graph 的成功算作引擎验收。

## 13. Day 8：扩展 decode CUDA Graph，随后冻结功能

先理解固定地址输入缓冲、warmup、capture 和 replay 的约束。[PyTorch CUDA Graph 文档](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs)

### 具体操作

1. 只捕获单步 decode，先做 bucket=1，验证后扩展到 2/4/8/16。
2. 固定 input IDs、三轴 positions、block tables、context lengths、KV slot mapping、GDN state slot IDs 和必要 mask 的存储。
3. 每轮把真实 metadata 拷进固定 buffer；检查容量后 replay，不要更换 capture 绑定的底层地址。
4. 以 batch=3 使用 bucket=4 为例，明确 padding lane 的行为：不能写入真实请求的 KV/GDN。使用合法的隔离 dummy slots 或算子支持的屏蔽路径。
5. 验证 batch 变化 `1 → 3 → 2 → 5 → 1`，包括请求结束、slot 复用和 prefix restore。
6. 给 block table 最大宽度设上界。运行请求超过 workspace 能力时明确回退 eager，不能硬拷贝越界。
7. capture/warmup 使用隔离状态，避免改变真实请求状态；每次对照实验从等价初态开始。

### 验收

- [ ] 1/2/4/8/16 桶都跑过真实 decode。
- [ ] 3/5 等非桶大小的 batch 不污染真实请求。
- [ ] eager 与 graph 的 token/logits 对比通过既定门槛。
- [ ] 请求进出、slot 复用、缓存恢复后仍正确。
- [ ] workspace 超限行为明确且有测试。

当晚 feature freeze。之后只修 bug、做验证和整理交付。不能把一次 batch=1 capture 成功当成全部 Graph 功能完成。

## 14. Day 9：正确性回归与可信 benchmark

### 14.1 正确性策略

不要求所有 GPU 算子逐位一致，但必须区分数值误差和状态错误。先冻结 fixture、容差及判定方法，再跑对比，不能看到失败才随意放宽阈值。

标准短 fixture 以 greedy token 对齐为目标；数值容差按锁定算子建立基线，不给所有层套一个武断阈值。出现分歧时在**同一 token 历史**下比较 logits，定位首个差异；一旦自由生成 token 不同，后面的 logits 已经不是同一个输入条件。

覆盖维度：文本/单图，eager/graph，cache miss/hit，FP32/BF16，单请求/continuous batching，full/chunked prefill，无抢占/强制抢占，B=1/2/4/8/16。

先做逐开关对照，再做关键组合，避免盲目跑完整笛卡尔积。至少包含“单图 + chunked + prefix hit + slot 复用 + graph”组合回归。

刻意注入错误做独立负例测试：错误 slot、未清理旧状态、缺失 conv snapshot、错一位恢复边界、错误图片身份、stale block table。只在独立测试中破坏，不把故障留在主实现。

### 14.2 性能负载

先跑适合开发卡的短负载：输入 512 token、输出 128 token，B 从 1 逐步增加到 16，验证各机制与对照方式。不要预先假定 B=16 一定放得下。

原目标另外保留为：输入长度 16,384 token，其中共享前缀 12,288 token；B=1/2/4/8/16；另设 50% 请求可命中的固定 trace。默认输出 128 token，并记录是否忽略 EOS。

这里的 16K 指输入长度，生成后总上下文会更长。图文负载以 Processor 后实际 token 数为准。容量不够就报告 OOM/未完成该点。若最终模型已统一改为 0.8B，则整套实验使用 0.8B 并明确标注，不混入 2B 的比较表。

对照项：

| 实验 | 只改变什么 |
|---|---|
| eager baseline vs graph | GPU 提交路径 |
| cache off vs joint cache on | 前缀复用 |
| FP32 vs BF16 snapshot | 快照精度 |
| visual cache off vs on | 图片 embedding 复用 |

先单独预热模型和 Graph，再清理/构造 cache 到指定状态。50% 是请求命中比例，不代表 50% token 被复用；同时报告实际复用 token 比例。频次准入要通过预置重复请求达到阈值，并把这段预热排除在测量之外。

### 14.3 指标定义

- **TTFT**：请求提交到首个生成 token 可用的时间，包含排队；另记调度等待和 prefill 时间便于解释。
- **TPOT**：每个请求 `(末 token 时间 - 首 token 时间) / (输出 token 数 - 1)`；单 token 输出单独处理。
- **输出吞吐**：总生成 token / 测量窗口墙钟时间。输入+输出合计吞吐若提供，另列名称。
- **命中率**：实际联合命中请求数 / 可缓存请求数；视觉命中率单列。
- **资源**：峰值显存、snapshot/视觉 cache 字节、活动 slot、抢占次数。

离线 `generate()` 一次返回全部结果时，总耗时无法推导 TTFT/TPOT。需要在引擎产生各 token 时插桩。GPU 时间用 CUDA events 或在实验边界合理同步，不能把异步提交时间当执行时间；频繁逐步同步会扰动吞吐，应单独说明测量方式。

每组至少 3 次独立测量，保存请求级数据，报告 p50/p95 与运行间波动。建议 trace 含至少 100 个请求；样本不足就写明，不能夸大统计精度。

交付：`benchmarks/config.json`、请求级 JSONL、汇总表和 `docs/results.md`。没有提速也是有效结果，先解释瓶颈，不能修改数据。

## 15. Day 10：让别人能复现，让你能讲清楚

### 上午：复现与整理

1. 从干净环境或新 worktree 按 README 走一次安装、下载、smoke test、正确性测试、benchmark。
2. 列出每个功能的状态：通过 / 部分验证 / 未完成，并链接对应命令与日志。
3. 写来源与贡献表：上游原有功能、参考实现、你的适配和新增机制。
4. 保留第三方 LICENSE 与归属；自己的项目描述不把借用代码说成独立发明。

### 下午：准备一个 5 分钟演示

建议顺序：文本生成 → 单图生成 → shared prefix 命中 → 请求进出与 slot 复用 → eager/graph 对照 → 实测结果。

### 必须能独立回答

1. prefill 与 decode 分别在算什么？
2. block table 和 state slot 各解决什么问题？
3. 为何不能用 batch 下标保存 GDN 状态？
4. chunk 结束时哪些状态必须保留？
5. 抢占后要重算哪些 token？如何避免重复消费最后一个 token？
6. 为什么只有 KV 的 prefix hit 不正确？conv state 为什么也要缓存？
7. 最长匹配前缀为什么不一定是最长可恢复前缀？
8. 图片不同但 placeholder IDs 相同，缓存为什么不能命中？
9. packed offset、请求位置、三轴位置分别是什么？
10. BF16 snapshot 的节省与误差分别怎样测？
11. batch=3 replay bucket=4 时，第四条 lane 写在哪里？
12. 哪部分来自上游？哪些是你实现并验证的？
13. TTFT 改善而 TPOT 不变可能说明什么？

每题按“数据是什么 → 谁拥有 → 生命周期 → 为什么 → 做错怎样被测试发现”回答。不要背术语堆砌的答案。

## 16. 用 AI 的方式：每次只交给它一个可验收任务

建议每次给 AI 如下模板：

```text
背景：当前基于 [commit]，已通过 [现有测试]。
本次只实现：[一个明确功能]。
先阅读：[相关文件]。
接口：[输入、输出、状态所有者]。
必须保持：[3–5 条 invariant]。
验收：[最小正例、边界例、失败例]。
请先指出依赖与风险，再给最小补丁；不要无关重构。
输出：改动说明、测试命令、真实运行结果、没有验证的部分。
```

StateManager 示例：

```text
实现 request_id 到稳定 state_slot_id 的映射。
部分 prefill 的请求也必须持有 slot。
新请求不能读到已结束请求的状态。
KV 或 slot 分配任一失败都不得泄漏另一类资源。
验收 A/B/C → C/D 的重排、A 结束后复用、请求取消和池耗尽。
本次不改模型数学、不加入 prefix cache。
```

每个核心补丁提交前：写 contract → review diff → 关掉 AI 自己讲一遍 → 运行会暴露错误的测试 → 记录结果。AI 没实际跑过的测试只能写“待执行”。

## 17. 排错和时间止损

| 症状 | 第一检查点 |
|---|---|
| 单请求首 token 就不同 | 输入 IDs、模板、权重加载、位置和模型数学 |
| 首 token 对，后续不同 | state 更新、最后 token 消费边界、decode metadata |
| 单请求对，batch 错 | packed offsets、slot mapping、最后有效 token 索引 |
| 短输入对，chunked 错 | chunk 边界状态、已计算长度、KV 可见范围 |
| 文本对，图文错 | placeholder 数量、视觉替换、三轴位置 |
| cache miss 对，hit 错 | 同边界 KV/GDN/conv 恢复、图片身份、共享状态被修改 |
| eager 对，graph 错 | 固定地址、stale metadata、padding lane、buffer 宽度 |
| 跑多轮后才错或 OOM | slot 复用污染、引用泄漏、缓存容量记账 |

单个问题卡住 60–90 分钟就缩成最小复现：一个请求、短上下文、关缓存、关 Graph。保存首个错误位置再继续。不要一口气打开所有开关。

三次进度决策：

- **Day 2 结束：** 选定的最终模型（默认 2B）文本未跑通，完整十天计划已高风险。优先解决基线，不并行堆新功能。
- **Day 5 结束：** 多模态/调度未对齐，后续缓存和 Graph 延后。
- **Day 8 结束：** 冻结新增功能。未通过的项目明确列为未完成；完整目标仍保留，但需要延长工期，不能用文档替代实现。

你的 CS336 实作与 Python/PyTorch 基础允许直接进行模型差异排查，CS61C 和正在学的 6.1810 可用于理解内存和生命周期。但这些不会自动解决 reference PR 的 bug、异步 GPU 行为和 24 GB 容量限制。前三天的实际日志与对齐结果，比课程数量更能判断十天目标是否仍可实现。

冲刺期间不把完整 CS336 Assignment 2 或完整 6.1810 设为前置任务。如果需要补 profiling，只定向参考 [CS336 Assignment 2](https://github.com/stanford-cs336/assignment2-systems) 的相关部分；完成本项目后再系统补 GPU kernel 优化和分布式训练。

## 18. 最终仓库应当留下什么

以下是建议的新文件，不代表参考仓库已经有这些脚本：

```text
README.md                       # 安装、运行、复现、功能状态
THIRD_PARTY_NOTICES.md           # 来源与许可证归属
docs/
  environment.md                # GPU、版本、SHA、模型 revision
  architecture.md               # 请求、内存和执行流程
  model-contract.md             # shape、dtype、输入输出
  state-lifecycle.md            # 分配、分块、抢占、释放
  prefix-contract.md            # 匹配、边界与恢复
  results.md                    # 测量方法、结果、限制
  contributions.md              # 复用内容与自己的贡献
tests/
  fixtures/
  test_state_lifecycle.py
  test_chunked_prefill.py
  test_multimodal_alignment.py
  test_joint_prefix_cache.py
  test_graph_replay.py
benchmarks/
  config.json
  requests.jsonl
  results.jsonl
```

最终完成条件：上述原定机制有对应测试；选定最终模型（默认 2B）的文本与单图路径真实运行；缓存与 Graph 的跨请求状态行为验证；benchmark 可复现；限制公开；你能不看 AI 解释实现。缺少其中一项，就准确标注完成范围。

## 19. 资料阅读顺序

这些链接在制定路线时查阅过。网页主线会变化，实际开发以 Day 1 固定的版本为准。

| 何时读 | 资料 | 只读哪些内容 |
|---|---|---|
| Day 1 | [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) | README、example、Sequence、scheduler、block manager、runner |
| Day 2 | [Qwen3.5-2B 模型卡](https://huggingface.co/Qwen/Qwen3.5-2B) | 官方加载方式、模型输入与依赖 |
| Day 2 | [2B config](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json) | 层类型、维度、dtype、RoPE 配置 |
| Day 2–5 | [Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5) | 模型与 Processor API，跟随链接查看锁定版本源码 |
| Day 2–5 | [PR #232](https://github.com/GeeeekExplorer/nano-vllm/pull/232) | 模型加载、多模态和 recurrent slot 相关 diff |
| Day 3 | [qLLM](https://github.com/RLS-ResearchLab/qLLM) | StateManager、生命周期测试与明确限制；跳过 MoE/量化扩展 |
| Day 8 | [PyTorch CUDA Graph](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs) | warmup、capture、固定缓冲、replay 与约束 |

**你现在的第一步：打开自己的 nano-vLLM 仓库，同时读 `sequence.py` 与 `scheduler.py`，追踪“请求进来、分配资源、执行一轮、更新进度、结束释放”。用一个真实三请求 trace 验证理解。Token、embedding 和基础模型 forward 不再复习。**
