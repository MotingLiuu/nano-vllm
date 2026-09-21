#### 关联补充：你在 sequence.py 和 scheduler.py 中的疑问

  • token_ids vs self.token_ids（sequence.py:30）：
  外部传入的是用户 prompt 的静态 list；Sequence 内部用 copy(token_ids)
  制作私有拷贝，在自回归 decode 时，每生成一个新词就会通过 append_token() 动态修改
  self.token_ids。
  • preempt 在干嘛？（scheduler.py:83）：
  抢占（Preemption）。自回归生成中，每个活跃请求每步都在增长。当总显存被耗尽（not
  can_append）时，系统不能 OOM 崩溃，调度器会选出牺牲者（Victim），将其全部物理块
  deallocate 掉，状态置回 WAITING，把显存让给其它请求先完成。
  ──────
  ### 第二部分：nano-vLLM 架构进阶思考题

  结合你的 10 天冲刺路线（理解调度器、Paged KV、变长 Prefill
  与生命周期），请思考以下 5
  个架构设计问题。试着回答它们，会让你对整个系统的脉络有极高维度的掌握：

  #### 思考题 1：Prefill 与 Decode 在显存行为上的根本区别

  │ 在 CS336 中，我们知道 Prefill 是一次性处理所有 prompt tokens，而 Decode
  │ 是每次只步进 1 个 token。
  │ 请从 BlockManager 的角度回答：为什么 Prefill 阶段调用的是 can_allocate
  │ 一次性索取多个块，而 Decode 阶段只需要调用 can_append 并且条件是 len(seq) %
  │ block_size == 1？

  Answer:
  During prefill, we need to allocate blocks for all the tokens in the prompt.
  But during decode, we only need to allocate a block for one token only when current block is full.

  #### 思考题 2：为什么 Sequence 结束释放时，必须 reversed(seq.block_table)？

  │ 查看 block_manager.py:110 的代码：
  │
  │   for block_id in reversed(seq.block_table):
  │
  │ 为什么释放块的时候要倒序（reversed）释放？如果改成正序释放，会对前缀缓存的命中率
  │ （LRU 淘汰策略）产生什么微妙的影响？

  Answer: 
  This is because `allocate` would alway return the leftmost free block.
  we should reverse the order of the block_table to make sure the most important block is in the last of free list.

  #### 思考题 3：尾块隔离与写时并发安全

  │ 在 can_allocate 中，遍历范围是 range(seq.num_blocks - 1)，明确排除了最后一个
  │ block。
  │ 假设 Prompt 长度是 300，block_size 是 256，它有两个块：Block 0（256 个
  │ token，满）和 Block 1（44 个 token，未满）。
  │ 为什么绝对不能把 Block 1
  算作缓存命中并共享给其他请求？如果共享了会发生什么灾难？

  Answer: Because, If we allow the last block can be shared. There is a situation that `num_cached_blocks == seq.num_blocks` , in `allocate` block1 would be append to `seq.block_table` and put new kv cache in it. At the same time there my be another seq also append `blcok1` into block_table. Crash.

  #### 思考题 4：调度器中的“三类 Token 计数”

  │ 在 sequence.py 中，定义了三个属性：
  │
  │ 1. seq.num_tokens
  │ 2. seq.num_cached_tokens
  │ 3. seq.num_scheduled_tokens
  │ 请描述在一个支持 Chunked Prefill（分块 Prefill） 的长 Prompt
  │ 请求中，这三个数值在一次调度迭代（Schedule → Run →
  │ Postprocess）前后分别是如何变化的？

  Answer: 
  1. num_tokens: it is the total number of tokens in the prompt. no change.
  2. num_cached_tokens: it is the number of tokens that have been cached, it would be seq.num_scheduled_token(max_num_batched_tokens) + num_cached_tokens
  3. num_scheduled_tokens: it is the number of tokens that have been scheduled(max_num_batched_tokens) 

  #### 思考题 5：为什么架构上不能用 batch 下标直接存状态？（连接 Day 3 的核心思考）

  │ 在很多简单的 PyTorch demo 中，我们习惯用一个形状为 (batch_size, ...)
  │ 的张量保存状态，比如第 i 个请求的状态就存在第 i 行。
  │ 为什么在 nano-vLLM 这种支持 Continuous
  │ Batching（连续批处理）的真实引擎中，绝对不能用当前 Batch
  │ 中的位置索引（batch_index）来绑定请求的状态？


Answer: I have not looked into engine. But I think this is because length of prompt is not fixed.
  ──────
  你可以尝试挑 1~2 个你最感兴趣的问题聊聊你的直觉或答案，我们可以顺着这些问题把整个
  engine 的运行链条彻底打通！





