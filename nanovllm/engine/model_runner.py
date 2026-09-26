import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # Question: what is init_process_group?
        # Answer: initialize Pytorch's distributed communication environment acorss multiple GPUs
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        # Qwen3ForCausalLM only loads the structure of the model
        load_model(self.model, config.model)
        # load_model() loads the weights of the model
        self.sampler = Sampler()
        # Sampler transform logits to next token id
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    # Summary: runs a synthetic worst-case forward pass using dummy sequences with max_num_batched_tokens before serving real requests.
    def warmup_model(self):
        torch.cuda.empty_cache()
        # Question: what is empty_cache?
        # Answer: free freememo in Pytorch Caching Allocator
        torch.cuda.reset_peak_memory_stats()
        # Question: what is reset_peak_memory_stats?
        # Answer: reset Pytorch allocated_bytes.all.peak to the current allocated level
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    # Summary:
    # 1. computes block_bytes, memo used for kv cache, num_kvcache_blocks
    # 2. apply for a continuous GPU memo to store kv cache
    # [2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim]
    # 3. allocate blocks to every layer
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        # total is the total memory of the GPU, used is the memory that is currently in use
        # Used includes the memory of CUDA Context, Driver communication, NCCL buffers, Pytorch buffers, alive tensor...
        # current is the exact memo bytes actively occupied by torch.Tensor objects at this moment
        # peak = dynamic memo used during inference + current
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        # peak is the peak used memory of the GPU in Prefill warmup.
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # Question: what does the current memo mean excatly?
        # Answer: the exact memo bytes actively occupied by torch.Tensor objects at this moment

        # Question: what is the difference between current and used? why substract both used and (peak-current)?
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # Question: hf_config.head_dim is computed here? why?
        # Answer: hf_config may contain head_dim, if not, it is computed here
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        # 2: k and v
        # num_hidden_layers: number of layers in the model
        # self.block_size: num of tokens in a block
        # num_kv_heads: number of heads in the model
        # head_dim: one head dimension
        # hf_config.dtype.itemsize: size of one element in the model

        # Question: what is the structure of hidden layes in Qwen3ForCausalLM?
        # Answer: one self_attn, one mlp
        # a block should contain num_hidden_layers * block_size kv cache.
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        # (peak-current) is the maximum dynamic memo used during inference
        # used is the memory that is currently in use
        # total * config.gpu_memory_utilization is the total memory of the GPU
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # apply a continuous GPU memo to store kv cache

        # [num_kv_heads, head_dim] is one element of kv cache
        # 2 is k and v
        # hf_config.num_hidden_layers is the number of layers in the model
        # config.num_kvcache_blocks is the number of blocks in the kv cache
        # self.block_size is the number of tokens in a block

        # This code is to band model.k_cache to self.kv_cache. erver layer.

        # Question: what is the structure of module in self.model.modules()? Is k_cache a member in module? Every model has k_cache or v_cache in huggingface? or just a new class for vllm?
        # Answer: Standard HuggingFace Transformers model does not have k_cache and v_cache, this attribute is implemented by vllm/ nano-vllm. In Attention: self.k_cache = self.v_cache = torch.tensor([])
        layer_id = 0
        for module in self.model.modules():
            # Question: what does self.model.modules() mean?
            # Answer: return a iterator of all modules in the model
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                # Question: is this copy or reference?
                # Answer: reference
                # In python or pytorch 
                # Simple Assignment(a = b) is always a reference
                # Pytorch distinguishes between Basic Indexing (view/reference) and Advanced Indexing(Copy)
                # Basic Slicing t[0], t[1:5]... is reference
                # Advanced Slicing t[[0, 2, 4]], t[torch.tensor([0, 1])]... is copy
                # Boolean Masking t[t>0] is Copy
                # Explicit Copy t.clone() t.detach().clone() is copy
                # use torch.shares_memory(a, b) to check if a and b share the same memory
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # Summary: pad the block_tables to the max length in seqs with -1
    # Return: a 2-dimension Tensor [batch_size, max_num_blocks]
    def prepare_block_tables(self, seqs: list[Sequence]):
        # In a batch, len(seq) in seqs are not all the same.
        # GPU CUDA/Triton only accepts a 2 dimension Tensor [batch_size, max_num_blocks]
        # So we need to pad the block_tables to the max length in seqs
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        # use -1 to pad the block_tables
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    # Summary: To avoid the overhead of padding with Variable-length Packing and PagedAttention
    # Return:
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        # cumulative length. if there are 2 sequences (1. 5 tokens, 2. 3 tokens), then cu_seqlens_q = [0, 5, 8]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # Summary: capture CUDA graphs for each batch size and save the pointer to CUDA Graphs' buffers to self.graph_vars = dict()
    # this cuda graph is used during decoding.
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        # id of decoding token
        positions = torch.zeros(max_bs, dtype=torch.int64)
        # where is the pos of decoding token
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        # where to store the kv cache
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        # len of context of each seq
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        # block_tables[i, j] is j-th block of i-th sequence
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # static output buffer

        # These are CUDA Graphs' static input/output buffers.
        # CUDA Graphs only remember the input_ids pointer = ... and positions pointer = ... and block_tables pointer = ...
        # we need to modify the content of these buffers to compute
        # graph_input_ids[:bs].copy_(new_input_ids)
        # graph.replay()


        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        # capture batch sizes
        self.graphs = {}
        # where to store CUDA Graphs
        self.graph_pool = None

        # capture CUDA graphs for each batch size
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # new a CUDA Graph obj
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # attention layer can use there information by context = get_context()
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
