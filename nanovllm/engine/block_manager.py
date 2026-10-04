# Question: what does BlockManager really control? The sapce to store is allocated in model_runner by calling allocate_kc_cache().
# Each GPU would save some heads's kv cache in its allocated GPU memo.
#
#
# Answer: BlockManager only manage the block'id and block content, dont manage the real kv cache's memo. A Block is contains metadata of some kv cache's memo.
# 1. initialize a seq's block_table(1. cached blocks' id  2. new blocks id)
# 2. recycle the seq's sources during deallocating. (just the block_id and block's content)
# 3. compute the block's hash
#


# Answer: The main functions are
# 1.can_allocate(seq), allocate(seq): find cached blocks, allocate new blocks if needed.
# 2.deallocate(seq): -1 ref count, if ref count reaches 0, _deallocate the block.
# 3.can_append(seq), may_append(seq): check if there is a free block, and allocate a new block if there is one.
# 4.compute_hash(token_ids, int)
# 5.hash_block(seq)

# Question: I know there is 2 check
# 1. hash
# 2. current token ids
# But I don't think this can ensure the uniqueness of the block with context.
# Is there a situation that context is diff, current token ids is same, but hash is same?
#
# Answer: yes, this can not guarantee the uniqueness of the block with context. but 99.9% of the time, it will be unique.

# can_allocate only runs on thread0, dont need locks
# can_allocate and allocate are integrated. can_allocate first determines if there are enough source
# blocks to allocate, and then allocate them if there are enough free blocks. If dont split the source's availablity and source request, code would cause source leak.


# can_append and may_append are integrated. can_append first checks if there is one free block, and then may_append allocates a new block if there is one.

# A block_id in free_block_ids moved out of free_block_ids throught self.allocate(cache hit), self._allocate(store new content)



from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # num_blocks is num_kv_cache_blocks computed at model runner.
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        # Question: what is xxh64?
        # Answer: create an incremental 64-bit hasher from the xxhash lib. It doesn't return the final hash value, need to feed it bytes with .update(), then
        # get the result with .intdigest()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
            # to_bytes converts an integer to a byte string, little means least-significant byte first
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # Summary: popleft from self.free_block_ids, if this block is still in self.used_block_ids, remove it from self.used_block_ids.
    # A map from hash to block id is only removed by _allocate_block, lazy remove.
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    # Summary: move block_id from self.used_block_ids to self.free_block_ids.
    # do not touch self.hash_to_block_id and contents in self.blocks[block_id]
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # Summary: find the num of cached blocks for given sequence and determine if there are enough free blocks
    # can_allocate only runs on thread0, dont need locks
    # can_allocate and allocate are integrated. can_allocate first determines if there are enough source
    # blocks to allocate, and then allocate them if there are enough free blocks. If dont split the source's availablity and source request, code would cause source leak.
    def can_allocate(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks

        ####
        # find the cached blocks
        ####
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            # if the token_ids with context 's kv is not stored, break.
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1

        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    # Summary:
    # store block_id hitted into seq.block_table, _allocate_block() for tokens not in cache.
    #
    # A block_id in free_block_ids moved out of free_block_ids throught self.allocate(cache hit), self._allocate(store new content)
    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        # check if there is one free block.
        # this would only used during decoding
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # if the last block is full, allocate a new block
        #
        # can_append and may_append are integrated. can_append first checks if there is one free block, and then may_append allocates a new block if there is one.
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        # for block in [start, end) of seq.block_table
        # save the block id and corresponding hash value to hash_to_block_id
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
