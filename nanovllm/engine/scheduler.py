# Summary:
# Scheduler contains a BlockManager, 1. main function is schedule, schedule a batch of sequences to be prefilled or decoded. 2. postprocess
#
# Seq's status: WAITTING(have not be prefilled), RUNNING(have been prefilled), FINISHED(have been decoded)
# WAITTING-schedule()->RUNNING-postprocess()->FINISHED
# RUNNING-schedule()-preempt()->WAITTING
#
# func: schedule, prefill or decode seqs, can execute chuncked prefill
#
# func: add(seq), add a request
#
# func: preempt, recycle resources by calling deallocate, and put seq back to waiting queue.
#
# func: is_finished, return True if all seqs are finished.
#
# func: postprocess, if prefill, just hash the blocks; if decode, hash the blocks and append token_id, if eos or max_tokens reached, set status to FINISHED, deallocate, and remove from running.

# kv_cache's size is (2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
#
#
# Prefill, a seq will be added to scheduled_seqs when it satisfies 1. can_allocate 2. num_tokens(will be scheduled) < remaining(except 1st seq)
# There is only one situtation that scheduled_seq returned is none: 1st seq's blocks can not be allocated
#
# Decode, pop the left most seq from self.running.
# If the seq can not can_append, preempt the right most seq from self.running. (This would put prefilled seq back to waiting to acquire its resources, just call deallocate
# and put seq back to left of self.waiting to ensure it would be the first one to be prefilled when memo avilable)
# If self.running is empty, break the loop and return scheduled_seqs.
#
# Every seq only executes allocate once.
# seq.status == SequenceStatus.RUNNING means, seq has finished prefill and can be decoded.

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        # the max num of sequences in this batch
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        # seqs waiting for prefill
        self.running: deque[Sequence] = deque()
        # seqs decoding

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    # Summary: This is the main function of scheduling. schedule a batch of sequences(prefill or decode), if not allocated, allocate(only prefill)
    #
    # If self.waiting, jsut prefilling.
    # If 1st seq can not be allocated, break, assert will not pass, panic. If 1st seq can be allocated, but remaining < num_tokens, chunked prefill
    # If 1st can be allocated, remaining >= num_tokens, full prefill, self.waiting.popleft, self.running.append, seq.status=RUNNING -> check next seq
    #
    # If not self.waiting, just decoding.
    #
    #
    def schedule(self) -> tuple[list[Sequence], bool]:
        # return tuple[list[]]
        scheduled_seqs = []
        num_batched_tokens = 0

        ####
        # Prefill, a seq will be added to scheduled_seqs when it satisfies 1. can_allocate 2. num_tokens(will be scheduled) < remaining(except 1st seq)
        # There is only one situtation that scheduled_seq returned is none: 1st seq's blocks can not be allocated
        #
        # Decode, pop the left most seq from self.running.
        # If the seq can not can_append, preempt the right most seq from self.running. (This would put prefilled seq back to waiting to acquire its resources, just call deallocate
        # and put seq back to left of self.waiting to ensure it would be the first one to be prefilled when memo avilable)
        # If self.running is empty, break the loop and return scheduled_seqs.
        #
        # Every seq only executes allocate once.
        # seq.status == SequenceStatus.RUNNING means, seq has finished prefill and can be decoded.
        ####
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
            # check whether there is token budget
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # If 1st seq cannot be allocated, break, assert will not pass
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)

            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # done seq prefill, WAITTING->RUNNING
                # pop from waiting, append to running
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True


        ####
        # Decode, pop the left most seq from self.running.
        # If the seq can not can_append, preempt the right most seq from self.running. (This would put prefilled seq back to waiting to acquire its resources, just call deallocate
        # and put seq back to left of self.waiting to ensure it would be the first one to be prefilled when memo avilable)
        # If self.running is empty, break the loop and return scheduled_seqs.
        ####
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                # If while loop end for true, execute else
                # add seq to scheduled
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        # reverse maintain the order of original self.running
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        # free the kv cache of seq, and move it to waiting
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        # add seq to left of waiting, to ensure it would be the first one to be prefilled when memo avilable

    # If is_prefill, just hash the blocks
    # If decode, hash the blocks and append token_id, if eos or max_tokens reached, set status to FINISHED, deallocate, and remove from running
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # set the status to FINISHED
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
