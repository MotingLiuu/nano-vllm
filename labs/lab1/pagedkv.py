from collections import deque

class Block:

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.data: list[int] = []


class Sequence:

    def __init__(self, token_ids: list[int]):
        self.num_token_allocated: int = 0
        self.token_ids: list[int] = token_ids
        self.block_table: list[int] = []


class BlockManager:

    def __init__(self):
        self.block_size: int = 4
        self.num_block: int = 5
        self.blocks: list[Block] = [Block(i) for i in range(self.num_block)]
        self.free_blocks: deque[int] = deque(range(self.num_block))
        self.used_blocks: set[int] = set()


    def can_allocate(self) -> int:
        if len(self.free_blocks) > 0:
            return self.free_blocks.popleft()
        else:
            return -1


    def allocate(self, seq: Sequence) -> int:
        num_allocate: int = (len(seq.token_ids) + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_allocate:
            return -1
        for i in range(num_allocate):
            block_id: int = self.can_allocate()
            self.blocks[block_id].data.clear()
            self.used_blocks.add(block_id)
            seq.block_table.append(block_id)
            for j in range(self.block_size):
                self.blocks[block_id].data.append(seq.token_ids[i * self.block_size + j])
                seq.num_token_allocated += 1
                if seq.num_token_allocated >= len(seq.token_ids):
                    break
        return 0


    def append(self, seq: Sequence, token_id: int) -> int:
        if seq.num_token_allocated >= len(seq.block_table) * self.block_size:
            block_id: int = self.can_allocate()
            if block_id == -1:
                return -1
            seq.token_ids.append(token_id)
            self.blocks[block_id].data.clear()
            self.used_blocks.add(block_id)
            seq.block_table.append(block_id)
            self.blocks[block_id].data.append(token_id)
        else:
            seq.token_ids.append(token_id)
            block_id: int = seq.block_table[-1]
            self.blocks[block_id].data.append(token_id)
        seq.num_token_allocated += 1
        return 0

    def deallocate(self, seq: Sequence) -> int:

        for block_id in seq.block_table:
            self.free_blocks.append(block_id)
            self.used_blocks.remove(block_id)

        seq.block_table.clear()
        seq.num_token_allocated = 0

        return 0


def print_detail(seqs: list[Sequence], block_manager: BlockManager):
    for index, seq in enumerate(seqs):
        print(f"seq{index+1}.token_ids: {seq.token_ids}\n")
        print(f"seq{index+1}.block_table: {seq.block_table}\n")
        for block_id in seq.block_table:
            print(f"block{block_id}.data: {block_manager.blocks[block_id].data}\n")

    print(f"block_manager.used_blocks: {block_manager.used_blocks}\n")
    print(f"block_manager.free_blocks: {block_manager.free_blocks}\n")


if __name__ == "__main__":
    block_manager: BlockManager = BlockManager()
    seq1: Sequence = Sequence([0, 1, 2, 3, 4])
    block_manager.allocate(seq1)
    print_detail([seq1], block_manager)

    seq2: Sequence = Sequence([5, 6, 7])
    block_manager.allocate(seq2)
    print_detail([seq1, seq2], block_manager)

    block_manager.append(seq1, 5)
    block_manager.append(seq1, 6)
    block_manager.append(seq1, 7)
    block_manager.append(seq1, 8)
    print_detail([seq1, seq2], block_manager)

    block_manager.deallocate(seq2)
    print_detail([seq1], block_manager)

    seq3 = Sequence([9, 10, 11, 12, 13])
    block_manager.allocate(seq3)
    print_detail([seq1, seq3], block_manager)









