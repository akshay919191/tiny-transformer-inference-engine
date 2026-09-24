class Sequence:
    """One request's state: its tokens and the physical blocks it owns."""

    def __init__(self, seq_id: int, prompt_token_ids: list[int], block_size: int):
        self.seq_id = seq_id
        self.block_size = block_size
        self.token_ids = list(prompt_token_ids)
        self.block_table: list[int] = []

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    def num_blocks_needed(self) -> int:
        return blocks_for(self.num_tokens, self.block_size)

    def num_new_blocks_needed(self) -> int:
        return self.num_blocks_needed() - len(self.block_table)

    def ensure_blocks(self, allocator) -> None:
        n = self.num_new_blocks_needed()
        if n > 0:
            self.block_table.extend(allocator.allocate_n(n))

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)     # blocks are allocated later by ensure_blocks

    def release(self, allocator) -> None:
        allocator.free(self.block_table)
        self.block_table.clear()


def blocks_for(num_tokens: int, block_size: int) -> int:
    return -(-num_tokens // block_size)     # ceiling division