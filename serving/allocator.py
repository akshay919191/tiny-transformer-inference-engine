class BlockAllocator:
    """Hands out physical KV block ids. Block 0 is reserved as scratch."""

    def __init__(self, num_blocks: int):
        assert num_blocks >= 2, "need block 0 plus at least one usable block"
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, 0, -1))   
        self._is_free = [False] + [True] * (num_blocks - 1)

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_usable(self) -> int:
        return self.num_blocks - 1

    def can_allocate(self, n: int) -> bool:
        return n <= len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise RuntimeError("out of KV blocks")
        b = self._free.pop()
        self._is_free[b] = False
        return b

    def allocate_n(self, n: int) -> list[int]:
        if not self.can_allocate(n):
            raise RuntimeError(f"need {n} blocks, only {self.num_free} free")
        return [self.allocate() for _ in range(n)]

    def free(self, blocks: list[int]) -> None:
        seen = set()
        for b in blocks:                                   # validate everything first
            if not (0 < b < self.num_blocks):
                raise ValueError(f"invalid block id {b}")
            if self._is_free[b] or b in seen:
                raise ValueError(f"double free of block {b}")
            seen.add(b)
        for b in blocks:
            self._is_free[b] = True
            self._free.append(b)