import torch

class KVPool:
    def __init__(self, num_layers, block_size, num_blocks, num_kv_heads, head_dim, dtype, device):
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self.kv = torch.zeros(
            (num_layers, 2, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.device = self.kv.device  
        self._shape_check()

        self.flat_k = [self.kv[l, 0].view(-1, num_kv_heads, head_dim) for l in range(num_layers)]
        self.flat_v = [self.kv[l, 1].view(-1, num_kv_heads, head_dim) for l in range(num_layers)]

    @classmethod
    def from_config(cls, cfg, device):
        return cls(
            num_layers=cfg.num_layers,
            block_size=cfg.block_size,
            num_blocks=cfg.num_gpu_blocks,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            dtype=cfg.kv_dtype,
            device=device,
        )

    def _shape_check(self):
        expected = (self.num_layers, 2, self.num_blocks, self.block_size,
                    self.num_kv_heads, self.head_dim)
        assert self.kv.shape == expected, f"shape mismatch: {self.kv.shape} vs {expected}"

    def make_slot_mapping(self, start: int, n: int, block_table: list[int]) -> torch.Tensor:
        assert start + n <= len(block_table) * self.block_size, \
            f"block table too short: need {start+n} slots, table gives {len(block_table)*self.block_size}"

        block_table_t = torch.tensor(block_table, dtype=torch.long, device=self.device)
        p = torch.arange(start, start + n, device=self.device)

        logical_block = p // self.block_size
        offset = p % self.block_size
        physical = block_table_t[logical_block]
        slot = physical * self.block_size + offset

        assert torch.all(slot >= self.block_size), \
            f"slot mapping touches block 0, which is reserved: {slot[slot < self.block_size]}"

        return slot

    def write(self, layer: int, slot_mapping: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert slot_mapping.dtype == torch.long
        assert slot_mapping.device == self.device, f"{slot_mapping.device} vs {self.device}"

        assert k.shape == v.shape, f"k/v shape mismatch: {k.shape} vs {v.shape}"
        assert k.shape[0] == slot_mapping.numel(), \
            f"k has {k.shape[0]} rows but slot_mapping has {slot_mapping.numel()} slots"
        assert k.shape[1:] == (self.num_kv_heads, self.head_dim), \
            f"k trailing dims {k.shape[1:]} != expected {(self.num_kv_heads, self.head_dim)}"
        assert k.dtype == self.dtype, f"k dtype {k.dtype} != pool dtype {self.dtype}"
        assert v.dtype == self.dtype, f"v dtype {v.dtype} != pool dtype {self.dtype}"

        self.flat_k[layer][slot_mapping] = k
        self.flat_v[layer][slot_mapping] = v

    def gather(self, layer: int, block_table: list[int], seq_len: int):
        assert seq_len <= len(block_table) * self.block_size, \
            f"seq_len={seq_len} exceeds capacity of block_table ({len(block_table)*self.block_size})"

        bt = torch.tensor(block_table, dtype=torch.long, device=self.device)
        k = self.kv[layer, 0][bt]
        v = self.kv[layer, 1][bt]

        k = k.reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]
        v = v.reshape(-1, self.num_kv_heads, self.head_dim)[:seq_len]
        return k, v