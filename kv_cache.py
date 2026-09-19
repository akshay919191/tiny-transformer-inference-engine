import torch
import torch.nn as nn


class KVCache_kv(nn.Module):
    """
    Fixed-shape KV cache for torch.compile(mode="reduce-overhead").

    - K/V buffers are allocated once; update() ALWAYS returns the full
      [B, H, max_seq_len, D] tensors (never a [:end] slice).
    - The write position is a GPU tensor (`pos`), not a Python int, so Dynamo
      compiles once and the same CUDA graph is replayed every decode step.
    - Unused slots are hidden with attn_mask(), not by shrinking the tensor.

    """

    def __init__(
        self,
        num_layers,
        batch_size,
        num_heads,  
        max_seq_len,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        shape = (num_layers, batch_size, num_heads, max_seq_len, head_dim)

        self.register_buffer("k", torch.zeros(shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer("v", torch.zeros(shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer("pos", torch.zeros((), dtype=torch.long, device=device), persistent=False)
        self.register_buffer("_slots", torch.arange(max_seq_len, device=device), persistent=False)

        for t in (self.k, self.v, self.pos):
            torch._dynamo.mark_static_address(t)

    @property
    def dtype(self):
        return self.k.dtype

    @property
    def length(self):
        return int(self.pos.item())

    def positions(self, num_tokens):
        """Absolute positions [num_tokens] (int64, on device) for the next write."""
        return self.pos + self._slots[:num_tokens]

    def update(self, layer_idx, k, v):
        if not (0 <= layer_idx < self.num_layers):
            raise ValueError(f"layer_idx={layer_idx} out of range [0, {self.num_layers})")
        if k.shape[3] != self.k.shape[4]:
            raise ValueError("Head dim mismatch")
        if k.device != self.k.device:
            raise ValueError("Device mismatch")

        idx = self.positions(k.shape[2])
        self.k[layer_idx].index_copy_(2, idx, k.to(self.k.dtype))
        self.v[layer_idx].index_copy_(2, idx, v.to(self.v.dtype))

        return self.k[layer_idx], self.v[layer_idx]

    def attn_mask(self, q_len, causal=True):
 
        if causal:
            limit = self.positions(q_len)[:, None]              # inclusive
            m = self._slots[None, :] <= limit
        else:
            m = (self._slots[None, :] < (self.pos + q_len)).expand(q_len, -1)
        return m[None, None]

    def advance(self, num_tokens=1):
        self.pos.add_(num_tokens)  

    def reset(self):
        self.pos.zero_()  

class KVCache(KVCache_kv):

    def __init__(
        self,
        num_layers,
        batch_size,
        num_heads,
        max_seq_len,
        head_dim,
        dtype,
        device,
    ):
        super().__init__(
            num_layers,
            batch_size,
            num_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )