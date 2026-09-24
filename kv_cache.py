import torch
import torch.nn as nn


class KVCache_kv(nn.Module):


    def __init__(
        self,
        num_layers,
        batch_size,
        num_heads,  # number of KV heads (same arg name as before)
        max_seq_len,
        head_dim,
        dtype=torch.float16,
        device="cuda",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self._py_len = 0

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
        if not torch.compiler.is_compiling() and self._py_len + k.shape[2] > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: {self._py_len} + {k.shape[2]} > {self.max_seq_len}"
            )

        idx = self.positions(k.shape[2])
        self.k[layer_idx].index_copy_(2, idx, k.to(self.k.dtype))
        self.v[layer_idx].index_copy_(2, idx, v.to(self.v.dtype))

        # Full fixed-size buffers. Same shape every step.
        return self.k[layer_idx], self.v[layer_idx]

    def attn_mask(self, q_len, causal=True):
        """
        Bool mask [1, 1, q_len, max_seq_len]; True = may attend.
        Call BEFORE advance(). causal=True: query i sees keys 0..pos+i.
        causal=False: every query sees keys 0..pos+q_len-1 (hides only the unused tail).
        """
        if causal:
            limit = self.positions(q_len)[:, None]              # inclusive
            m = self._slots[None, :] <= limit
        else:
            m = (self._slots[None, :] < (self.pos + q_len)).expand(q_len, -1)
        return m[None, None]

    def advance(self, num_tokens=1):
        if not torch.compiler.is_compiling():
            if self._py_len + num_tokens > self.max_seq_len:
                raise ValueError(
                    f"KV cache overflow: {self._py_len} + {num_tokens} > {self.max_seq_len}"
                )
            self._py_len += num_tokens
        self.pos.add_(num_tokens)  # in-place on a static buffer

    def reset(self):
        self._py_len = 0
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