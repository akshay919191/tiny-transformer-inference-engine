# models/mqa_paged.py
"""
MQA_Paged -- multi-query attention backed by the fused paged-attention
CUDA kernels.

The two forward paths (`forward_paged` and `forward_paged_batched`) now
call `paged_varlen_attention` from `paged_varlen_attn.py`, which is a thin
autograd wrapper around the compiled `pagedattn` extension.

Requires:
    pagedattn            (the compiled .so from setup.py)
    paged_varlen_attn.py (the autograd wrapper)
"""

import torch
import torch.nn.functional as F

from models.mqa import MQA_Cached
from .fake_paged_attn import paged_varlen_attention


def _get_kv_pool(pool, layer_idx):
    """
    Return the raw [num_blocks, blocksize, Hkv, D] K/V tensors the kernel reads.
    Your KVPool stores them at pool.kv[layer_idx, 0] and pool.kv[layer_idx, 1].
    """
    if hasattr(pool, "kv"):
        return pool.kv[layer_idx, 0], pool.kv[layer_idx, 1]

    # Fallback for other pool layouts (kept for portability).
    for k_name, v_name in (
        ("k_buffer",    "v_buffer"),
        ("k_buffers",   "v_buffers"),
        ("k_pool",      "v_pool"),
        ("k_cache",     "v_cache"),
    ):
        if hasattr(pool, k_name) and hasattr(pool, v_name):
            return getattr(pool, k_name)[layer_idx], getattr(pool, v_name)[layer_idx]

    raise AttributeError(
        f"KVPool layout not recognised: {type(pool).__name__} "
        f"has attrs {[a for a in dir(pool) if not a.startswith('_')]}"
    )


class MQA_Paged(MQA_Cached):

    def forward_paged(self, x, pool, layer_idx, block_table, start):
        B, SQ, _ = x.shape
        assert B == 1, "forward_paged is single-sequence only"

        # Route through the batched path with a synthetic batch of 1.
        positions       = list(range(start, start + SQ))
        query_start_loc = [0, SQ]
        block_tables    = [list(block_table)]

        out = self.forward_paged_batched(
            x.reshape(SQ, x.shape[-1]),
            pool, layer_idx,
            positions, query_start_loc, block_tables,
        )
        return out.view(B, SQ, x.shape[-1])

    def forward_paged_batched(
        self,
        x,                      # [T, d_model]  flattened across all sequences
        pool,
        layer_idx,
        positions,              # list[int]     abs pos of each query token
        query_start_loc,        # list[int]     cumulative offsets, len N+1
        block_tables,           # list[list[int]] physical block IDs per seq
    ):
        T, d_model = x.shape
        device = x.device

        q = self.q_proj(x).view(T, self.num_heads,    self.headdim)   # [T, Hq,  D]
        k = self.k_proj(x).view(T, self.num_kv_heads, self.headdim)   # [T, Hkv, D]
        v = self.v_proj(x).view(T, self.num_kv_heads, self.headdim)   # [T, Hkv, D]

        positions_tensor = torch.as_tensor(positions, device=device, dtype=torch.long)
        q4 = q.transpose(0, 1).unsqueeze(0)      # [1, Hq,  T, D]
        k4 = k.transpose(0, 1).unsqueeze(0)      # [1, Hkv, T, D]
        q4, k4 = self.rope.forward_at(q4, k4, positions=positions_tensor)
        q = q4.squeeze(0).transpose(0, 1).contiguous()    # [T, Hq,  D]
        k = k4.squeeze(0).transpose(0, 1).contiguous()    # [T, Hkv, D]

        slot_mappings = []
        for i in range(len(block_tables)):
            s0, s1 = query_start_loc[i], query_start_loc[i + 1]
            sq_i = s1 - s0
            if sq_i == 0:
                continue
            seq_start_pos = int(positions[s0])
            slot_mappings.append(
                pool.make_slot_mapping(seq_start_pos, sq_i, block_tables[i])
            )

        if slot_mappings:
            pool.write(layer_idx, torch.cat(slot_mappings), k, v)

        k_pool, v_pool = _get_kv_pool(pool, layer_idx)

        out = paged_varlen_attention(
            q, k_pool, v_pool,
            block_tables, query_start_loc, positions,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale=None,                 # -> 1 / sqrt(head_dim)
            non_causal=False,
        )                               # -> [T, Hq, D]

        out = out.contiguous().view(T, d_model)
        return self.out(out)