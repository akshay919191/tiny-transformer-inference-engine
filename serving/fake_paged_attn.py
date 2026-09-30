"""
paged_varlen_attn.py
====================

Thin autograd wrapper around the compiled `pagedattn` CUDA extension.

Public API is identical to the pure-PyTorch reference (see
paged_varlen_attn_ref.py) so you can swap between them with a one-line
import change.

    from paged_varlen_attn import paged_varlen_attention

The wrapper:

  * converts ragged `block_tables / query_start_loc / positions` into the
    packed tensors the kernel wants:  blocktable [N, max_blk], seq_lens [N],
    queryloc [N+1], max_q_len;
  * calls pagedattn.paged_attn_fwd in forward;
  * calls pagedattn.paged_attn_bwd in backward so gradients flow into
    q, k_pool, v_pool (including prefix-shared physical blocks).

Limitations (hard errors, not silent fallbacks):
  * head_dim must be a multiple of 8 (kernel requirement);
  * Hq must be a multiple of Hkv;
  * blocksize (k_pool.shape[1]) must be a power of two;
  * only self-attention (causal or bidirectional) is supported with
    gradients; cross-attention has no backward kernel.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F

import pagedattn


__all__ = ["paged_varlen_attention", "PagedVarlenAttention", "is_available"]


_KERNEL_DTYPE = torch.float16


def is_available() -> bool:
    """True if the compiled extension is importable and a CUDA device exists."""
    return torch.cuda.is_available() and hasattr(pagedattn, "paged_attn_fwd")


def _pack_metadata(
    block_tables: List[List[int]],
    query_start_loc: List[int],
    positions: List[int],
    T: int,
    device: torch.device,
    blocksize: int,
):
    num_seqs = len(block_tables)
    if len(query_start_loc) != num_seqs + 1:
        raise ValueError("query_start_loc must have length num_seqs + 1")
    if query_start_loc[0] != 0 or query_start_loc[-1] != T:
        raise ValueError("query_start_loc must start at 0 and end at T")
    if len(positions) != T:
        raise ValueError(f"positions must have length T ({T})")

    # -- blocktable [N, max_blk]: pad ragged rows with 0.  Unused slots are
    #    never read because the kernel guards on kv_pos < seq_len.
    max_blocks = max((len(bt) for bt in block_tables), default=0)
    blocktable = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=device)
    for i, bt in enumerate(block_tables):
        if len(bt) > 0:
            blocktable[i, :len(bt)] = torch.as_tensor(
                bt, dtype=torch.int32, device=device
            )

    # -- seq_lens [N] = kv_len per seq, derived from positions.
    #    kv_len is (last query's abs position) + 1.
    seq_lens = torch.zeros(num_seqs, dtype=torch.int32, device=device)
    for i in range(num_seqs):
        s0 = query_start_loc[i]
        s1 = query_start_loc[i + 1]
        if s1 > s0:
            seq_lens[i] = positions[s1 - 1] + 1
            # sanity: kv_len must fit in the number of physical blocks
            cap = len(block_tables[i]) * blocksize
            if seq_lens[i].item() > cap:
                raise ValueError(
                    f"seq {i}: kv_len {seq_lens[i].item()} exceeds "
                    f"{len(block_tables[i])} * blocksize ({cap})"
                )

    queryloc = torch.as_tensor(query_start_loc, dtype=torch.int32, device=device)

    max_q_len = max(
        (query_start_loc[i + 1] - query_start_loc[i] for i in range(num_seqs)),
        default=0,
    )
    return blocktable, seq_lens, queryloc, int(max_q_len)


class _PagedVarlenAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,              # [T, Hq, D]  fp16, contiguous
        k_pool,         # [nb, bs, Hkv, D]  fp16
        v_pool,         # [nb, bs, Hkv, D]  fp16
        blocktable,     # [N, max_blk] int32
        seq_lens,       # [N]          int32
        queryloc,       # [N+1]        int32
        max_q_len,      # python int
        scale,          # python float (<= 0 -> 1/sqrt(D))
        causal,         # python bool
    ):
        O, Lse = pagedattn.paged_attn_fwd(
            q, k_pool, v_pool,
            blocktable, seq_lens, queryloc,
            max_q_len, scale, causal, True,
        )
        ctx.save_for_backward(
            q, k_pool, v_pool, blocktable, seq_lens, queryloc, O, Lse
        )
        ctx.max_q_len = max_q_len
        ctx.scale = scale
        ctx.causal = causal
        return O

    @staticmethod
    def backward(ctx, dO):
        q, k_pool, v_pool, blocktable, seq_lens, queryloc, O, Lse = ctx.saved_tensors

        dO = dO.contiguous()
        if dO.dtype != _KERNEL_DTYPE:
            dO = dO.to(_KERNEL_DTYPE)

        dQ, dK, dV = pagedattn.paged_attn_bwd(
            q, k_pool, v_pool,
            blocktable, seq_lens, queryloc,
            dO, Lse, O,
            ctx.max_q_len, ctx.scale, ctx.causal,
        )

        # q, k_pool, v_pool, blocktable, seq_lens, queryloc,
        # max_q_len, scale, causal -> grads only for q/k/v
        return dQ, dK, dV, None, None, None, None, None, None


def paged_varlen_attention(
    q: torch.Tensor,                          # [T, Hq, D]
    k_pool: torch.Tensor,                     # [num_blocks, blocksize, Hkv, D]
    v_pool: torch.Tensor,                     # [num_blocks, blocksize, Hkv, D]
    block_tables: List[List[int]],
    query_start_loc: List[int],
    positions: List[int],
    num_heads: int,
    num_kv_heads: int,
    scale: Optional[float] = None,
    non_causal: bool = False,
) -> torch.Tensor:
    """
    Drop-in replacement for the reference implementation.  Uses the
    compiled paged-attention CUDA kernels under the hood.
    """
    if not is_available():
        raise RuntimeError(
            "pagedattn extension or CUDA is not available; "
            "build the extension and/or run on a CUDA device."
        )
    if q.dim() != 3:
        raise ValueError(f"q must be [T, Hq, D], got {tuple(q.shape)}")

    T, Hq, D = q.shape
    if Hq != num_heads:
        raise ValueError(f"q H={Hq} != num_heads={num_heads}")
    if k_pool.shape[2] != num_kv_heads:
        raise ValueError(
            f"pool Hkv={k_pool.shape[2]} != num_kv_heads={num_kv_heads}"
        )
    if Hq % num_kv_heads != 0:
        raise ValueError(f"Hq ({Hq}) must be a multiple of Hkv ({num_kv_heads})")
    if D % 8 != 0:
        raise ValueError(f"head_dim {D} must be a multiple of 8")

    blocksize = k_pool.shape[1]
    if blocksize & (blocksize - 1) != 0 or blocksize <= 0:
        raise ValueError(f"blocksize {blocksize} must be a power of two")

    if non_causal:
        # forward-only; the fused kernel has no cross-attn backward
        raise NotImplementedError(
            "non_causal gradient path is not implemented; "
            "use the pure-PyTorch reference for non-causal training."
        )

    if q.device.type != "cuda":
        raise RuntimeError("paged_varlen_attention requires a CUDA tensor for q")

    device = q.device

    # --- pack metadata ---
    blocktable, seq_lens, queryloc, max_q_len = _pack_metadata(
        block_tables, query_start_loc, positions, T, device, blocksize
    )

    # --- dtype cast (kernel is fp16) ---
    orig_dtype = q.dtype
    qh = q if q.dtype == _KERNEL_DTYPE else q.to(_KERNEL_DTYPE)
    kh = k_pool if k_pool.dtype == _KERNEL_DTYPE else k_pool.to(_KERNEL_DTYPE)
    vh = v_pool if v_pool.dtype == _KERNEL_DTYPE else v_pool.to(_KERNEL_DTYPE)

    qh = qh.contiguous()
    kh = kh.contiguous()
    vh = vh.contiguous()

    scale_val = float(scale) if scale is not None else -1.0   # <=0 -> 1/sqrt(D)

    # --- run ---
    O = _PagedVarlenAttnFn.apply(
        qh, kh, vh,
        blocktable, seq_lens, queryloc,
        max_q_len, scale_val, True,       # causal self-attention
    )

    if O.dtype != orig_dtype:
        O = O.to(orig_dtype)
    return O


class PagedVarlenAttention(torch.nn.Module):
    """Same as the reference module but backed by the fused CUDA kernels."""

    def __init__(self, num_heads: int, num_kv_heads: int,
                 scale: Optional[float] = None, non_causal: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.non_causal = non_causal

    def forward(
        self,
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        block_tables: List[List[int]],
        query_start_loc: List[int],
        positions: List[int],
    ) -> torch.Tensor:
        return paged_varlen_attention(
            q, k_pool, v_pool,
            block_tables, query_start_loc, positions,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
            non_causal=self.non_causal,
        )

