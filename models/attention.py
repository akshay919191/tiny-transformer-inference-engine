import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.rope import RoPE
from kv_cache import KVCache
from kernels.kernel import (
    Rope,
    FlashAttn,
    rope_cache,
    rope_cuda,
    flashattn,
)


class MHA(nn.Module):
    def __init__(
        self,
        numhead: int,
        dmodel: int,
        dropout: float = 0.0,
        bias: bool = False,
        backend: str = "cuda",
        max_seq_len: int = 4096,
    ):
        super().__init__()

        assert dmodel % numhead == 0

        self.numhead = numhead
        self.headdim = dmodel // numhead
        self.dropout = dropout if dropout is not None else 0.0
        self.backend = backend

        assert self.backend in ("cuda", "pytorch"), \
            f"backend must be 'cuda' or 'pytorch', got {self.backend!r}"

        self.query = nn.Linear(dmodel, dmodel, bias=bias)
        self.key = nn.Linear(dmodel, dmodel, bias=bias)
        self.value = nn.Linear(dmodel, dmodel, bias=bias)
        self.rotary_dim = self.headdim

        if self.backend == "cuda":
            reference = torch.empty(1, dtype=torch.float16, device="cuda")
            cos, sin = rope_cache(reference, max_seq_len, self.rotary_dim)
            self.register_buffer("cos_cache", cos.float().contiguous(), persistent=False)
            self.register_buffer("sin_cache", sin.float().contiguous(), persistent=False)
            self.rope = None
        else:
            self.rope = RoPE(self.headdim)
            self.cos_cache = None
            self.sin_cache = None

        self.out = nn.Linear(dmodel, dmodel, bias=bias)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self.cos_cache is not None:
            self.cos_cache = self.cos_cache.float()
        if self.sin_cache is not None:
            self.sin_cache = self.sin_cache.float()
        return self

    def _apply_rope(self, q, k, position_offset=0):
        if self.backend == "cuda":
            if self.training:
                q = Rope.apply(q, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)
                k = Rope.apply(k, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)
                return q, k

            orig_dtype = q.dtype
            q_c = q if q.is_contiguous() else q.contiguous()
            k_c = k if k.is_contiguous() else k.contiguous()
            q_half = q_c.half() if q_c.dtype != torch.float16 else q_c
            k_half = k_c.half() if k_c.dtype != torch.float16 else k_c

            q = rope_cuda.forward(q_half, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)
            k = rope_cuda.forward(k_half, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)

            if q.dtype != orig_dtype:
                q, k = q.to(orig_dtype), k.to(orig_dtype)
            return q, k

        return self.rope(q, k, position_offset=position_offset)

    def _attention(self, q, k, v, causal, return_attn):
        if self.backend == "cuda":
            SQ = q.shape[2]
            SK = k.shape[2]
            if causal and (SQ == SK):
                needs_mask = True
            else:
                needs_mask = False

            if self.training:
                out = FlashAttn.apply(q, k, v, needs_mask)
            else:
                orig_dtype = q.dtype
                q_c = q if q.is_contiguous() else q.contiguous()
                k_c = k if k.is_contiguous() else k.contiguous()
                v_c = v if v.is_contiguous() else v.contiguous()

                q_half = q_c.half() if q_c.dtype != torch.float16 else q_c
                k_half = k_c.half() if k_c.dtype != torch.float16 else k_c
                v_half = v_c.half() if v_c.dtype != torch.float16 else v_c

                out, _ = flashattn.flash_fwd(q_half, k_half, v_half, needs_mask)
                if out.dtype != orig_dtype:
                    out = out.to(orig_dtype)

            return out, None

        SQ = q.shape[2]
        SK = k.shape[2]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.headdim)

        if causal:
            mask = torch.triu(
                torch.ones(SQ, SK, dtype=torch.bool, device=scores.device),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)

        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=True)

        out = torch.matmul(attn, v)
        return out, attn

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
        return_attn: bool = False,
    ):
        B, SQ, D = query.shape
        _, SK, _ = key.shape

        q = self.query(query).view(B, SQ, self.numhead, self.headdim).transpose(1, 2)
        k = self.key(key).view(B, SK, self.numhead, self.headdim).transpose(1, 2)
        v = self.value(value).view(B, SK, self.numhead, self.headdim).transpose(1, 2)

        q, k = self._apply_rope(q, k)
        out, attn = self._attention(q, k, v, causal, return_attn)

        out = out.transpose(1, 2)
        if not out.is_contiguous():
            out = out.contiguous()
        out = out.view(B, SQ, D)

        return self.out(out)


class MHA_CACHED(nn.Module):
    def __init__(
        self,
        numhead: int,
        dmodel: int,
        dropout: float = 0.0,
        bias: bool = False,
        backend: str = "cuda",
        max_seq_len: int = 4096,
    ):
        super().__init__()

        assert dmodel % numhead == 0

        self.numhead = numhead
        self.headdim = dmodel // numhead
        self.dropout = dropout if dropout is not None else 0.0
        self.backend = backend

        assert self.backend in ("cuda", "pytorch"), \
            f"backend must be 'cuda' or 'pytorch', got {self.backend!r}"

        self.query = nn.Linear(dmodel, dmodel, bias=bias)
        self.key = nn.Linear(dmodel, dmodel, bias=bias)
        self.value = nn.Linear(dmodel, dmodel, bias=bias)
        self.rotary_dim = self.headdim

        if self.backend == "cuda":
            reference = torch.empty(1, dtype=torch.float16, device="cuda")
            cos, sin = rope_cache(reference, max_seq_len, self.rotary_dim)
            self.register_buffer("cos_cache", cos.float().contiguous(), persistent=False)
            self.register_buffer("sin_cache", sin.float().contiguous(), persistent=False)
            self.rope = None
        else:
            self.rope = RoPE(self.headdim)
            self.cos_cache = None
            self.sin_cache = None

        self.out = nn.Linear(dmodel, dmodel, bias=bias)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self.cos_cache is not None:
            self.cos_cache = self.cos_cache.float()
        if self.sin_cache is not None:
            self.sin_cache = self.sin_cache.float()
        return self

    def _apply_rope(self, q, k, position_offset=0):
        # No-cache path only: position_offset is a Python int here.
        if self.backend == "cuda":
            if self.training:
                q = Rope.apply(q, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)
                k = Rope.apply(k, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)
                return q, k

            orig_dtype = q.dtype
            q_c = q if q.is_contiguous() else q.contiguous()
            k_c = k if k.is_contiguous() else k.contiguous()
            q_half = q_c.half() if q_c.dtype != torch.float16 else q_c
            k_half = k_c.half() if k_c.dtype != torch.float16 else k_c

            q = rope_cuda.forward(q_half, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)
            k = rope_cuda.forward(k_half, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)

            if q.dtype != orig_dtype:
                q, k = q.to(orig_dtype), k.to(orig_dtype)
            return q, k

        return self.rope(q, k, position_offset=position_offset)

    def _rope_cached(self, x, kv_cache):
        pos = kv_cache.positions(x.shape[2])
        cos = self.cos_cache.index_select(0, pos)
        sin = self.sin_cache.index_select(0, pos)
        return Rope.apply(x, None, cos, sin, self.rotary_dim, 0)

    def _attention(self, q, k, v, causal, SQ, SK, return_attn):
        if self.backend == "cuda":
            if causal and (SQ == SK):
                needs_mask = True
            else:
                needs_mask = False

            if self.training:
                out = FlashAttn.apply(q, k, v, needs_mask)
            else:
                orig_dtype = q.dtype
                q_c = q if q.is_contiguous() else q.contiguous()
                k_c = k if k.is_contiguous() else k.contiguous()
                v_c = v if v.is_contiguous() else v.contiguous()

                q_half = q_c.half() if q_c.dtype != torch.float16 else q_c
                k_half = k_c.half() if k_c.dtype != torch.float16 else k_c
                v_half = v_c.half() if v_c.dtype != torch.float16 else v_c

                out, _ = flashattn.flash_fwd(q_half, k_half, v_half, needs_mask)
                if out.dtype != orig_dtype:
                    out = out.to(orig_dtype)

            return out, None

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.headdim)

        if causal:
            query_positions = torch.arange(SK - SQ, SK, device=scores.device)
            key_positions = torch.arange(SK, device=scores.device)
            mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)

        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=True)

        out = attn @ v
        return out, attn

    def forward(
        self,
        query,
        key,
        value,
        kv_cache=None,
        layer_idx=None,
        causal=False,
        return_attn=False,
        attn_mask=None,
    ):
        B, SQ, D = query.shape

        q = self.query(query).view(B, SQ, self.numhead, self.headdim).transpose(1, 2)
        k = self.key(key).view(B, key.shape[1], self.numhead, self.headdim).transpose(1, 2)
        v = self.value(value).view(B, value.shape[1], self.numhead, self.headdim).transpose(1, 2)

        if kv_cache is None:
            q, k = self._apply_rope(q, k, position_offset=0)
            out, attn = self._attention(q, k, v, causal, SQ, k.shape[2], return_attn)
        else:
            if self.backend == "cuda":
                q = self._rope_cached(q, kv_cache)
                k = self._rope_cached(k, kv_cache)
            else:
                q, k = self.rope(q, k, position_offset=kv_cache.length)

            k, v = kv_cache.update(layer_idx, k, v)  # FULL [B, H, max_seq_len, D]

            if attn_mask is None:
                attn_mask = kv_cache.attn_mask(SQ, causal=causal)

            q = q.to(k.dtype)  # cache dtype; no-op if they already match
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            ).to(query.dtype)

        out = out.transpose(1, 2)
        if not out.is_contiguous():
            out = out.contiguous()
        out = out.view(B, SQ, D)

        return self.out(out)