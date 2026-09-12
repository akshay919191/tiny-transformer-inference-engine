import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.kernel import (
    rope_cache,
    Rope,
    FlashAttn,
    rope_cuda,
    flashattn,
)
from .rope import RoPE


class MQA(nn.Module):
    def __init__(self, config, backend="cuda"):
        super().__init__()

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.bias = config.bias
        self.dropout = config.dropout
        self.backend = backend

        assert self.backend in ("cuda", "pytorch")
        assert self.d_model % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0

        self.headdim = self.d_model // self.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads
        self.rotary_dim = self.headdim

        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.headdim, bias=self.bias)
        self.k_proj = nn.Linear(self.d_model, self.num_kv_heads * self.headdim, bias=self.bias)
        self.v_proj = nn.Linear(self.d_model, self.num_kv_heads * self.headdim, bias=self.bias)
        self.out = nn.Linear(self.d_model, self.d_model, bias=self.bias)

        if self.backend == "cuda":
            reference = torch.empty(1, dtype=torch.float16, device="cuda")
            cos, sin = rope_cache(reference, config.max_seq_len, self.rotary_dim)
            self.register_buffer("cos_cache", cos.float().contiguous(), persistent=False)
            self.register_buffer("sin_cache", sin.float().contiguous(), persistent=False)
            self.rope = None
        else:
            self.rope = RoPE(self.rotary_dim, config.max_seq_len)
            self.cos_cache = None
            self.sin_cache = None

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self.cos_cache is not None:
            self.cos_cache = self.cos_cache.float()
        if self.sin_cache is not None:
            self.sin_cache = self.sin_cache.float()
        return self

    def _apply_rope(self, q, k, position_offset):
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

    def forward(self, query, key, value, causal=False, return_attn=False, position_offset=0):
        B, SQ, _ = query.shape
        SK = key.shape[1]
        orig_dtype = query.dtype

        q = self.q_proj(query).view(B, SQ, self.num_heads, self.headdim).transpose(1, 2)
        k = self.k_proj(key).view(B, SK, self.num_kv_heads, self.headdim).transpose(1, 2)
        v = self.v_proj(value).view(B, SK, self.num_kv_heads, self.headdim).transpose(1, 2)

        q, k = self._apply_rope(q, k, position_offset)
        needs_mask = causal and (SQ == SK)

        if self.backend == "cuda":
            if self.training:
                out = FlashAttn.apply(q, k, v, needs_mask)
            else:
                q_c = q if q.is_contiguous() else q.contiguous()
                k_c = k if k.is_contiguous() else k.contiguous()
                v_c = v if v.is_contiguous() else v.contiguous()

                q_half = q_c.half() if q_c.dtype != torch.float16 else q_c
                k_half = k_c.half() if k_c.dtype != torch.float16 else k_c
                v_half = v_c.half() if v_c.dtype != torch.float16 else v_c

                out, _ = flashattn.flash_fwd(q_half, k_half, v_half, needs_mask)
                if out.dtype != orig_dtype:
                    out = out.to(orig_dtype)
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=needs_mask,
                dropout_p=self.dropout if self.training else 0.0,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )

        out = out.transpose(1, 2).contiguous().view(B, SQ, self.d_model)
        out = self.out(out)

        if return_attn:
            return out, None
        return out


class MQA_Cached(nn.Module):
    def __init__(self, config, backend="cuda"):
        super().__init__()

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.bias = config.bias
        self.dropout = config.dropout
        self.backend = backend

        assert self.backend in ("cuda", "pytorch")
        assert self.d_model % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0

        self.headdim = self.d_model // self.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads
        self.rotary_dim = self.headdim

        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.headdim, bias=self.bias)
        self.k_proj = nn.Linear(self.d_model, self.num_kv_heads * self.headdim, bias=self.bias)
        self.v_proj = nn.Linear(self.d_model, self.num_kv_heads * self.headdim, bias=self.bias)
        self.out = nn.Linear(self.d_model, self.d_model, bias=self.bias)

        if self.backend == "cuda":
            reference = torch.empty(1, dtype=torch.float16, device="cuda")
            cos, sin = rope_cache(reference, config.max_seq_len, self.rotary_dim)
            self.register_buffer("cos_cache", cos.float().contiguous(), persistent=False)
            self.register_buffer("sin_cache", sin.float().contiguous(), persistent=False)
            self.rope = None
        else:
            self.rope = RoPE(self.rotary_dim, config.max_seq_len)
            self.cos_cache = None
            self.sin_cache = None

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self.cos_cache is not None:
            self.cos_cache = self.cos_cache.float()
        if self.sin_cache is not None:
            self.sin_cache = self.sin_cache.float()
        return self

    def _apply_rope(self, q, k, position_offset):
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

    def forward(
        self,
        query,
        key,
        value,
        kv_cache=None,
        layer_idx=None,
        causal=True,
        return_attn=False,
    ):
        B, SQ, _ = query.shape
        SK_new = key.shape[1]
        orig_dtype = query.dtype

        q = self.q_proj(query).view(B, SQ, self.num_heads, self.headdim).transpose(1, 2)
        k = self.k_proj(key).view(B, SK_new, self.num_kv_heads, self.headdim).transpose(1, 2)
        v = self.v_proj(value).view(B, SK_new, self.num_kv_heads, self.headdim).transpose(1, 2)

        position_offset = kv_cache.length if kv_cache is not None else 0
        q, k = self._apply_rope(q, k, position_offset)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        SK = k.shape[2]
        needs_mask = causal and (SQ == SK)

        if self.backend == "cuda":
            if self.training:
                out = FlashAttn.apply(q, k, v, needs_mask)
            else:
                if q.is_contiguous() and k.is_contiguous() and v.is_contiguous():
                    q_half = q.half() if q.dtype != torch.float16 else q
                    k_half = k.half() if k.dtype != torch.float16 else k
                    v_half = v.half() if v.dtype != torch.float16 else v
                    out, _ = flashattn.flash_fwd(q_half, k_half, v_half, needs_mask)
                    if out.dtype != orig_dtype:
                        out = out.to(orig_dtype)
                else:
                    out = F.scaled_dot_product_attention(
                        q, k, v, is_causal=needs_mask, dropout_p=0.0,
                        enable_gqa=(self.num_kv_heads != self.num_heads)
                    )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=needs_mask,
                dropout_p=self.dropout if self.training else 0.0,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )

        out = out.transpose(1, 2).contiguous().view(B, SQ, self.d_model)
        out = self.out(out)

        if return_attn:
            return out, None
        return out