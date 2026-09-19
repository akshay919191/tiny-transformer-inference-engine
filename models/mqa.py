import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.kernel import (
    rope_cache,
    Rope,
    FlashAttn,
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
            q = Rope.apply(q, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)
            k = Rope.apply(k, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)
            return q, k

        return self.rope(q, k, position_offset=position_offset)

    def forward(self, query, key, value, causal=False, return_attn=False, position_offset=0):
        B, SQ, _ = query.shape
        SK = key.shape[1]

        q = self.q_proj(query).view(B, SQ, self.num_heads, self.headdim).transpose(1, 2)
        k = self.k_proj(key).view(B, SK, self.num_kv_heads, self.headdim).transpose(1, 2)
        v = self.v_proj(value).view(B, SK, self.num_kv_heads, self.headdim).transpose(1, 2)

        q, k = self._apply_rope(q, k, position_offset)
        if causal and (SQ == SK):
            needs_mask = True
        else:
            needs_mask = False

        if self.backend == "cuda":
            out = FlashAttn.apply(q, k, v, needs_mask)
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
        # No-cache path only: position_offset is a Python int here.
        if self.backend == "cuda":
            q = Rope.apply(q, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)
            k = Rope.apply(k, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)
            return q, k

        return self.rope(q, k, position_offset=position_offset)

    def _rope_cached(self, x, kv_cache):
        """
        RoPE at the cache's tensor position, with no Python-int offset.

        The custom op takes `position_offset: int`, which would change every
        step and force a recompile + new CUDA graph. Instead we gather the
        cos/sin rows for the current positions on-device and hand the op that
        small slice with offset 0. Same rows, same math, no int in the graph.
        (Assumes cos_cache/sin_cache are indexed by position on dim 0.)
        """
        pos = kv_cache.positions(x.shape[2])
        cos = self.cos_cache.index_select(0, pos)
        sin = self.sin_cache.index_select(0, pos)
        return Rope.apply(x, None, cos, sin, self.rotary_dim, 0)

    def _sdpa_masked(self, q, k, v, mask):

        B, H, SQ, D = q.shape
        Hkv = k.shape[1]
        G = H // Hkv

        if mask is not None and mask.dim() == 2:
            mask = mask[None, None]

        if G > 1:
            q = q.reshape(B, Hkv, G * SQ, D)
            if mask is not None and mask.size(-2) != 1:
                mask = mask.repeat(1, 1, G, 1)  # tile the SQ mask rows G times

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return out.reshape(B, H, SQ, D)

    def forward(
        self,
        query,
        key,
        value,
        kv_cache=None,
        layer_idx=None,
        causal=True,
        return_attn=False,
        attn_mask=None,
    ):
        B, SQ, _ = query.shape
        SK_new = key.shape[1]

        q = self.q_proj(query).view(B, SQ, self.num_heads, self.headdim).transpose(1, 2)
        k = self.k_proj(key).view(B, SK_new, self.num_kv_heads, self.headdim).transpose(1, 2)
        v = self.v_proj(value).view(B, SK_new, self.num_kv_heads, self.headdim).transpose(1, 2)

        if kv_cache is None:
            q, k = self._apply_rope(q, k, 0)

            SK = k.shape[2]
            if causal and (SQ == SK):
                needs_mask = True
            else:
                needs_mask = False

            if self.backend == "cuda":
                out = FlashAttn.apply(q, k, v, needs_mask)
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    is_causal=needs_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    enable_gqa=(self.num_kv_heads != self.num_heads),
                )
        else:
            capturing = torch.cuda.is_current_stream_capturing()
            start = None if capturing else int(kv_cache.length)

            if self.backend == "cuda":
                q = self._rope_cached(q, kv_cache)
                k = self._rope_cached(k, kv_cache)
            else:
                q, k = self.rope.forward_at(q, k, kv_cache.positions(q.shape[2]))

            k_full, v_full = kv_cache.update(layer_idx, k, v)  # FULL [B, Hkv, max_seq_len, D]
            q = q.to(k_full.dtype)

            if attn_mask is None and not capturing:
                L = start + SQ
                is_prefill = causal and SQ > 1 and start == 0
                assert SQ == 1 or start == 0, (
                    "chunked prefill (SQ>1 with a non-empty cache) needs a "
                    "bottom-right-aligned causal mask; pass attn_mask explicitly"
                )

                k_v = k_full[:, :, :L]
                v_v = v_full[:, :, :L]

                if self.backend == "cuda":
                    out = FlashAttn.apply(q, k_v, v_v, is_prefill)
                else:
                    out = F.scaled_dot_product_attention(
                        q, k_v, v_v,
                        is_causal=is_prefill,
                        dropout_p=self.dropout if self.training else 0.0,
                        enable_gqa=(self.num_kv_heads != self.num_heads),
                    )
            else:
                if attn_mask is None:
                    attn_mask = kv_cache.attn_mask(SQ, causal=causal)
                out = self._sdpa_masked(q, k_full, v_full, attn_mask)

            out = out.to(query.dtype)

        out = out.transpose(1, 2).contiguous().view(B, SQ, self.d_model)
        out = self.out(out)

        if return_attn:
            return out, None
        return out