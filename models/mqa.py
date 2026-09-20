import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.profiler import record_function

from kernels.kernel import (
    rope_cache,
    Rope,
    FlashAttn,
)
from .rope import RoPE

PROFILE_REGIONS = False  # profile script turns this on only for the profiler run


def _r(name):
    return record_function(name) if PROFILE_REGIONS else nullcontext()


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
        if self.backend == "cuda":
            q = Rope.apply(q, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)
            k = Rope.apply(k, None, self.cos_cache, self.sin_cache, self.rotary_dim, position_offset)
            return q, k

        return self.rope(q, k, position_offset=position_offset)

    def _rope_qk_cached(self, q, k, kv_cache, layer_idx):
        """
        RoPE at the cache's tensor position, with no Python-int offset.

        The positions and the gathered cos/sin rows depend only on the decode
        step, not on the layer and not on q vs k. So build them once per forward
        (at layer 0) and reuse them for every later layer and for both q and k,
        instead of 2x positions + 4x index_select per layer. Still no Python
        int in the graph. (Assumes cos_cache/sin_cache are indexed by position
        on dim 0 and identical across layers, and that layers run in order.)
        """
        if not layer_idx or getattr(kv_cache, "_rope_cs", None) is None:
            with _r("rope_tables"):
                pos = kv_cache.positions(q.shape[2])
                kv_cache._rope_cs = (
                    self.cos_cache.index_select(0, pos),
                    self.sin_cache.index_select(0, pos),
                )
        cos, sin = kv_cache._rope_cs

        with _r("rope_q_apply"):
            q = Rope.apply(q, None, cos, sin, self.rotary_dim, 0)
        with _r("rope_k_apply"):
            k = Rope.apply(k, None, cos, sin, self.rotary_dim, 0)
        return q, k

    def _sdpa_masked(self, q, k, v, mask):
        """
        Masked attention over the FULL cache buffer, used only when a Python-int
        slice is impossible (CUDA graph capture) or the caller passed a mask.

        Flash can't take an arbitrary mask and the mem-efficient kernel doesn't
        do enable_gqa, so passing (mask + enable_gqa) drops to the math path
        (repeat_interleave on K/V). Instead fold the query-head groups into the
        query length: each KV head just sees G*SQ query rows. No K/V repeat.

        q: [B, H, SQ, D]; k, v: [B, Hkv, S, D]; mask: bool/float, broadcastable
        to [B, 1, SQ, S].
        """
        B, H, SQ, D = q.shape
        Hkv = k.shape[1]
        G = H // Hkv

        if mask is not None and mask.dim() == 2:
            mask = mask[None, None]

        if G > 1:
            # head h = kv*G + g  ->  row g*SQ + s within KV head `kv`
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

            # capture, where an int() sync would be illegal.
            start = None if capturing else int(kv_cache.length)

            if self.backend == "cuda":
                q, k = self._rope_qk_cached(q, k, kv_cache, layer_idx)
            else:
                with _r("rope_pytorch"):
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
                # ---- graph-safe / explicit-mask path: full buffer + mask ----
                if attn_mask is None:
                    attn_mask = kv_cache.attn_mask(SQ, causal=causal)
                out = self._sdpa_masked(q, k_full, v_full, attn_mask)

            out = out.to(query.dtype)

        out = out.transpose(1, 2).contiguous().view(B, SQ, self.d_model)
        out = self.out(out)

        if return_attn:
            return out, None
        return out