import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RoPE
from kernels.kernel import (
    Rope,
    FlashAttn,
    RMSNormFunction,
    Softmax,       
    rope_cache 
)

class MQA(nn.Module):
    """
    Multi-Query / Grouped-Query Attention.

    Q has num_heads.

    K/V have num_kv_heads.

    True MQA:
        num_kv_heads = 1

    GQA:
        1 < num_kv_heads < num_heads
    """

    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads

        self.bias = config.bias
        self.dropout = config.dropout

        assert self.d_model % self.num_heads == 0

        self.headdim = (
            self.d_model // self.num_heads
        )

        assert self.num_heads % self.num_kv_heads == 0

        self.num_groups = (
            self.num_heads // self.num_kv_heads
        )


        self.q_proj = nn.Linear(
            self.d_model,
            self.num_heads * self.headdim,
            bias=self.bias,
        )

        self.k_proj = nn.Linear(
            self.d_model,
            self.num_kv_heads * self.headdim,
            bias=self.bias,
        )

        self.v_proj = nn.Linear(
            self.d_model,
            self.num_kv_heads * self.headdim,
            bias=self.bias,
        )

        self.out = nn.Linear(
            self.d_model,
            self.d_model,
            bias=self.bias,
        )


        self.rotary_dim = self.headdim

        reference = torch.empty(
            1,
            dtype=torch.float16,
            device="cuda",
        )

        cos, sin = rope_cache(
            reference,
            config.max_seq_len,
            self.rotary_dim,
        )

        self.register_buffer(
            "cos_cache",
            cos,
            persistent=False,
        )

        self.register_buffer(
            "sin_cache",
            sin,
            persistent=False,
        )

    def forward(
        self,
        query,
        key,
        value,
        causal=False,
        return_attn=False,
        position_offset=0,
    ):

        B, SQ, _ = query.shape
        SK = key.shape[1]

        q = self.q_proj(query)

        k = self.k_proj(key)

        v = self.v_proj(value)


        q = q.view(
            B,
            SQ,
            self.num_heads,
            self.headdim,
        ).transpose(1, 2).contiguous()

        k = k.view(
            B,
            SK,
            self.num_kv_heads,
            self.headdim,
        ).transpose(1, 2).contiguous()

        v = v.view(
            B,
            SK,
            self.num_kv_heads,
            self.headdim,
        ).transpose(1, 2).contiguous()


        q = Rope.apply(
            q,
            None,
            self.cos_cache,
            self.sin_cache,
            self.rotary_dim,
            position_offset,
        )

        k = Rope.apply(
            k,
            None,
            self.cos_cache,
            self.sin_cache,
            self.rotary_dim,
            position_offset,
        )

        k = k.repeat_interleave(
            self.num_groups,
            dim=1,
        )

        v = v.repeat_interleave(
            self.num_groups,
            dim=1,
        )

        needs_mask = causal and (SQ == SK)

        out = FlashAttn.apply(
            q,
            k,
            v,
            needs_mask,
        )

        out = (
            out
            .transpose(1, 2)
            .contiguous()
            .view(
                B,
                SQ,
                self.d_model,
            )
        )

        out = self.out(out)

        if return_attn:
            return out, None

        return out
class MQA_Cached(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads

        self.bias = config.bias
        self.dropout = config.dropout

        assert self.d_model % self.num_heads == 0

        self.headdim = self.d_model // self.num_heads

        assert self.num_heads % self.num_kv_heads == 0

        self.num_groups = self.num_heads // self.num_kv_heads

        self.rotary_dim = self.headdim

        self.q_proj = nn.Linear(
            self.d_model,
            self.num_heads * self.headdim,
            bias=self.bias,
        )

        self.k_proj = nn.Linear(
            self.d_model,
            self.num_kv_heads * self.headdim,
            bias=self.bias,
        )

        self.v_proj = nn.Linear(
            self.d_model,
            self.num_kv_heads * self.headdim,
            bias=self.bias,
        )

        self.out = nn.Linear(
            self.d_model,
            self.d_model,
            bias=self.bias,
        )

        reference = torch.empty(
            1,
            dtype=torch.float16,
            device="cuda",
        )

        cos, sin = rope_cache(
            reference,
            config.max_seq_len,
            self.rotary_dim,
        )

        self.register_buffer(
            "cos_cache",
            cos,
            persistent=False,
        )

        self.register_buffer(
            "sin_cache",
            sin,
            persistent=False,
        )

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
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(
            B,
            SQ,
            self.num_heads,
            self.headdim,
        ).transpose(1, 2).contiguous()

        k = k.view(
            B,
            SK_new,
            self.num_kv_heads,
            self.headdim,
        ).transpose(1, 2).contiguous()

        v = v.view(
            B,
            SK_new,
            self.num_kv_heads,
            self.headdim,
        ).transpose(1, 2).contiguous()


        position_offset = 0

        if kv_cache is not None:
            position_offset = kv_cache.length

        # Custom RoPE

        q = Rope.apply(
            q,
            None,
            self.cos_cache,
            self.sin_cache,
            self.rotary_dim,
            position_offset,
        )

        k = Rope.apply(
            k,
            None,
            self.cos_cache,
            self.sin_cache,
            self.rotary_dim,
            position_offset,
        )

        # KV CACHE

        if kv_cache is not None:

            k, v = kv_cache.update(
                layer_idx,
                k,
                v,
            )

        SK = k.shape[2]
        needs_mask = causal and (SQ == SK)

        # GQA

        k = k.repeat_interleave(
            self.num_groups,
            dim=1,
        )

        v = v.repeat_interleave(
            self.num_groups,
            dim=1,
        )

        # CUSTOM FLASH ATTENTION

        out = FlashAttn.apply(
            q,
            k,
            v,
            needs_mask,
        )

        # Merge heads
        out = (
            out
            .transpose(1, 2)
            .contiguous()
            .view(
                B,
                SQ,
                self.d_model,
            )
        )

        # Output projection

        out = self.out(out)

        if return_attn:
            return out, None

        return out