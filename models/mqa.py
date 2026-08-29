import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RoPE


class MQA(nn.Module):
    """
    Multi-Query Attention.

    Q has num_heads.

    K/V have num_kv_heads.

    For true MQA:

        num_kv_heads = 1

    For GQA:

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

        assert (
            self.num_heads % self.num_kv_heads == 0
        )

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

        self.rope = RoPE(
            self.headdim
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
        ).transpose(1, 2)

        k = k.view(
            B,
            SK,
            self.num_kv_heads,
            self.headdim,
        ).transpose(1, 2)

        v = v.view(
            B,
            SK,
            self.num_kv_heads,
            self.headdim,
        ).transpose(1, 2)


        q, k = self.rope(
            q,
            k,
            position_offset=position_offset,
        )

        k = k.repeat_interleave(
            self.num_groups,
            dim=1,
        )

        v = v.repeat_interleave(
            self.num_groups,
            dim=1,
        )

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.headdim)

        if causal:

            query_positions = torch.arange(
                position_offset,
                position_offset + SQ,
                device=q.device,
            )

            key_positions = torch.arange(
                SK,
                device=q.device,
            )

            mask = (
                key_positions.unsqueeze(0)
                >
                query_positions.unsqueeze(1)
            )

            scores = scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

        attn = F.softmax(
            scores.float(),
            dim=-1,
        ).to(q.dtype)

        if self.training and self.dropout > 0:

            attn = F.dropout(
                attn,
                p=self.dropout,
                training=True,
            )


        out = attn @ v

        # [B, H, S, D]
        #
        # ->
        #
        # [B, S, d_model]

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
            return out, attn

        return out


class MQA_Cached(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads

        self.bias = config.bias
        self.dropout = config.dropout

        assert (
            self.d_model % self.num_heads == 0
        )

        self.headdim = (
            self.d_model // self.num_heads
        )

        assert (
            self.num_heads % self.num_kv_heads == 0
        )

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

        self.rope = RoPE(
            self.headdim
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
        ).transpose(1, 2)

        k = k.view(
            B,
            SK_new,
            self.num_kv_heads,
            self.headdim,
        ).transpose(1, 2)

        v = v.view(
            B,
            SK_new,
            self.num_kv_heads,
            self.headdim,
        ).transpose(1, 2)


        position_offset = 0

        if kv_cache is not None:
            position_offset = kv_cache.length

        q, k = self.rope(
            q,
            k,
            position_offset=position_offset,
        )

        if kv_cache is not None:

            k, v = kv_cache.update(
                layer_idx,
                k,
                v,
            )

        k = k.repeat_interleave(
            self.num_groups,
            dim=1,
        )

        v = v.repeat_interleave(
            self.num_groups,
            dim=1,
        )

        SK = k.shape[2]

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.headdim)

        if causal:

            query_positions = torch.arange(
                position_offset,
                position_offset + SQ,
                device=q.device,
            )

            key_positions = torch.arange(
                SK,
                device=q.device,
            )

            mask = (
                key_positions.unsqueeze(0)
                >
                query_positions.unsqueeze(1)
            )

            scores = scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )

        attn = F.softmax(
            scores.float(),
            dim=-1,
        ).to(q.dtype)

        if self.training and self.dropout > 0:

            attn = F.dropout(
                attn,
                p=self.dropout,
                training=True,
            )

        out = attn @ v

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
            return out, attn

        return out