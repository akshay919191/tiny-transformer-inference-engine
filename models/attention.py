import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RoPE


class MHA(nn.Module):
    def __init__(
        self,
        numhead: int,
        dmodel: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        assert dmodel % numhead == 0

        self.numhead = numhead
        self.headdim = dmodel // numhead
        self.dropout = dropout if dropout is not None else 0.0

        self.query = nn.Linear(dmodel, dmodel, bias=bias)
        self.key = nn.Linear(dmodel, dmodel, bias=bias)
        self.value = nn.Linear(dmodel, dmodel, bias=bias)

        self.rope = RoPE(self.headdim)

        self.out = nn.Linear(dmodel, dmodel, bias=bias)

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


        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        q = q.view(
            B, SQ, self.numhead, self.headdim
        ).transpose(1, 2)

        k = k.view(
            B, SK, self.numhead, self.headdim
        ).transpose(1, 2)

        v = v.view(
            B, SK, self.numhead, self.headdim
        ).transpose(1, 2)

        q, k = self.rope(q, k)


        scores = torch.matmul(
            q,
            k.transpose(-1, -2),
        ) / math.sqrt(self.headdim)

        # scores: [B,H,SQ,SK]


        if causal:
            mask = torch.triu(
                torch.ones(
                    SQ,
                    SK,
                    dtype=torch.bool,
                    device=scores.device,
                ),
                diagonal=1,
            )

            scores = scores.masked_fill(
                mask,
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
        out = torch.matmul(attn, v)

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, SQ, D)
        )

        out = self.out(out)

        if return_attn:
            return out, attn

        return out


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RoPE


class MHA_CACHED(nn.Module):
    def __init__(
        self,
        numhead: int,
        dmodel: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        assert dmodel % numhead == 0

        self.numhead = numhead
        self.headdim = dmodel // numhead
        self.dropout = dropout if dropout is not None else 0.0

        self.query = nn.Linear(
            dmodel,
            dmodel,
            bias=bias,
        )

        self.key = nn.Linear(
            dmodel,
            dmodel,
            bias=bias,
        )

        self.value = nn.Linear(
            dmodel,
            dmodel,
            bias=bias,
        )

        self.rope = RoPE(self.headdim)

        self.out = nn.Linear(
            dmodel,
            dmodel,
            bias=bias,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache=None,
        causal: bool = False,
        return_attn: bool = False,
    ):

        B, SQ, D = query.shape


        q = self.query(query)
        k = self.key(key)
        v = self.value(value)


        q = q.view(
            B,
            SQ,
            self.numhead,
            self.headdim,
        ).transpose(1, 2)

        k = k.view(
            B,
            key.shape[1],
            self.numhead,
            self.headdim,
        ).transpose(1, 2)

        v = v.view(
            B,
            value.shape[1],
            self.numhead,
            self.headdim,
        ).transpose(1, 2)

        cache_len = 0

        if kv_cache is not None:
            cache_len = kv_cache[0].shape[2]

        q, k = self.rope(
            q,
            k,
            position_offset=cache_len,
        )

        if kv_cache is not None:

            k_cache, v_cache = kv_cache

            k = torch.cat(
                [k_cache, k],
                dim=2,
            )

            v = torch.cat(
                [v_cache, v],
                dim=2,
            )

        new_kv_cache = (k, v)

        SK = k.shape[2]

        scores = (
            q @ k.transpose(-1, -2)
        ) / math.sqrt(self.headdim)


        if causal:

            query_positions = torch.arange(
                SK - SQ,
                SK,
                device=scores.device,
            )

            key_positions = torch.arange(
                SK,
                device=scores.device,
            )

            mask = (
                key_positions.unsqueeze(0)
                > query_positions.unsqueeze(1)
            )

            scores = scores.masked_fill(
                mask,
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
            .view(B, SQ, D)
        )

        out = self.out(out)

        if return_attn:

            return out, attn, new_kv_cache

        return out, new_kv_cache