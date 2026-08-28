import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import RoPE
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kv_cache import KVCache

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
        query,
        key,
        value,
        kv_cache=None,
        layer_idx=None,
        causal=False,
        return_attn=False,
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

        # Position of the new token(s)
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
            return out, attn

        return out