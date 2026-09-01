import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rope import RoPE
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kv_cache import KVCache

from kernels.kernel import (
    Rope,
    FlashAttn,
    rope_cache,
)


class MHA(nn.Module):
    """
    backend:
        "cuda"    -> use the custom CUDA kernels (Rope.apply / FlashAttn.apply)
        "pytorch" -> use the plain PyTorch path (RoPE module + manual
                     matmul/softmax attention), as originally written here
    """

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
            reference = torch.empty(
                1,
                dtype=torch.float16,
                device="cuda",
            )

            cos, sin = rope_cache(
                reference,
                max_seq_len,
                self.rotary_dim,
            )

            self.register_buffer("cos_cache", cos, persistent=False)
            self.register_buffer("sin_cache", sin, persistent=False)

            self.rope = None
        else:
            self.rope = RoPE(self.headdim)
            self.cos_cache = None
            self.sin_cache = None

        self.out = nn.Linear(dmodel, dmodel, bias=bias)

    def _apply_rope(self, q, k, position_offset=0):
        if self.backend == "cuda":
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

            return q, k
        else:
            return self.rope(q, k, position_offset=position_offset)

    def _attention(self, q, k, v, causal, return_attn):
        if self.backend == "cuda":
            SQ = q.shape[2]
            SK = k.shape[2]
            needs_mask = causal and (SQ == SK)

            out = FlashAttn.apply(q, k, v, needs_mask)

            attn = None if return_attn else None
            return out, attn
        else:
            SQ = q.shape[2]
            SK = k.shape[2]

            scores = torch.matmul(
                q,
                k.transpose(-1, -2),
            ) / math.sqrt(self.headdim)

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

        q, k = self._apply_rope(q, k)

        out, attn = self._attention(q, k, v, causal, return_attn)

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

        self.rotary_dim = self.headdim

        if self.backend == "cuda":
            reference = torch.empty(
                1,
                dtype=torch.float16,
                device="cuda",
            )

            cos, sin = rope_cache(
                reference,
                max_seq_len,
                self.rotary_dim,
            )

            self.register_buffer("cos_cache", cos, persistent=False)
            self.register_buffer("sin_cache", sin, persistent=False)

            self.rope = None
        else:
            self.rope = RoPE(self.headdim)
            self.cos_cache = None
            self.sin_cache = None

        self.out = nn.Linear(
            dmodel,
            dmodel,
            bias=bias,
        )

    def _apply_rope(self, q, k, position_offset=0):
        if self.backend == "cuda":
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

            return q, k
        else:
            return self.rope(q, k, position_offset=position_offset)

    def _attention(self, q, k, v, causal, SQ, SK, return_attn):
        if self.backend == "cuda":
            needs_mask = causal and (SQ == SK)

            out = FlashAttn.apply(q, k, v, needs_mask)

            return out, None
        else:
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

        q, k = self._apply_rope(
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

        out, attn = self._attention(q, k, v, causal, SQ, SK, return_attn)

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