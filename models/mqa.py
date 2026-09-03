import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RoPE
from kernels.kernel import (
    Rope,
    FlashAttn,
    rope_cache,
)


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

        if self.backend == "cuda":

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
                cos.float().contiguous(),
                persistent=False,
            )

            self.register_buffer(
                "sin_cache",
                sin.float().contiguous(),
                persistent=False,
            )

            self.rope = None

        else:

            self.rope = RoPE(
                self.rotary_dim,
                config.max_seq_len,
            )

            self.cos_cache = None
            self.sin_cache = None

    def _apply_rope(
        self,
        q,
        k,
        position_offset,
    ):

        if self.backend == "cuda":

            q = Rope.apply(
                q.contiguous(),
                None,
                self.cos_cache,
                self.sin_cache,
                self.rotary_dim,
                position_offset,
            )

            k = Rope.apply(
                k.contiguous(),
                None,
                self.cos_cache,
                self.sin_cache,
                self.rotary_dim,
                position_offset,
            )

            return q, k

        else:

            q, k = self.rope(
                q,
                k,
                position_offset=position_offset,
            )

            return q, k

    def _attention(
        self,
        q,
        k,
        v,
        causal,
    ):

        if self.backend == "cuda":

            return FlashAttn.apply(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                causal,
            )

        else:

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=causal,
                dropout_p=(
                    self.dropout
                    if self.training
                    else 0.0
                ),
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

        q, k = self._apply_rope(
            q,
            k,
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

        out = self._attention(
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
            cos.float().contiguous(),
            persistent=False,
        )

        self.register_buffer(
            "sin_cache",
            sin.float().contiguous(),
            persistent=False,
        )

    def _apply_rope(
        self,
        q,
        k,
        position_offset,
    ):

        if self.backend == "cuda":

            q = Rope.apply(
                q.contiguous(),
                None,
                self.cos_cache,
                self.sin_cache,
                self.rotary_dim,
                position_offset,
            )

            k = Rope.apply(
                k.contiguous(),
                None,
                self.cos_cache,
                self.sin_cache,
                self.rotary_dim,
                position_offset,
            )

            return q, k

        S_q = q.shape[-2]
        S_k = k.shape[-2]

        half = self.rotary_dim // 2

        cos_q = self.cos_cache[
            position_offset:position_offset + S_q,
            :half,
        ]

        sin_q = self.sin_cache[
            position_offset:position_offset + S_q,
            :half,
        ]

        cos_q = torch.cat(
            [cos_q, cos_q],
            dim=-1,
        )

        sin_q = torch.cat(
            [sin_q, sin_q],
            dim=-1,
        )

        q_rot = q[..., :self.rotary_dim]
        q_pass = q[..., self.rotary_dim:]

        q1, q2 = q_rot.chunk(
            2,
            dim=-1,
        )

        q_rotated = torch.cat(
            [-q2, q1],
            dim=-1,
        )

        q_rot = (
            q_rot * cos_q
            + q_rotated * sin_q
        )

        q = torch.cat(
            [q_rot, q_pass],
            dim=-1,
        )

        cos_k = self.cos_cache[
            position_offset:position_offset + S_k,
            :half,
        ]

        sin_k = self.sin_cache[
            position_offset:position_offset + S_k,
            :half,
        ]

        cos_k = torch.cat(
            [cos_k, cos_k],
            dim=-1,
        )

        sin_k = torch.cat(
            [sin_k, sin_k],
            dim=-1,
        )

        k_rot = k[..., :self.rotary_dim]
        k_pass = k[..., self.rotary_dim:]

        k1, k2 = k_rot.chunk(
            2,
            dim=-1,
        )

        k_rotated = torch.cat(
            [-k2, k1],
            dim=-1,
        )

        k_rot = (
            k_rot * cos_k
            + k_rotated * sin_k
        )

        k = torch.cat(
            [k_rot, k_pass],
            dim=-1,
        )

        return q, k

    def _attention(
        self,
        q,
        k,
        v,
        causal,
    ):

        if self.backend == "cuda":

            return FlashAttn.apply(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                causal,
            )

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=causal,
            dropout_p=(
                self.dropout
                if self.training
                else 0.0
            ),
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
        ).transpose(
            1,
            2,
        ).contiguous()

        k = k.view(
            B,
            SK_new,
            self.num_kv_heads,
            self.headdim,
        ).transpose(
            1,
            2,
        ).contiguous()

        v = v.view(
            B,
            SK_new,
            self.num_kv_heads,
            self.headdim,
        ).transpose(
            1,
            2,
        ).contiguous()

        position_offset = 0

        if kv_cache is not None:
            position_offset = kv_cache.length

        q, k = self._apply_rope(
            q,
            k,
            position_offset,
        )

        if kv_cache is not None:

            k, v = kv_cache.update(
                layer_idx,
                k,
                v,
            )

        SK = k.shape[2]

        needs_mask = causal and (SQ == SK)

        if self.backend == "pytorch":
            out = F.scaled_dot_product_attention(
                q, k, v,                     
                is_causal=needs_mask,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
        else:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)
            out = self._attention(q, k, v, needs_mask)

        out = self._attention(
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