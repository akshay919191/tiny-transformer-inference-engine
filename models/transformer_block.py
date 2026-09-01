import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import argparse

from .attention import MHA, MHA_CACHED
from .mqa import MQA, MQA_Cached
from .embedding import TokenEmbedding
from .mlp import SwiGLU
from .model_config import ModelConfig
from .rmsnorm import RMSNorm
import sys
from pathlib import Path

from kernels.kernel import (
    TopK,
    Rope,
    FlashAttn,
    RMSNormFunction,
)
topk = TopK()

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kv_cache import KVCache_kv


def _build_attention(config, attn_type, backend, cached):
    """
    attn_type: "mha" or "mqa"
    backend:   "cuda" or "pytorch" (forwarded to the chosen attention module)
    cached:    True -> return the KV-cache-aware variant
    """

    assert attn_type in ("mha", "mqa"), \
        f"attn_type must be 'mha' or 'mqa', got {attn_type!r}"

    if attn_type == "mqa":
        cls = MQA_Cached if cached else MQA
        return cls(config, backend=backend)

    cls = MHA_CACHED if cached else MHA

    return cls(
        numhead=config.num_heads,
        dmodel=config.d_model,
        dropout=config.dropout,
        bias=config.bias,
        backend=backend,
        max_seq_len=config.max_seq_len,
    )


class TransformerBlock_Nocache(nn.Module):

    def __init__(self, config, attn_type="mqa", backend="cuda"):
        super().__init__()

        self.attn_norm = RMSNorm(config.d_model)

        self.attn = _build_attention(
            config,
            attn_type=attn_type,
            backend=backend,
            cached=False,
        )

        self.mlp_norm = RMSNorm(config.d_model)

        self.mlp = SwiGLU(
            config.d_model,
            config.hidden_size,
            config.bias,
        )

    def forward(self, x):

        residual = x

        x = self.attn_norm(x)

        x = self.attn(
            x,
            x,
            x,
            causal=True,
        )

        x = x + residual

        residual = x

        x = self.mlp_norm(x)

        x = self.mlp(x)

        x = x + residual

        return x


class Transformer_nocache(nn.Module):

    def __init__(self, config, attn_type="mqa", backend="cuda"):
        super().__init__()

        self.embedding = TokenEmbedding(
            config.vocab_size,
            config.d_model,
        )

        self.layers = nn.ModuleList([
            TransformerBlock_Nocache(
                config,
                attn_type=attn_type,
                backend=backend,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        for layer in self.layers:

            x = layer(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


class TransformerBlock(nn.Module):

    def __init__(self, config, layer_idx, attn_type="mqa", backend="cuda"):
        super().__init__()

        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(config.d_model)

        self.attn = _build_attention(
            config,
            attn_type=attn_type,
            backend=backend,
            cached=True,
        )

        self.mlp_norm = RMSNorm(config.d_model)

        self.mlp = SwiGLU(
            config.d_model,
            config.hidden_size,
            config.bias,
        )

    def forward(self, x, kv_cache):

        residual = x

        x = self.attn_norm(x)

        x = self.attn(
            x,
            x,
            x,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
            causal=True,
        )

        x = x + residual

        residual = x

        x = self.mlp_norm(x)

        x = self.mlp(x)

        x = x + residual

        return x


class Transformer(nn.Module):

    def __init__(self, config, attn_type="mqa", backend="cuda"):
        super().__init__()

        self.embedding = TokenEmbedding(
            config.vocab_size,
            config.d_model,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                config,
                layer_idx=i,
                attn_type=attn_type,
                backend=backend,
            )
            for i in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def forward(self, input_ids, kv_cache=None):

        x = self.embedding(input_ids)

        for layer in self.layers:

            x = layer(
                x,
                kv_cache=kv_cache,
            )

        if kv_cache is not None:
            kv_cache.advance(input_ids.shape[1])

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


def make_kv_cache(model, config, batch_size, max_seq_len, device):

    head_dim = config.d_model // config.num_heads

    return KVCache(
        num_layers=config.num_layers,
        batch_size=batch_size,
        num_heads=config.num_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        dtype=next(model.parameters()).dtype,
        device=device,
    )

def make_kv_cache_(model, config, batch_size, max_seq_len, device):

    head_dim = config.d_model // config.num_heads

    return KVCache_kv(
        num_layers=config.num_layers,
        batch_size=batch_size,
        num_heads=config.num_kv_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        dtype=next(model.parameters()).dtype,
        device=device,
    )

