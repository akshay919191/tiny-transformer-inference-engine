import math
import torch
import torch.nn as nn
import torch.nn.functional as F




class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"

        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)

        self.register_buffer("cos", freqs.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin", freqs.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, position_offset: int = 0):
        # q, k: [B, H, T, D]
        T = q.shape[-2]
        
        cos = self.cos[:, :, position_offset : position_offset + T, :].to(dtype=q.dtype)
        sin = self.sin[:, :, position_offset : position_offset + T, :].to(dtype=q.dtype)

        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]

        q_out[..., 0::2] = q_even * cos - q_odd * sin
        q_out[..., 1::2] = q_even * sin + q_odd * cos

        k_out[..., 0::2] = k_even * cos - k_odd * sin
        k_out[..., 1::2] = k_even * sin + k_odd * cos

        return q_out, k_out