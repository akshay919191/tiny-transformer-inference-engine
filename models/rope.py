import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()

        assert dim % 2 == 0, "RoPE dimension must be even"

        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, dim, 2).float() / dim
            )
        )

        positions = torch.arange(max_seq_len).float()

        freqs = torch.outer(positions, inv_freq)

        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())

    def forward(self, q, k, position_offset=0):

        B, H, T, D = q.shape


        positions = torch.arange(
            position_offset,
            position_offset + T,
            device=q.device,
        )

        cos = self.cos[positions].to(q.dtype)
        sin = self.sin[positions].to(q.dtype)

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_even = q[..., 0::2]
        q_odd = q[..., 1::2]

        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]

        q_rot_even = q_even * cos - q_odd * sin
        q_rot_odd = q_even * sin + q_odd * cos

        k_rot_even = k_even * cos - k_odd * sin
        k_rot_odd = k_even * sin + k_odd * cos

        q_rot = torch.stack(
            (q_rot_even, q_rot_odd),
            dim=-1
        ).flatten(-2)

        k_rot = torch.stack(
            (k_rot_even, k_rot_odd),
            dim=-1
        ).flatten(-2)

        return q_rot, k_rot