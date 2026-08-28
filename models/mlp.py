import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        bias: bool = False,
    ):
        super().__init__()

        self.gate_proj = nn.Linear(
            d_model,
            hidden_dim,
            bias=bias,
        )

        self.up_proj = nn.Linear(
            d_model,
            hidden_dim,
            bias=bias,
        )

        self.down_proj = nn.Linear(
            hidden_dim,
            d_model,
            bias=bias,
        )

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        x = F.silu(gate) * up

        x = self.down_proj(x)

        return x