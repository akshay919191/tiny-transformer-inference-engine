from contextlib import nullcontext

import torch.nn as nn
from torch.profiler import record_function

from kernels.fused_swiglu import fused_swiglu

PROFILE_REGIONS = False  


def _r(name):
    return record_function(name) if PROFILE_REGIONS else nullcontext()


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x):
        with _r("mlp_gate_proj"):
            gate = self.gate_proj(x)
        with _r("mlp_up_proj"):
            up = self.up_proj(x)
        with _r("mlp_silu_mul"):
            h = fused_swiglu(gate, up)     
        with _r("mlp_down_proj"):
            return self.down_proj(h)