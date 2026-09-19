import torch
import torch.nn as nn
from kernels.kernel import RMSNormFunction


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return RMSNormFunction.apply(x, self.weight, self.eps)