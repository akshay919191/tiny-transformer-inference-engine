import torch
import torch.nn as nn
from kernels.kernel import rmsnorm_cuda, RMSNormFunction


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return RMSNormFunction.apply(x, self.weight, self.eps)

        # INFERENCE / EVAL FAST PATH
        orig_shape = x.shape
        orig_dtype = x.dtype

        if x.dim() == 3:
            x_4d = x.unsqueeze(1)
        elif x.dim() == 2:
            x_4d = x.unsqueeze(1).unsqueeze(1)
        else:
            x_4d = x

        x_c = x_4d if x_4d.is_contiguous() else x_4d.contiguous()
        gamma_c = self.weight if self.weight.is_contiguous() else self.weight.contiguous()

        x_half = x_c.half() if x_c.dtype != torch.float16 else x_c
        gamma_half = gamma_c.half() if gamma_c.dtype != torch.float16 else gamma_c

        res = rmsnorm_cuda.forward(x_half, gamma_half, self.eps)
        out = res[0] if isinstance(res, (tuple, list)) else res
        
        out = out.view(orig_shape)
        return out.to(orig_dtype) if out.dtype != orig_dtype else out