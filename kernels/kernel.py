"""
here are the kernels will be warpped up for usage
"""


import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT / "kernels"

KERNEL_PATHS = [
    CUDA_DIR / "rmsnorm_kernel",
    CUDA_DIR / "softmax_kernel",
    CUDA_DIR / "rope_kernel",
    CUDA_DIR / "cuda-kSAMPLING",
    CUDA_DIR / "flashattn",
]

for p in KERNEL_PATHS:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


import rmsnorm_cuda
import softmax_cuda
import rope_cuda
import flash_acc_reg_ext as flashattn
import topk_cuda

"""
classes for each kernel so it can work with autograd
"""

class RMSNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, gamma, eps=1e-4):

        y = rmsnorm_cuda.forward(
            x,
            gamma,
            eps
        )

        ctx.save_for_backward(x, gamma)
        ctx.eps = eps

        return y

    @staticmethod
    def backward(ctx, grad_output):

        x, gamma = ctx.saved_tensors
        eps = ctx.eps

        dx, dgamma = rmsnorm_cuda.backward(
            grad_output,
            x,
            gamma,
            eps
        )

        return dx, dgamma, None

class Softmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx , x):
        y = softmax_cuda.forward(x)

        ctx.save_for_backward(x)

        return y

    @staticmethod
    def backward(ctx , grad_output):
        x, = ctx.saved_tensors

        final = softmax_cuda.backward(grad_output , x)

        return final

class TopK:
    def __call__(self, x, k):
        return topk_cuda.topk(x, k)

class FlashAttn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool
    ):
        out, L = flashattn.flash_fwd(
            q, k, v, causal
        )

        ctx.save_for_backward(
            q, k, v, out, L
        )

        ctx.causal = causal

        return out

    @staticmethod
    def backward(ctx, grad_output):

        q, k, v, out, L = ctx.saved_tensors
        causal = ctx.causal

        dq, dk, dv = flashattn.flash_bwd(
            q,
            k,
            v,
            out,
            grad_output,
            L,
            causal
        )

        return dq, dk, dv, None


### rope has diff story we build static cache

## cache

def rope_cache(reference, max_seq_len, rotary_dim):
    out = rope_cuda.build_cache(
        reference,
        max_seq_len,
        rotary_dim,
        10000.0
    )

    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise AssertionError(
            "RoPE build_cache should return two tensors: "
            "cos_cache, sin_cache"
        )

    cos = out[0].float().contiguous()
    sin = out[1].float().contiguous()

    return cos, sin


class Rope(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        position_ids,
        cos,
        sin,
        rotary_dim,
        position_offset,
    ):
        result = rope_cuda.forward(
            x,
            position_ids,
            cos,
            sin,
            rotary_dim,
            position_offset,
        )

        if position_ids is not None:
            ctx.save_for_backward(position_ids, cos, sin)
        else:
            ctx.save_for_backward(cos, sin)

        ctx.has_position_ids = position_ids is not None
        ctx.rotary_dim = rotary_dim
        ctx.position_offset = position_offset

        return result

    @staticmethod
    def backward(ctx, grad_output):

        if ctx.has_position_ids:
            position_ids, cos, sin = ctx.saved_tensors
        else:
            cos, sin = ctx.saved_tensors
            position_ids = None

        dx = rope_cuda.backward(
            grad_output,
            position_ids,
            cos,
            sin,
            ctx.rotary_dim,
            ctx.position_offset,
        )

        return (
            dx,
            None,
            None,
            None,
            None,
            None,
        )