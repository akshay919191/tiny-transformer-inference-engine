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
classes for each kernel so it can work with autograd.

IMPORTANT: these kernels require fp16 inputs internally. Rather than
requiring every caller to remember to cast, each Function casts to fp16
right before calling into CUDA, and casts results back to the ORIGINAL
input dtype before returning. This means:
  - callers can pass normal fp32 tensors (e.g. straight from an fp32
    model/optimizer) without thinking about dtype at all
  - the model's actual parameters (gamma, etc.) stay fp32 for the
    optimizer, avoiding the eps-underflow NaN issue
  - gradients returned to autograd match the dtype of what was passed in,
    which autograd requires
"""


class RMSNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, gamma, eps=1e-4):
        orig_dtype = x.dtype

        x_half = x.half()
        gamma_half = gamma.half()

        y = rmsnorm_cuda.forward(
            x_half,
            gamma_half,
            eps
        )

        ctx.save_for_backward(x_half, gamma_half)
        ctx.eps = eps
        ctx.orig_dtype = orig_dtype

        return y.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x_half, gamma_half = ctx.saved_tensors
        eps = ctx.eps
        orig_dtype = ctx.orig_dtype

        grad_output_half = grad_output.half().contiguous()

        dx, dgamma = rmsnorm_cuda.backward(
            grad_output_half,
            x_half,
            gamma_half,
            eps
        )

        return dx.to(orig_dtype), dgamma.to(orig_dtype), None


class Softmax(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        orig_dtype = x.dtype
        x_half = x.half()

        y = softmax_cuda.forward(x_half)

        ctx.save_for_backward(x_half)
        ctx.orig_dtype = orig_dtype

        return y.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x_half, = ctx.saved_tensors
        orig_dtype = ctx.orig_dtype

        grad_output_half = grad_output.half().contiguous()

        final = softmax_cuda.backward(grad_output_half, x_half)

        return final.to(orig_dtype)


class TopK:
    def __call__(self, x, k):
        orig_dtype = x.dtype
        result = topk_cuda.topk(x.half(), k)
        return result.to(orig_dtype) if torch.is_tensor(result) else result


class FlashAttn(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool
    ):
        orig_dtype = q.dtype

        q_half = q.half().contiguous()
        k_half = k.half().contiguous()
        v_half = v.half().contiguous()

        out, L = flashattn.flash_fwd(
            q_half, k_half, v_half, causal
        )

        ctx.save_for_backward(
            q_half, k_half, v_half, out, L
        )

        ctx.causal = causal
        ctx.orig_dtype = orig_dtype

        return out.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        q_half, k_half, v_half, out, L = ctx.saved_tensors
        causal = ctx.causal
        orig_dtype = ctx.orig_dtype

        grad_output_half = grad_output.half().contiguous()

        dq, dk, dv = flashattn.flash_bwd(
            q_half,
            k_half,
            v_half,
            out,
            grad_output_half,
            L,
            causal
        )

        return dq.to(orig_dtype), dk.to(orig_dtype), dv.to(orig_dtype), None


### rope has diff story we build static cache

## cache
# NOTE: cos/sin caches are deliberately kept in fp32 (not cast here) —
# they're built once and reused every forward pass, not part of the
# optimizer's parameter set, so keeping them fp32 costs nothing and
# avoids repeated precision loss across many reuses.

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
        orig_dtype = x.dtype

        x_half = x.half()
        # rope_cuda requires cos/sin as float32 specifically (mixed-precision
        # kernel signature) — do NOT cast these to half, unlike x.
        cos_f32 = cos.float().contiguous()
        sin_f32 = sin.float().contiguous()

        result = rope_cuda.forward(
            x_half,
            position_ids,
            cos_f32,
            sin_f32,
            rotary_dim,
            position_offset,
        )

        if position_ids is not None:
            ctx.save_for_backward(position_ids, cos_f32, sin_f32)
        else:
            ctx.save_for_backward(cos_f32, sin_f32)

        ctx.has_position_ids = position_ids is not None
        ctx.rotary_dim = rotary_dim
        ctx.position_offset = position_offset
        ctx.orig_dtype = orig_dtype

        return result.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        orig_dtype = ctx.orig_dtype

        if ctx.has_position_ids:
            position_ids, cos_f32, sin_f32 = ctx.saved_tensors
        else:
            cos_f32, sin_f32 = ctx.saved_tensors
            position_ids = None

        grad_output_half = grad_output.half().contiguous()

        dx = rope_cuda.backward(
            grad_output_half,
            position_ids,
            cos_f32,
            sin_f32,
            ctx.rotary_dim,
            ctx.position_offset,
        )

        return (
            dx.to(orig_dtype),
            None,
            None,
            None,
            None,
            None,
        )