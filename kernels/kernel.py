import sys
from pathlib import Path
import torch

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


class TopK:
    def __call__(self, x, k):
        orig_dtype = x.dtype
        x_f32 = x.float() if x.dtype != torch.float32 else x
        result = topk_cuda.topk(x_f32, k)
        if isinstance(result, (list, tuple)):
            result = result[0]
        return result.to(orig_dtype) if torch.is_tensor(result) else result


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        orig_dtype = x.dtype
        x_half = x.half().contiguous() if x.dtype != torch.float16 else (x if x.is_contiguous() else x.contiguous())
        y = softmax_cuda.forward(x_half)
        ctx.save_for_backward(x_half)
        ctx.orig_dtype = orig_dtype
        return y.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x_half, = ctx.saved_tensors
        grad_half = grad_output.half().contiguous() if grad_output.dtype != torch.float16 else (grad_output if grad_output.is_contiguous() else grad_output.contiguous())
        final = softmax_cuda.backward(grad_half, x_half)
        return final.to(ctx.orig_dtype)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, eps=1e-4):
        orig_shape = x.shape
        orig_dtype = x.dtype
        if x.dim() == 3:
            x_4d = x.unsqueeze(1)
        elif x.dim() == 2:
            x_4d = x.unsqueeze(1).unsqueeze(1)
        else:
            x_4d = x

        x_c = x_4d if x_4d.is_contiguous() else x_4d.contiguous()
        gamma_c = gamma if gamma.is_contiguous() else gamma.contiguous()

        x_half = x_c.half() if x_c.dtype != torch.float16 else x_c
        gamma_half = gamma_c.half() if gamma_c.dtype != torch.float16 else gamma_c

        result = rmsnorm_cuda.forward(x_half, gamma_half, eps)
        y = result[0] if isinstance(result, (list, tuple)) else result

        ctx.save_for_backward(x_half, gamma_half)
        ctx.eps = eps
        ctx.orig_shape = orig_shape
        ctx.orig_dtype = orig_dtype
        return y.view(orig_shape).to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x_half, gamma_half = ctx.saved_tensors
        if grad_output.dim() == 3:
            grad_4d = grad_output.unsqueeze(1)
        elif grad_output.dim() == 2:
            grad_4d = grad_output.unsqueeze(1).unsqueeze(1)
        else:
            grad_4d = grad_output

        grad_c = grad_4d if grad_4d.is_contiguous() else grad_4d.contiguous()
        grad_half = grad_c.half() if grad_c.dtype != torch.float16 else grad_c

        dx, dgamma = rmsnorm_cuda.backward(grad_half, x_half, gamma_half, ctx.eps)
        return dx.view(ctx.orig_shape).to(ctx.orig_dtype), dgamma.to(ctx.orig_dtype), None


class FlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        orig_dtype = q.dtype
        q_c = q if q.is_contiguous() else q.contiguous()
        k_c = k if k.is_contiguous() else k.contiguous()
        v_c = v if v.is_contiguous() else v.contiguous()

        q_half = q_c.half() if q_c.dtype != torch.float16 else q_c
        k_half = k_c.half() if k_c.dtype != torch.float16 else k_c
        v_half = v_c.half() if v_c.dtype != torch.float16 else v_c

        out, L = flashattn.flash_fwd(q_half, k_half, v_half, causal)
        ctx.save_for_backward(q_half, k_half, v_half, out, L)
        ctx.causal = causal
        ctx.orig_dtype = orig_dtype
        return out.to(orig_dtype)

    @staticmethod
    def backward(ctx, grad_output):
        q_half, k_half, v_half, out, L = ctx.saved_tensors
        grad_c = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
        grad_half = grad_c.half() if grad_c.dtype != torch.float16 else grad_c

        dq, dk, dv = flashattn.flash_bwd(q_half, k_half, v_half, out, grad_half, L, ctx.causal)
        return dq.to(ctx.orig_dtype), dk.to(ctx.orig_dtype), dv.to(ctx.orig_dtype), None


def rope_cache(reference, max_seq_len, rotary_dim):
    out = rope_cuda.build_cache(reference, max_seq_len, rotary_dim, 10000.0)
    cos = out[0].float().contiguous()
    sin = out[1].float().contiguous()
    return cos, sin


class Rope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, position_ids, cos, sin, rotary_dim, position_offset):
        orig_dtype = x.dtype
        x_c = x if x.is_contiguous() else x.contiguous()
        x_half = x_c.half() if x_c.dtype != torch.float16 else x_c
        cos_f32 = cos if cos.dtype == torch.float32 else cos.float()
        sin_f32 = sin if sin.dtype == torch.float32 else sin.float()

        result = rope_cuda.forward(x_half, position_ids, cos_f32, sin_f32, rotary_dim, position_offset)

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
        if ctx.has_position_ids:
            position_ids, cos_f32, sin_f32 = ctx.saved_tensors
        else:
            cos_f32, sin_f32 = ctx.saved_tensors
            position_ids = None

        grad_c = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
        grad_half = grad_c.half() if grad_c.dtype != torch.float16 else grad_c

        dx = rope_cuda.backward(grad_half, position_ids, cos_f32, sin_f32, ctx.rotary_dim, ctx.position_offset)
        return dx.to(ctx.orig_dtype), None, None, None, None, None