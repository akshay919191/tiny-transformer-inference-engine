import sys
from pathlib import Path
from typing import Optional, Tuple
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


def _fp16c(x: torch.Tensor) -> torch.Tensor:
    """FP16 + contiguous."""
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    return x if x.is_contiguous() else x.contiguous()


def _f32(x: torch.Tensor) -> torch.Tensor:
    return x if x.dtype == torch.float32 else x.float()


def _first(r):
    """Some kernels return [tensor, ...]; take the primary output."""
    return r[0] if isinstance(r, (list, tuple)) else r


def _to4d(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 2:
        return x.unsqueeze(1).unsqueeze(1)
    return x


def _fresh_like(t: torch.Tensor) -> torch.Tensor:
    """Fresh, DEFAULT-CONTIGUOUS empty tensor with t's shape/dtype/device.

    NOT torch.empty_like(t): empty_like also copies t's *strides*, which is
    wrong whenever t is a non-contiguous view (e.g. a .transpose(1, 2) of a
    Q/K/V projection). Every real kernel here calls _fp16c()/.contiguous()
    internally and returns a freshly allocated, standard row-major tensor --
    it never inherits the input's original memory layout. A fake (meta)
    kernel that uses empty_like(input) on a transposed input will report the
    wrong output strides to Dynamo/Inductor; under torch.compile that mismatch
    surfaces at runtime as an `assert_size_stride` failure (or worse, wrong
    results, if the size happens to still line up as it did here since
    batch == num_heads).
    """
    return torch.empty(t.shape, dtype=t.dtype, device=t.device)


# ---------------------------------------------------------------------------
# TopK
#   Input  -> float32 (if not already)
#   Output -> original dtype
# ---------------------------------------------------------------------------
@torch.library.custom_op("custom_cuda::topk", mutates_args=())
def _topk_op(x: torch.Tensor, k: int) -> torch.Tensor:
    orig_dtype = x.dtype
    x_f32 = _f32(x)
    x_f32 = x_f32 if x_f32.is_contiguous() else x_f32.contiguous()
    result = _first(topk_cuda.topk(x_f32, k))
    return result.to(orig_dtype)

@_topk_op.register_fake
def _(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.empty(*x.shape[:-1], k, dtype=x.dtype, device=x.device)


# ---------------------------------------------------------------------------
# Softmax
#   Input        -> float16
#   Output       -> original dtype (the forward input's dtype)
#   grad_output  -> float16
#   Result       -> original dtype (the forward input's dtype, not grad's)
# ---------------------------------------------------------------------------
@torch.library.custom_op("custom_cuda::softmax_fwd", mutates_args=())
def _softmax_fwd_op(x: torch.Tensor) -> torch.Tensor:
    return _first(softmax_cuda.forward(_fp16c(x))).to(x.dtype)

@_softmax_fwd_op.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    return _fresh_like(x)

@torch.library.custom_op("custom_cuda::softmax_bwd", mutates_args=())
def _softmax_bwd_op(grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # cast to x's original dtype (== ctx.orig_dtype in the autograd.Function version),
    # not grad's dtype
    return _first(softmax_cuda.backward(_fp16c(grad), _fp16c(x))).to(x.dtype)

@_softmax_bwd_op.register_fake
def _(grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return _fresh_like(grad)

def _softmax_setup(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)

def _softmax_backward(ctx, grad_out):
    (x,) = ctx.saved_tensors
    return torch.ops.custom_cuda.softmax_bwd(grad_out, x)

torch.library.register_autograd(
    "custom_cuda::softmax_fwd", _softmax_backward, setup_context=_softmax_setup
)


# ---------------------------------------------------------------------------
# RMSNorm
#   x, gamma -> float16
#   Output   -> original dtype (x's)
#   grad_output -> float16
#   dx, dgamma  -> original dtype (x's, for BOTH outputs)
# ---------------------------------------------------------------------------
@torch.library.custom_op("custom_cuda::rmsnorm_fwd", mutates_args=())
def _rmsnorm_fwd_op(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    orig_shape = x.shape
    result = rmsnorm_cuda.forward(_fp16c(_to4d(x)), _fp16c(gamma), eps)
    y = result[0] if isinstance(result, (list, tuple)) else result
    return y.reshape(orig_shape).to(x.dtype)

@_rmsnorm_fwd_op.register_fake
def _(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    return _fresh_like(x)

@torch.library.custom_op("custom_cuda::rmsnorm_bwd", mutates_args=())
def _rmsnorm_bwd_op(
    grad: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_shape = x.shape
    dx, dgamma = rmsnorm_cuda.backward(
        _fp16c(_to4d(grad)), _fp16c(_to4d(x)), _fp16c(gamma), eps
    )
    # both outputs cast to x's original dtype (== ctx.orig_dtype in the
    # autograd.Function version) -- dgamma does NOT use gamma.dtype
    return dx.reshape(orig_shape).to(x.dtype), dgamma.to(x.dtype)

@_rmsnorm_bwd_op.register_fake
def _(grad: torch.Tensor, x: torch.Tensor, gamma: torch.Tensor, eps: float):
    return _fresh_like(x), _fresh_like(gamma)

def _rmsnorm_setup(ctx, inputs, output):
    x, gamma, eps = inputs
    ctx.save_for_backward(x, gamma)
    ctx.eps = eps

def _rmsnorm_backward(ctx, grad_out):
    x, gamma = ctx.saved_tensors
    dx, dgamma = torch.ops.custom_cuda.rmsnorm_bwd(grad_out, x, gamma, ctx.eps)
    return dx, dgamma, None

torch.library.register_autograd(
    "custom_cuda::rmsnorm_fwd", _rmsnorm_backward, setup_context=_rmsnorm_setup
)


# ---------------------------------------------------------------------------
# FlashAttn
#   q, k, v      -> float16 (independently checked/cast)
#   Output       -> original dtype (q's)
#   grad_output  -> float16
#   dq, dk, dv   -> original dtype (q's, for ALL THREE outputs)
#
#   q/k/v are almost always non-contiguous transposed views (.transpose(1,2)
#   of a QKV projection) -- this is the op where the empty_like-inherits-
#   strides bug actually bit (see _fresh_like docstring above).
# ---------------------------------------------------------------------------
@torch.library.custom_op("custom_cuda::flash_fwd", mutates_args=())
def _flash_fwd_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, L = flashattn.flash_fwd(_fp16c(q), _fp16c(k), _fp16c(v), causal)
    return out.to(q.dtype), L

@_flash_fwd_op.register_fake
def _(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool):
    out = _fresh_like(q)
    # L layout assumed (B, H, S) from a (B, H, S, D) q -- adjust if yours differs
    L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), dtype=torch.float32, device=q.device)
    return out, L

@torch.library.custom_op("custom_cuda::flash_bwd", mutates_args=())
def _flash_bwd_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    out: torch.Tensor, grad: torch.Tensor, L: torch.Tensor, causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = flashattn.flash_bwd(
        _fp16c(q), _fp16c(k), _fp16c(v), _fp16c(out), _fp16c(grad), L, causal
    )
    # all three cast to q's original dtype (== ctx.orig_dtype in the
    # autograd.Function version) -- dk/dv do NOT use k.dtype/v.dtype
    return dq.to(q.dtype), dk.to(q.dtype), dv.to(q.dtype)

@_flash_bwd_op.register_fake
def _(q, k, v, out, grad, L, causal):
    return _fresh_like(q), _fresh_like(k), _fresh_like(v)

def _flash_setup(ctx, inputs, output):
    q, k, v, causal = inputs
    out, L = output
    ctx.mark_non_differentiable(L)
    ctx.save_for_backward(q, k, v, out, L)
    ctx.causal = causal

def _flash_backward(ctx, grad_out, grad_L):
    q, k, v, out, L = ctx.saved_tensors
    dq, dk, dv = torch.ops.custom_cuda.flash_bwd(q, k, v, out, grad_out, L, ctx.causal)
    return dq, dk, dv, None

torch.library.register_autograd(
    "custom_cuda::flash_fwd", _flash_backward, setup_context=_flash_setup
)


# ---------------------------------------------------------------------------
# Rope
#   x           -> float16
#   cos, sin    -> float32 (if not already)
#   Output      -> original dtype (x's)
#   grad_output -> float16
#   dx          -> original dtype (x's; approximated via grad's dtype since
#                  x itself isn't saved for backward -- see note below)
#
#   x is also almost always a non-contiguous transposed Q/K view here, same
#   as FlashAttn -- fake must not inherit its strides.
# ---------------------------------------------------------------------------
@torch.library.custom_op("custom_cuda::rope_fwd", mutates_args=())
def _rope_fwd_op(
    x: torch.Tensor, position_ids: Optional[torch.Tensor],
    cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int, position_offset: int,
) -> torch.Tensor:
    out = _first(rope_cuda.forward(_fp16c(x), position_ids, _f32(cos), _f32(sin), rotary_dim, position_offset))
    return out.to(x.dtype)

@_rope_fwd_op.register_fake
def _(x, position_ids, cos, sin, rotary_dim, position_offset):
    return _fresh_like(x)

@torch.library.custom_op("custom_cuda::rope_bwd", mutates_args=())
def _rope_bwd_op(
    grad: torch.Tensor, position_ids: Optional[torch.Tensor],
    cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int, position_offset: int,
) -> torch.Tensor:
    dx = _first(rope_cuda.backward(_fp16c(grad), position_ids, _f32(cos), _f32(sin), rotary_dim, position_offset))
    # NOTE: original x is not saved for backward, so we cast to grad's dtype.
    # grad_output's dtype always matches the forward output's dtype (which was
    # itself cast to x's original dtype), so this is equivalent in practice.
    return dx.to(grad.dtype)

@_rope_bwd_op.register_fake
def _(grad, position_ids, cos, sin, rotary_dim, position_offset):
    return _fresh_like(grad)

def _rope_setup(ctx, inputs, output):
    x, position_ids, cos, sin, rotary_dim, position_offset = inputs
    ctx.save_for_backward(position_ids, cos, sin)  # position_ids may be None
    ctx.rotary_dim = rotary_dim
    ctx.position_offset = position_offset

def _rope_backward(ctx, grad_out):
    position_ids, cos, sin = ctx.saved_tensors
    dx = torch.ops.custom_cuda.rope_bwd(
        grad_out, position_ids, cos, sin, ctx.rotary_dim, ctx.position_offset
    )
    return dx, None, None, None, None, None

torch.library.register_autograd(
    "custom_cuda::rope_fwd", _rope_backward, setup_context=_rope_setup
)


class TopK:
    def __call__(self, x, k):
        return torch.ops.custom_cuda.topk(x, k)


class Softmax:
    @staticmethod
    def apply(x):
        return torch.ops.custom_cuda.softmax_fwd(x)


class RMSNormFunction:
    @staticmethod
    def apply(x, gamma, eps=1e-4):
        return torch.ops.custom_cuda.rmsnorm_fwd(x, gamma, eps)


class FlashAttn:
    @staticmethod
    def apply(q, k, v, causal):
        out, _ = torch.ops.custom_cuda.flash_fwd(q, k, v, causal)
        return out


class Rope:
    @staticmethod
    def apply(x, position_ids, cos, sin, rotary_dim, position_offset):
        return torch.ops.custom_cuda.rope_fwd(x, position_ids, cos, sin, rotary_dim, position_offset)


def rope_cache(reference, max_seq_len, rotary_dim):
    out = rope_cuda.build_cache(reference, max_seq_len, rotary_dim, 10000.0)
    cos = out[0].float().contiguous()
    sin = out[1].float().contiguous()
    return cos, sin