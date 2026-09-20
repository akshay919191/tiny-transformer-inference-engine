
import torch
import triton
import triton.language as tl

_BLOCK_SIZE = 1024
_NUM_WARPS = 4


@triton.jit
def _swiglu_fwd_kernel(
    gate_ptr, up_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    result = gate * tl.sigmoid(gate) * up

    tl.store(out_ptr + offsets, result.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    grad_out_ptr, gate_ptr, up_ptr,
    grad_gate_ptr, grad_up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(gate)
    silu_gate = gate * sig
    dsilu_dgate = sig * (1.0 + gate * (1.0 - sig))

    grad_gate = grad_out * up * dsilu_dgate
    grad_up = grad_out * silu_gate

    tl.store(grad_gate_ptr + offsets, grad_gate.to(grad_gate_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_up_ptr + offsets, grad_up.to(grad_up_ptr.dtype.element_ty), mask=mask)


def _launch_fwd(gate, up):
    gate = gate.contiguous()
    up = up.contiguous()
    out = torch.empty_like(gate)
    n = gate.numel()
    grid = (triton.cdiv(n, _BLOCK_SIZE),)
    _swiglu_fwd_kernel[grid](gate, up, out, n, BLOCK_SIZE=_BLOCK_SIZE, num_warps=_NUM_WARPS)
    return out


def _launch_bwd(grad_out, gate, up):
    grad_out = grad_out.contiguous()
    gate = gate.contiguous()
    up = up.contiguous()
    grad_gate = torch.empty_like(gate)
    grad_up = torch.empty_like(up)
    n = gate.numel()
    grid = (triton.cdiv(n, _BLOCK_SIZE),)
    _swiglu_bwd_kernel[grid](
        grad_out, gate, up, grad_gate, grad_up, n,
        BLOCK_SIZE=_BLOCK_SIZE, num_warps=_NUM_WARPS,
    )
    return grad_gate, grad_up


_HAS_CUSTOM_OP = hasattr(torch.library, "custom_op")     # torch >= 2.4

if _HAS_CUSTOM_OP:
    @torch.library.custom_op("custom_triton::swiglu_fwd", mutates_args=())
    def _swiglu_fwd_op(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return _launch_fwd(gate, up)

    @_swiglu_fwd_op.register_fake
    def _(gate, up):
        return torch.empty_like(gate)

    @torch.library.custom_op("custom_triton::swiglu_bwd", mutates_args=())
    def _swiglu_bwd_op(
        grad_out: torch.Tensor, gate: torch.Tensor, up: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _launch_bwd(grad_out, gate, up)

    @_swiglu_bwd_op.register_fake
    def _(grad_out, gate, up):
        return torch.empty_like(gate), torch.empty_like(up)

    def _setup_context(ctx, inputs, output):
        gate, up = inputs
        ctx.save_for_backward(gate, up)

    def _backward(ctx, grad_out):
        gate, up = ctx.saved_tensors
        return _swiglu_bwd_op(grad_out, gate, up)

    _swiglu_fwd_op.register_autograd(_backward, setup_context=_setup_context)

    def _fused(gate, up):
        return _swiglu_fwd_op(gate, up)

else:
    class _FusedSwiGLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, gate, up):
            ctx.save_for_backward(gate, up)
            return _launch_fwd(gate, up)

        @staticmethod
        def backward(ctx, grad_out):
            gate, up = ctx.saved_tensors
            return _launch_bwd(grad_out, gate, up)

    def _fused(gate, up):
        if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
            return _FusedSwiGLU.apply(gate, up)
        return _launch_fwd(gate, up)


def fused_swiglu(gate, up):
    """silu(gate) * up, fused. gate and up must have the same shape and dtype."""
    assert gate.shape == up.shape and gate.dtype == up.dtype
    if not gate.is_cuda or gate.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return torch.nn.functional.silu(gate) * up          
    return _fused(gate, up)