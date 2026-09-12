import torch
import torch.nn.functional as F

from kernels.kernel import rmsnorm, flash_attn, rope, rope_cache


def test_rmsnorm_grad():
    torch.manual_seed(0)
    B, H, N, D = 2, 1, 16, 256

    x = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
    g = torch.ones(D, device="cuda", dtype=torch.float32, requires_grad=True)

    x_ref = x.detach().clone().requires_grad_(True)
    g_ref = g.detach().clone().requires_grad_(True)

    out = rmsnorm(x, g, 1e-5)
    assert out.requires_grad, "FAST PATH LEAKED INTO TRAINING"

    rms = x_ref.float().pow(2).mean(-1, keepdim=True).add(1e-5).rsqrt()
    out_ref = (x_ref * rms) * g_ref

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)

    print("rmsnorm dx err :", (x.grad - x_ref.grad).abs().max().item())
    print("rmsnorm dg err :", (g.grad - g_ref.grad).abs().max().item())


def test_flash_grad():
    torch.manual_seed(0)
    B, H, N, D = 2, 4, 64, 64

    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, requires_grad=True)

    qr, kr, vr = (t.detach().clone().requires_grad_(True) for t in (q, k, v))

    out = flash_attn(q, k, v, True)
    assert out.requires_grad, "FAST PATH LEAKED INTO TRAINING"

    out_ref = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True)

    grad = torch.randn_like(out)
    out.backward(grad)
    out_ref.backward(grad)

    print("flash out err :", (out - out_ref).abs().max().item())
    print("flash dq err  :", (q.grad - qr.grad).abs().max().item())
    print("flash dk err  :", (k.grad - kr.grad).abs().max().item())
    print("flash dv err  :", (v.grad - vr.grad).abs().max().item())


def test_fast_path_engaged():
    """Under no_grad the wrapper must NOT build a graph."""
    x = torch.randn(1, 1, 8, 256, device="cuda", dtype=torch.float16, requires_grad=True)
    g = torch.ones(256, device="cuda", dtype=torch.float16, requires_grad=True)

    with torch.no_grad():
        out = rmsnorm(x, g, 1e-5)
    assert not out.requires_grad, "autograd path taken during inference"
    print("fast path OK")


if __name__ == "__main__":
    test_rmsnorm_grad()
    test_flash_grad()
    test_fast_path_engaged()