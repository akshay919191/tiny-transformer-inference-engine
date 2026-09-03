import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from kernels.kernel import (
    RMSNormFunction,
    FlashAttn,
    Rope,
    rope_cache,
)

from models.mqa import MQA_Cached
from models.model_config import ModelConfig


CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required",
)


def manual_rope(
    x,
    cos,
    sin,
    rotary_dim,
    position_offset=0,
):

    S = x.shape[-2]

    half = rotary_dim // 2

    cos_half = cos[
        position_offset:position_offset + S,
        :half,
    ]

    sin_half = sin[
        position_offset:position_offset + S,
        :half,
    ]

    cos_s = torch.cat(
        [cos_half, cos_half],
        dim=-1,
    )

    sin_s = torch.cat(
        [sin_half, sin_half],
        dim=-1,
    )

    x_rot = x[..., :rotary_dim]

    x_pass = x[..., rotary_dim:]

    x1, x2 = x_rot.chunk(
        2,
        dim=-1,
    )

    rotated = torch.cat(
        [-x2, x1],
        dim=-1,
    )

    out_rot = (
        x_rot * cos_s
        + rotated * sin_s
    )

    return torch.cat(
        [out_rot, x_pass],
        dim=-1,
    )


def manual_attention(
    q,
    k,
    v,
    causal,
):

    head_dim = q.shape[-1]

    scores = (
        q @ k.transpose(-1, -2)
    ) / math.sqrt(head_dim)

    if causal:

        SQ = q.shape[-2]
        SK = k.shape[-2]

        query_positions = torch.arange(
            SK - SQ,
            SK,
            device=q.device,
        )

        key_positions = torch.arange(
            SK,
            device=q.device,
        )

        mask = (
            key_positions.unsqueeze(0)
            >
            query_positions.unsqueeze(1)
        )

        scores = scores.masked_fill(
            mask,
            float("-inf"),
        )

    attn = F.softmax(
        scores.float(),
        dim=-1,
    ).to(q.dtype)

    return attn @ v


@CUDA_ONLY
def test_rmsnorm_forward():

    torch.manual_seed(0)

    B = 2
    H = 2
    S = 8
    D = 64

    eps = 1e-4

    x = torch.randn(
        B,
        H,
        S,
        D,
        device="cuda",
        dtype=torch.float32,
    )

    gamma = torch.randn(
        D,
        device="cuda",
        dtype=torch.float32,
    )

    out_cuda = RMSNormFunction.apply(
        x,
        gamma,
        eps,
    )

    ref = (
        x
        * torch.rsqrt(
            x.pow(2).mean(
                dim=-1,
                keepdim=True,
            )
            + eps
        )
    )

    out_ref = ref * gamma

    max_diff = (
        out_cuda - out_ref
    ).abs().max().item()

    print(
        f"RMSNorm max diff: {max_diff}"
    )

    assert torch.allclose(
        out_cuda,
        out_ref,
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.parametrize(
    "causal",
    [True, False],
)
@CUDA_ONLY
def test_flash_attention(causal):

    torch.manual_seed(0)

    B = 2
    H = 4
    S = 16
    D = 32

    q = torch.randn(
        B,
        H,
        S,
        D,
        device="cuda",
        dtype=torch.float16,
    ).contiguous()

    k = torch.randn(
        B,
        H,
        S,
        D,
        device="cuda",
        dtype=torch.float16,
    ).contiguous()

    v = torch.randn(
        B,
        H,
        S,
        D,
        device="cuda",
        dtype=torch.float16,
    ).contiguous()

    out_cuda = FlashAttn.apply(
        q,
        k,
        v,
        causal,
    )

    out_ref = manual_attention(
        q,
        k,
        v,
        causal,
    )

    max_diff = (
        out_cuda - out_ref
    ).abs().max().item()

    print(
        f"FlashAttn causal={causal} "
        f"max diff: {max_diff}"
    )

    assert torch.allclose(
        out_cuda,
        out_ref,
        atol=5e-2,
        rtol=5e-2,
    )


@CUDA_ONLY
def test_rope_forward():

    torch.manual_seed(0)

    B = 2
    H = 4
    S = 16
    D = 32

    rotary_dim = D
    max_seq_len = 64

    reference = torch.empty(
        1,
        dtype=torch.float16,
        device="cuda",
    )

    cos, sin = rope_cache(
        reference,
        max_seq_len,
        rotary_dim,
    )

    x = torch.randn(
        B,
        H,
        S,
        D,
        device="cuda",
        dtype=torch.float16,
    ).contiguous()

    cos = cos.float().contiguous()
    sin = sin.float().contiguous()

    out_cuda = Rope.apply(
        x,
        None,
        cos,
        sin,
        rotary_dim,
        0,
    )

    out_ref = manual_rope(
        x,
        cos,
        sin,
        rotary_dim,
        0,
    )

    # Compare in the same dtype.
    out_cuda_f32 = out_cuda.float()
    out_ref_f32 = out_ref.float()

    max_diff = (
        out_cuda_f32 - out_ref_f32
    ).abs().max().item()

    print(
        f"RoPE max diff: {max_diff}"
    )

    assert torch.allclose(
        out_cuda_f32,
        out_ref_f32,
        atol=2e-2,
        rtol=2e-2,
    )


@CUDA_ONLY
def test_mqa_cached_backend_parity():

    torch.manual_seed(0)

    config = ModelConfig(
        vocab_size=50257,
        hidden_size=512,
        num_layers=1,
        num_heads=8,
        num_kv_heads=1,
        max_seq_len=64,
        d_model=256,
    )

    model_pytorch = MQA_Cached(
        config,
        backend="pytorch",
    ).cuda().half()

    model_cuda = MQA_Cached(
        config,
        backend="cuda",
    ).cuda().half()

    model_cuda.load_state_dict(
        model_pytorch.state_dict(),
        strict=False,
    )

    B = 2
    S = 16

    x = torch.randn(
        B,
        S,
        config.d_model,
        device="cuda",
        dtype=torch.float16,
    )

    model_pytorch.eval()
    model_cuda.eval()

    with torch.no_grad():

        out_pytorch = model_pytorch(
            x,
            x,
            x,
            causal=True,
        )

        out_cuda = model_cuda(
            x,
            x,
            x,
            causal=True,
        )

    max_diff = (
        out_pytorch - out_cuda
    ).abs().max().item()

    print(
        f"MQA_Cached backend max diff: "
        f"{max_diff}"
    )

    assert torch.allclose(
        out_pytorch,
        out_cuda,
        atol=5e-2,
        rtol=5e-2,
    )


@CUDA_ONLY
def test_mqa_cached_incremental():

    torch.manual_seed(0)

    config = ModelConfig(
        vocab_size=50257,
        hidden_size=512,
        num_layers=1,
        num_heads=8,
        num_kv_heads=1,
        max_seq_len=64,
        d_model=256,
    )

    model = MQA_Cached(
        config,
        backend="cuda",
    ).cuda().half()

    model.eval()

    B = 1
    prompt_len = 8

    prompt = torch.randn(
        B,
        prompt_len,
        config.d_model,
        device="cuda",
        dtype=torch.float16,
    )

    from kv_cache import KVCache_kv

    cache = KVCache_kv(
        num_layers=1,
        batch_size=B,
        num_heads=config.num_kv_heads,
        max_seq_len=config.max_seq_len,
        head_dim=config.d_model // config.num_heads,
        dtype=torch.float16,
        device="cuda",
    )

    with torch.no_grad():

        out_prefill = model(
            prompt,
            prompt,
            prompt,
            kv_cache=cache,
            layer_idx=0,
            causal=True,
        )

        cache.advance(
            prompt_len
        )

        next_token = torch.randn(
            B,
            1,
            config.d_model,
            device="cuda",
            dtype=torch.float16,
        )

        out_decode = model(
            next_token,
            next_token,
            next_token,
            kv_cache=cache,
            layer_idx=0,
            causal=True,
        )

    print(
        "Prefill output shape:",
        out_prefill.shape,
    )

    print(
        "Decode output shape:",
        out_decode.shape,
    )

    assert out_prefill.shape == (
        B,
        prompt_len,
        config.d_model,
    )

    assert out_decode.shape == (
        B,
        1,
        config.d_model,
    )


if __name__ == "__main__":

    sys.exit(
        pytest.main(
            [
                __file__,
                "-v",
            ]
        )
    )