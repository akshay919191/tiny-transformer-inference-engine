import sys
from pathlib import Path
import itertools

import torch
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.mqa import MQA, MQA_Cached
from kv_cache import KVCache_kv


@dataclass
class ModelConfig:

    vocab_size: int = 50257
    hidden_size: int = 128
    num_layers: int = 1
    num_heads: int = 8
    num_kv_heads: int = 1
    max_seq_len: int = 512
    d_model: int = 256
    causal: bool = True
    dropout: float = 0.0
    bias: bool = False
    batch: int = 1
    dtype: torch.dtype = torch.float32


device = "cuda"
torch.manual_seed(42)


# Which backend to actually validate. "cuda" is what exercises
# FlashAttn.apply / the kernel's GQA-MQA head-grouping and the
# dK/dV atomic-accumulation path — "pytorch" only checks the
# reference implementation against itself and proves nothing
# about the kernel.
BACKEND = "cuda"

# fp16 is required by the cuda RoPE/attention kernels; the pytorch
# backend can run in fp32 for a tighter numerical baseline.
DTYPE = torch.float16 if BACKEND == "cuda" else torch.float32

# fp16 accumulation needs looser tolerances than fp32.
ATOL = 1e-2 if BACKEND == "cuda" else 1e-5
RTOL = 1e-2 if BACKEND == "cuda" else 1e-4


BATCHES = [1, 2, 4]

SEQLENS = [128, 256]

DMODELS = [256, 512]

HEAD_CONFIGS = [
    (4, 1),   # MQA
    (4, 2),   # GQA
    (8, 1),   # MQA
    (8, 2),   # GQA
    (8, 4),   # GQA
    (8, 8),   # MHA
]



@torch.no_grad()
def test_one(
    batch,
    seq_len,
    d_model,
    num_heads,
    num_kv_heads,
):

    assert d_model % num_heads == 0
    assert num_heads % num_kv_heads == 0

    config = ModelConfig(
        batch=batch,
        max_seq_len=seq_len + 1,
        d_model=d_model,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        dtype=DTYPE,
    )

    # Both models must use the SAME backend so the comparison actually
    # validates that backend's cache-vs-full-sequence equivalence.
    reference = MQA(
        config,
        backend=BACKEND,
    ).cuda()

    cached = MQA_Cached(
        config,
        backend=BACKEND,
    ).cuda()

    if DTYPE == torch.float16:
        reference = reference.half()
        cached = cached.half()

    reference.eval()
    cached.eval()

    cached.load_state_dict(
        reference.state_dict()
    )


    prompt = torch.randn(
        batch,
        seq_len,
        d_model,
        device=device,
        dtype=config.dtype,
    )

    next_token = torch.randn(
        batch,
        1,
        d_model,
        device=device,
        dtype=config.dtype,
    )

    full_input = torch.cat(
        [
            prompt,
            next_token,
        ],
        dim=1,
    )

    reference_output = reference(
        full_input,
        full_input,
        full_input,
        causal=True,
    )

    reference_last = reference_output[:, -1:, :]

    cache = KVCache_kv(
        num_layers=1,
        batch_size=batch,
        num_heads=num_kv_heads,
        max_seq_len=seq_len + 1,
        head_dim=d_model // num_heads,
        dtype=config.dtype,
        device=device,
    )

    _ = cached(
        prompt,
        prompt,
        prompt,
        kv_cache=cache,
        layer_idx=0,
        causal=True,
    )

    cache.advance(seq_len)

    cached_last = cached(
        next_token,
        next_token,
        next_token,
        kv_cache=cache,
        layer_idx=0,
        causal=True,
    )

    cache.advance(1)

    error = (
        cached_last.float() - reference_last.float()
    ).abs().max().item()

    passed = torch.allclose(
        cached_last.float(),
        reference_last.float(),
        atol=ATOL,
        rtol=RTOL,
    )

    name = (
        f"backend={BACKEND} "
        f"B={batch} "
        f"S={seq_len} "
        f"D={d_model} "
        f"H={num_heads} "
        f"KV={num_kv_heads}"
    )

    if passed:
        print(
            f"[PASS] {name} "
            f"| max error = {error:.6e}"
        )
    else:
        print(
            f"[FAIL] {name} "
            f"| max error = {error:.6e}"
        )

    return passed



def main():

    total = 0
    passed = 0

    for (
        batch,
        seq_len,
        d_model,
        (num_heads, num_kv_heads),
    ) in itertools.product(
        BATCHES,
        SEQLENS,
        DMODELS,
        HEAD_CONFIGS,
    ):

        total += 1

        try:

            ok = test_one(
                batch=batch,
                seq_len=seq_len,
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )

            if ok:
                passed += 1

        except Exception as e:

            print(
                f"[ERROR] "
                f"B={batch} "
                f"S={seq_len} "
                f"D={d_model} "
                f"H={num_heads} "
                f"KV={num_kv_heads}"
            )

            print(
                f"        {type(e).__name__}: {e}"
            )

    print()
    print("=" * 70)
    print(
        f"RESULT: {passed}/{total} tests passed "
        f"(backend={BACKEND})"
    )
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()