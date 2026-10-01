"""
Accuracy + speed test: paged_varlen_attention  vs  torch SDPA

ASSUMED LAYOUTS (edit the CONFIG / gather_kv section if yours differ):
  q                : [total_q_tokens, H, D]            fp16
  k_pool, v_pool   : [num_blocks, block_size, Hkv, D]  fp16
  block_tables     : [num_seqs, max_blocks]            int32
  query_start_loc  : [num_seqs + 1]                    int32  (cumulative q lengths)
  positions        : [total_q_tokens]                  int32  absolute position of each
                     query token inside its own sequence (causal: key_j visible iff j <= pos)
  out              : [total_q_tokens, H, D]            fp16

Run:  python test_paged_attn.py
"""
import math
import sys
import torch
import torch.nn.functional as F

from serving.fake_paged_attn import paged_varlen_attention

torch.manual_seed(0)
dev = "cuda"
torch.set_grad_enabled(False)  # inference only; avoids the non_causal grad-path NotImplementedError
DTYPE = torch.float16

# ----------------------------------------------------------------------------
# Scenarios: list of (q_len, kv_len) per sequence. kv_len >= q_len.
#   prefill        : q_len == kv_len
#   decode         : q_len == 1, kv_len long
#   chunked prefill: q_len < kv_len (new chunk attends to cached context + itself)
# ----------------------------------------------------------------------------
SCENARIOS = {
    "prefill":   [(128, 128), (300, 300), (77, 77), (512, 512)],
    "decode":    [(1, 1024), (1, 37), (1, 2048), (1, 513), (1, 129), (1, 4000), (1, 16), (1, 777)],
    "chunked":   [(64, 700), (100, 100), (33, 1000), (128, 640)],
    "mixed":     [(256, 256), (1, 900), (1, 5), (50, 450), (1, 2000)],
}

# (H, Hkv, D, block_size)
MODEL_CFGS = [
    (8, 8, 64, 16),     # MHA
    (16, 4, 128, 16),   # GQA
    (8, 1, 64, 32),     # MQA
    (14, 2, 80, 16),    # odd head dim (D % 8 == 0, not pow2)
]

# Larger shapes only for the timing section
BENCH_SCENARIOS = {
    "prefill 8x1024":  [(1024, 1024)] * 8,
    "decode 64x2048":  [(1, 2048)] * 64,
    "decode 64x4096":  [(1, 4096)] * 64,   # sized for 6GB GPU
    "mixed":           [(512, 512)] * 2 + [(1, 3000)] * 32,
}
BENCH_CFG = (32, 8, 128, 16)  # H, Hkv, D, block_size


# ----------------------------------------------------------------------------
# Build a paged problem
# ----------------------------------------------------------------------------
def build_problem(seqs, H, Hkv, D, bs, causal=True):
    num_seqs = len(seqs)
    q_lens = [a for a, _ in seqs]
    kv_lens = [b for _, b in seqs]
    total_q = sum(q_lens)

    blocks_per_seq = [math.ceil(kv / bs) for kv in kv_lens]
    max_blocks = max(blocks_per_seq)
    used_blocks = sum(blocks_per_seq)
    num_blocks = used_blocks + 16  # spare blocks that stay as garbage

    # Garbage everywhere: if the kernel reads an unowned block / past kv_len, results blow up.
    k_pool = torch.full((num_blocks, bs, Hkv, D), 1e3, device=dev, dtype=DTYPE)
    v_pool = torch.full((num_blocks, bs, Hkv, D), 1e3, device=dev, dtype=DTYPE)

    # Shuffled physical block ids so tables are non-contiguous
    perm = torch.randperm(num_blocks, device=dev)
    block_tables = torch.zeros(num_seqs, max_blocks, device=dev, dtype=torch.int32)
    cursor = 0
    k_dense, v_dense = [], []
    for s in range(num_seqs):
        nb = blocks_per_seq[s]
        ids = perm[cursor:cursor + nb]
        cursor += nb
        block_tables[s, :nb] = ids.to(torch.int32)

        k = torch.randn(kv_lens[s], Hkv, D, device=dev, dtype=DTYPE)
        v = torch.randn(kv_lens[s], Hkv, D, device=dev, dtype=DTYPE)
        k_dense.append(k)
        v_dense.append(v)
        for b in range(nb):
            lo, hi = b * bs, min((b + 1) * bs, kv_lens[s])
            k_pool[ids[b], : hi - lo] = k[lo:hi]
            v_pool[ids[b], : hi - lo] = v[lo:hi]
            # tail of last block (hi-lo < bs) stays 1e3 garbage on purpose

    q = torch.randn(total_q, H, D, device=dev, dtype=DTYPE)

    qsl = [0]
    for ql in q_lens:
        qsl.append(qsl[-1] + ql)
    query_start_loc = torch.tensor(qsl, device=dev, dtype=torch.int32)

    pos = []
    for ql, kv in zip(q_lens, kv_lens):
        pos.append(torch.arange(kv - ql, kv, device=dev, dtype=torch.int32))
    positions = torch.cat(pos)

    return dict(
        q=q, k_pool=k_pool, v_pool=v_pool, block_tables=block_tables,
        query_start_loc=query_start_loc, positions=positions,
        q_lens=q_lens, kv_lens=kv_lens, k_dense=k_dense, v_dense=v_dense,
        H=H, Hkv=Hkv, D=D, bs=bs,
    )


# ----------------------------------------------------------------------------
# Call the kernel under test
# ----------------------------------------------------------------------------
def run_paged(p, causal=True):
    try:
        return paged_varlen_attention(
            p["q"], p["k_pool"], p["v_pool"],
            p["block_tables"], p["query_start_loc"], p["positions"],
            num_heads=p["H"],
            num_kv_heads=p["Hkv"],
            scale=None,
            non_causal=not causal,
        )
    except NotImplementedError as e:
        print(f"   SKIP ({str(e)[:70]}...)")
        return None


# ----------------------------------------------------------------------------
# References
# ----------------------------------------------------------------------------
def causal_mask(q_len, kv_len, device):
    # query i has absolute position kv_len - q_len + i ; key j visible iff j <= pos
    pos = torch.arange(kv_len - q_len, kv_len, device=device).unsqueeze(1)
    j = torch.arange(kv_len, device=device).unsqueeze(0)
    return j <= pos  # [q_len, kv_len] bool, True = keep


def gather_kv(p, s):
    """Gather dense K/V for sequence s out of the paged pool (what a non-paged baseline needs)."""
    bs, kv = p["bs"], p["kv_lens"][s]
    nb = math.ceil(kv / bs)
    ids = p["block_tables"][s, :nb].long()
    k = p["k_pool"][ids].reshape(nb * bs, p["Hkv"], p["D"])[:kv]
    v = p["v_pool"][ids].reshape(nb * bs, p["Hkv"], p["D"])[:kv]
    return k, v


def sdpa_one(q, k, v, H, Hkv, causal, dtype=None):
    """q: [Lq,H,D]  k,v: [Lkv,Hkv,D] -> [Lq,H,D]"""
    if dtype is not None:
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    g = H // Hkv
    qh = q.permute(1, 0, 2).unsqueeze(0)                                  # [1,H,Lq,D]
    kh = k.permute(1, 0, 2).repeat_interleave(g, dim=0).unsqueeze(0)      # [1,H,Lkv,D]
    vh = v.permute(1, 0, 2).repeat_interleave(g, dim=0).unsqueeze(0)
    mask = causal_mask(q.shape[0], k.shape[0], q.device) if causal else None
    o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
    return o.squeeze(0).permute(1, 0, 2).contiguous()


def run_ref(p, causal=True, dtype=None, from_pool=True):
    outs = []
    off = 0
    for s, ql in enumerate(p["q_lens"]):
        if from_pool:
            k, v = gather_kv(p, s)
        else:
            k, v = p["k_dense"][s], p["v_dense"][s]
        outs.append(sdpa_one(p["q"][off:off + ql], k, v, p["H"], p["Hkv"], causal, dtype))
        off += ql
    return torch.cat(outs)


# ----------------------------------------------------------------------------
# Accuracy
# ----------------------------------------------------------------------------
def metrics(out, ref):
    out, ref = out.float(), ref.float()
    diff = (out - ref).abs()
    cos = F.cosine_similarity(out.reshape(-1, out.shape[-1]), ref.reshape(-1, ref.shape[-1]), dim=-1).min().item()
    return diff.max().item(), diff.mean().item(), cos


def accuracy_suite():
    print("=" * 96)
    print("ACCURACY  (reference = SDPA in fp32 on dense K/V;  also shows fp16 SDPA error for scale)")
    print("=" * 96)
    hdr = f"{'scenario':<9}{'H/Hkv/D/bs':<14}{'causal':<8}{'paged maxerr':>13}{'paged mean':>12}{'min cos':>10}{'sdpa16 maxerr':>15}  status"
    print(hdr)
    print("-" * len(hdr))
    ok_all = True
    for (H, Hkv, D, bs) in MODEL_CFGS:
        for name, seqs in SCENARIOS.items():
            for causal in (True, False):
                p = build_problem(seqs, H, Hkv, D, bs, causal)
                out = run_paged(p, causal)
                if out is None:
                    continue
                torch.cuda.synchronize()
                ref32 = run_ref(p, causal, dtype=torch.float32, from_pool=False)
                ref16 = run_ref(p, causal, dtype=None, from_pool=True)

                finite = torch.isfinite(out).all().item()
                mx, mean, cos = metrics(out, ref32)
                mx16, _, _ = metrics(ref16, ref32)

                # pass if finite, and error within a small multiple of fp16 SDPA's own error
                tol_max = max(2e-2, 4 * mx16)
                ok = finite and mx <= tol_max and cos > 0.999
                ok_all &= ok
                print(f"{name:<9}{f'{H}/{Hkv}/{D}/{bs}':<14}{str(causal):<8}"
                      f"{mx:>13.2e}{mean:>12.2e}{cos:>10.5f}{mx16:>15.2e}  {'OK' if ok else 'FAIL'}"
                      f"{'' if finite else '  (NaN/Inf!)'}")
                if not ok:
                    bad = (out.float() - ref32).abs().amax(dim=(1, 2))
                    idx = bad.argmax().item()
                    # which sequence does worst token belong to
                    qsl = p["query_start_loc"].tolist()
                    sq = next(i for i in range(len(qsl) - 1) if qsl[i] <= idx < qsl[i + 1])
                    print(f"          worst token {idx} (seq {sq}, q_len={p['q_lens'][sq]}, kv_len={p['kv_lens'][sq]}) err={bad[idx].item():.3e}")
    return ok_all


# ----------------------------------------------------------------------------
# Speed
# ----------------------------------------------------------------------------
def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    return times[len(times) // 2]  # median ms


def attn_flops(p, causal):
    H, D = p["H"], p["D"]
    total = 0
    for ql, kv in zip(p["q_lens"], p["kv_lens"]):
        if causal:
            # query i sees (kv - ql + i + 1) keys
            visible = ql * (kv - ql) + ql * (ql + 1) // 2
        else:
            visible = ql * kv
        total += visible
    return 4.0 * total * H * D  # QK^T + PV


def kv_bytes(p):
    return sum(p["kv_lens"]) * p["Hkv"] * p["D"] * 2 * 2  # K and V, fp16


def speed_suite():
    print()
    print("=" * 96)
    print("SPEED  (median ms; causal=True)  H/Hkv/D/bs = %s" % (BENCH_CFG,))
    print("  sdpa-loop    : per-seq loop, gathers K/V from pool each call (apples-to-apples with paged)")
    print("  sdpa-pregath : per-seq loop, K/V already dense (best case for SDPA, unrealistic in serving)")
    print("=" * 96)
    H, Hkv, D, bs = BENCH_CFG
    hdr = f"{'scenario':<18}{'paged':>9}{'sdpa-loop':>11}{'sdpa-pregath':>14}{'speedup':>9}{'TFLOPs':>9}{'GB/s':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, seqs in BENCH_SCENARIOS.items():
        p = build_problem(seqs, H, Hkv, D, bs)
        # pre-gather once for the "best case" baseline
        dense = list(zip(p["k_dense"], p["v_dense"]))
        offs = p["query_start_loc"].tolist()

        def f_paged():
            return run_paged(p, True)

        def f_loop():
            return run_ref(p, True, from_pool=True)

        def f_pre():
            outs = []
            for s, (k, v) in enumerate(dense):
                outs.append(sdpa_one(p["q"][offs[s]:offs[s + 1]], k, v, H, Hkv, True))
            return torch.cat(outs)

        t_paged = bench(f_paged)
        t_loop = bench(f_loop)
        t_pre = bench(f_pre)
        tflops = attn_flops(p, True) / (t_paged * 1e-3) / 1e12
        gbs = kv_bytes(p) / (t_paged * 1e-3) / 1e9
        print(f"{name:<18}{t_paged:>9.3f}{t_loop:>11.3f}{t_pre:>14.3f}{t_loop / t_paged:>8.1f}x{tflops:>9.1f}{gbs:>9.0f}")
        del p, dense
        torch.cuda.empty_cache()
    print("\n(TFLOPs/GB/s are for the paged kernel. Decode is bandwidth-bound: compare GB/s to your GPU's peak.)")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    print(f"GPU: {torch.cuda.get_device_name()}  torch {torch.__version__}\n")
    ok = accuracy_suite()
    speed_suite()
    print("\nRESULT:", "ALL ACCURACY CHECKS PASSED" if ok else "SOME ACCURACY CHECKS FAILED")
    sys.exit(0 if ok else 1)