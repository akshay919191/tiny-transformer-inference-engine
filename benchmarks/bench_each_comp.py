# benchmarks/bench_each_comp.py
import argparse
import os
import statistics
import sys
from pathlib import Path
import torch
import torch._dynamo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.transformer_block import Transformer
from models.model_config import ModelConfig
from kv_cache import KVCache_kv

# Allow more cached graphs. Name differs across torch versions.
for _name in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _name):
        setattr(torch._dynamo.config, _name, 64)

torch.set_float32_matmul_precision("high")


def _step_begin():
    """Tell CUDA-graph trees a new step starts (safe no-op if unused/unavailable)."""
    fn = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if fn is not None:
        fn()


def build_model(cfg, state_dict, backend, attn_type, device, compile_model=False,
                compile_mode="reduce-overhead", dtype=torch.float32):
    model = Transformer(cfg, attn_type=attn_type, backend=backend).to(device)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model = model.to(dtype=dtype)  # cast AFTER loading fp32 weights; the KV cache follows this dtype
    model.eval()

    if compile_model:
        print(f"[INFO] Compiling model with torch.compile(mode='{compile_mode}')...")
        # The write position now lives in a GPU tensor inside the cache, so no
        # Python int changes between decode steps. Only two shapes ever occur:
        # prefill [B, prompt_len] and decode [B, 1]. dynamic=False gives each its
        # own fully specialised graph instead of a symbolic-seq-len one.
        model = torch.compile(model, mode=compile_mode, dynamic=False)

    return model


def pick_cache_len(cfg, args):
    """Cache size. Attention scans the WHOLE buffer every step, so keep it tight."""
    need = args.prompt_len + args.num_tokens
    assert need <= cfg.max_seq_len, (
        f"prompt+tokens ({need}) exceeds cfg.max_seq_len ({cfg.max_seq_len}); "
        "RoPE tables are only that long"
    )
    if args.cache_len:
        n = args.cache_len
    else:
        n = ((need + 63) // 64) * 64          # round up to a multiple of 64
        n = min(n, cfg.max_seq_len)
    assert n >= need, f"--cache_len {n} < prompt+tokens {need}"
    return n


def make_cache_fn(model, cfg, batch_size, device, cache_len):
    """
    Allocate ONE cache and hand it back (reset) on every call.

    Do not allocate a new cache per iteration: the buffers are marked static
    for CUDA graphs, so a new object means new addresses and a re-recorded
    graph inside the timed region. reset() just zeroes `pos`.
    """
    head_dim = cfg.d_model // cfg.num_heads
    cache = KVCache_kv(
        num_layers=cfg.num_layers,
        batch_size=batch_size,
        max_seq_len=cache_len,
        num_heads=cfg.num_kv_heads,
        head_dim=head_dim,
        device=device,
        dtype=next(model.parameters()).dtype,
    )

    def make():
        cache.reset()
        return cache

    return make


@torch.no_grad()
def check_correctness(model, cfg, make_cache, batch, prompt_len, device, steps=16):
    """
    Compare the (compiled) cached path against an independent oracle:
    the eager model with NO cache, recomputing the full sequence every step.
    Both are fed the same tokens (taken from the oracle) so errors don't diverge.
    """
    ref_model = getattr(model, "_orig_mod", model)  # eager, uncompiled
    cache = make_cache()
    seq = torch.randint(0, cfg.vocab_size, (batch, prompt_len), device=device)

    _step_begin()
    got = model(seq, kv_cache=cache)[:, -1, :].float().clone()

    worst_rel, agree, total = 0.0, 0, 0
    for i in range(steps + 1):
        ref = ref_model(seq)[:, -1, :].float()
        scale = ref.abs().max().item()
        rel = (got - ref).abs().max().item() / max(scale, 1e-6)
        worst_rel = max(worst_rel, rel)
        nxt = ref.argmax(-1, keepdim=True)
        agree += (got.argmax(-1, keepdim=True) == nxt).sum().item()
        total += nxt.numel()
        if i == steps:
            break
        seq = torch.cat([seq, nxt], dim=1)
        _step_begin()
        got = model(nxt, kv_cache=cache)[:, -1, :].float().clone()

    ok = worst_rel < 0.05
    print(f"\n[CHECK] cached vs no-cache oracle over {steps + 1} positions: "
          f"worst rel logit diff {worst_rel:.4f} | top-1 agreement {agree}/{total} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("[CHECK] Cached output does NOT match the oracle. Do not trust the timings below.")
    torch.cuda.synchronize()
    return ok


@torch.no_grad()  # warmup must run under no_grad, same as the timed runs
def warmup_compiled_graphs(model, cfg, make_cache, batch_size, prompt_len, device, rounds=3, decode_steps=6):
    """Compile prefill [B, T] and decode [B, 1] on the SAME cache object used by the benchmark."""
    for _ in range(rounds):
        cache = make_cache()
        tokens = torch.randint(0, cfg.vocab_size, (batch_size, prompt_len), device=device)

        _step_begin()
        logits = model(tokens, kv_cache=cache)
        tok = logits[:, -1, :].argmax(-1, keepdim=True)

        for _ in range(decode_steps):
            _step_begin()
            logits = model(tok, kv_cache=cache)
            tok = logits[:, -1, :].argmax(-1, keepdim=True)

    torch.cuda.synchronize()


@torch.no_grad()
def prefill(model, tokens, kv_cache):
    _step_begin()
    logits = model(tokens, kv_cache=kv_cache)
    return logits[:, -1, :]


@torch.no_grad()
def decode_one(model, next_token, kv_cache):
    _step_begin()
    logits = model(next_token, kv_cache=kv_cache)
    return logits[:, -1, :]


def bench_prefill(model, tokens, make_cache, warmup=10, iters=50):
    for _ in range(warmup):
        prefill(model, tokens, make_cache())
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        cache = make_cache()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        prefill(model, tokens, cache)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def bench_decode(model, prompt_tokens, make_cache, num_tokens=128, warmup=10):
    # Warmup
    for _ in range(warmup):
        cache = make_cache()
        tok = prefill(model, prompt_tokens, cache).argmax(-1, keepdim=True)
        for _ in range(10):
            tok = decode_one(model, tok, cache).argmax(-1, keepdim=True)
    torch.cuda.synchronize()

    cache = make_cache()
    tok = prefill(model, prompt_tokens, cache).argmax(-1, keepdim=True)

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
              for _ in range(num_tokens)]

    # NVTX Range Push to isolate GPU kernels during decode phase
    torch.cuda.nvtx.range_push("decode_phase")
    for s, e in events:
        s.record()
        logits = decode_one(model, tok, cache)
        e.record()
        tok = logits.argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    return [s.elapsed_time(e) for s, e in events]


def fmt_stats(ms):
    return (f"p50 {statistics.median(ms):8.3f} ms | "
            f"mean {statistics.mean(ms):8.3f} ms | "
            f"min {min(ms):8.3f} ms | "
            f"max {max(ms):8.3f} ms")


def run_one(label, model, cfg, args, device, make_cache):
    tokens = torch.randint(0, cfg.vocab_size, (args.batch, args.prompt_len), device=device)

    prefill_ms = bench_prefill(model, tokens, make_cache, args.warmup, args.iters)
    ttft = statistics.median(prefill_ms)
    print(f"\n--- Benchmark Results [{label}] ---")
    print(f"Prefill  ({args.batch}x{args.prompt_len})  {fmt_stats(prefill_ms)}")
    print(f"  TTFT               : {ttft:.3f} ms")
    print(f"  Prefill throughput : {args.batch * args.prompt_len / (ttft / 1000):.1f} tok/s")

    torch.cuda.reset_peak_memory_stats(device)
    decode_ms = bench_decode(model, tokens, make_cache, args.num_tokens, args.warmup)
    for i in range(0, len(decode_ms), 50):
        print(f"step {i:3d}: {decode_ms[i]:.3f} ms")
    per_tok = statistics.median(decode_ms)
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"Decode   ({args.num_tokens} steps)   {fmt_stats(decode_ms)}")
    print(f"  Per-token latency  : {per_tok:.3f} ms  (one step, whole batch)")
    print(f"  Decode throughput  : {args.batch * 1000 / per_tok:.1f} tok/s  (batch {args.batch})")
    print(f"  Peak VRAM          : {peak:.0f} MB")


def main():
    p = argparse.ArgumentParser(description="Prefill / decode benchmark")
    p.add_argument("--ckpt", default="checkpoints/ckpt_final.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prompt_len", type=int, default=100)
    p.add_argument("--num_tokens", type=int, default=128)
    p.add_argument("--cache_len", type=int, default=0,
                   help="KV cache length. Default: prompt+tokens rounded up to a multiple of 64. "
                        "Try 1024 to see how much decode time scales with buffer size.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--compile", action="store_true", help="Enable torch.compile compilation")
    p.add_argument("--compile_mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--compare", action="store_true", help="also benchmark uncompiled eager mode vs compiled mode")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                   help="Run the model (and KV cache) in this dtype")
    p.add_argument("--skip_check", action="store_true", help="skip the correctness check against the no-cache oracle")
    args = p.parse_args()

    assert args.device == "cuda" and torch.cuda.is_available(), "CUDA events need a GPU"

    device = args.device
    cfg = ModelConfig()
    model_state = None

    if os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location=device)
        for k, v in ckpt.get("model_config", {}).items():
            setattr(cfg, k, v)
        model_state = ckpt.get("model", None)
        tcfg = ckpt.get("train_config", {})
        attn_type = tcfg.get("attn_type", "mqa")
        backend = tcfg.get("backend", "pytorch")
    else:
        print(f"[WARNING] Checkpoint '{args.ckpt}' not found. Initializing dummy model parameters for profiling.")
        attn_type = "mqa"
        backend = "pytorch"

    cache_len = pick_cache_len(cfg, args)

    params = sum(p.numel() for p in Transformer(cfg, backend="pytorch").parameters())
    print("Tiny Transformer Inference Benchmark")
    print(f"  params {params/1e6:.1f}M | attn {attn_type} | "
          f"d_model {cfg.d_model} | layers {cfg.num_layers} | "
          f"batch {args.batch} | prompt {args.prompt_len}")
    print(f"  cache_len {cache_len} (cfg.max_seq_len {cfg.max_seq_len})")

    def bench_model(compile_model):
        model = build_model(
            cfg, model_state, backend, attn_type, device,
            compile_model=compile_model, compile_mode=args.compile_mode,
            dtype=getattr(torch, args.dtype),
        )
        print(f"  model dtype {next(model.parameters()).dtype} | compiled={compile_model}")
        make_cache = make_cache_fn(model, cfg, args.batch, device, cache_len)

        if compile_model:
            warmup_compiled_graphs(model, cfg, make_cache, args.batch, args.prompt_len, device)
        if not args.skip_check:
            check_correctness(model, cfg, make_cache, args.batch, args.prompt_len, device)

        label = f"backend={backend} attn={attn_type} compiled={compile_model}"
        run_one(label, model, cfg, args, device, make_cache)

    # Primary Run
    bench_model(args.compile)

    # Secondary Comparative Run (Eager vs Compiled)
    if args.compare:
        print("\n" + "=" * 50)
        print(f"Running comparative benchmark (compiled= {not args.compile})...")
        print("=" * 50)
        bench_model(not args.compile)


if __name__ == "__main__":
    main()