# benchmarks/bench.py
import argparse
import statistics
import sys
from pathlib import Path
import torch

from models.transformer_block import Transformer
from models.model_config import ModelConfig
from kv_cache import KVCache_kv  
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_model(cfg, state_dict, backend, attn_type, device):
    model = Transformer(cfg, attn_type=attn_type, backend=backend).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def make_cache_fn(model, cfg, batch_size, device):
    """Factory that returns a FRESH cache each call. ADJUST kwargs to your KVCache signature."""
    head_dim = cfg.d_model // cfg.num_heads

    def make():
        return KVCache_kv(
            num_layers=cfg.num_layers,          # <-- was missing
            batch_size=batch_size,
            max_seq_len=cfg.max_seq_len,
            num_heads=cfg.num_kv_heads,
            head_dim=head_dim,
            device=device,
            dtype=next(model.parameters()).dtype,
        )
    return make   



@torch.no_grad()
def prefill(model, tokens, kv_cache):
    logits = model(tokens, kv_cache=kv_cache)
    return logits[:, -1, :]


@torch.no_grad()
def decode_one(model, next_token, kv_cache):
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
    # warmup with short decode runs
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
    for s, e in events:
        s.record()
        logits = decode_one(model, tok, cache)
        e.record()
        tok = logits.argmax(-1, keepdim=True)  # greedy
    torch.cuda.synchronize()  # one sync at the end, events measure each step
    return [s.elapsed_time(e) for s, e in events]




def fmt_stats(ms):
    return (f"p50 {statistics.median(ms):8.3f} ms | "
            f"mean {statistics.mean(ms):8.3f} ms | "
            f"min {min(ms):8.3f} ms")


def run_one(label, model, cfg, args, device):
    assert args.prompt_len + args.num_tokens <= cfg.max_seq_len, \
        "prompt + generated tokens must fit in max_seq_len"

    tokens = torch.randint(0, cfg.vocab_size, (args.batch, args.prompt_len), device=device)
    make_cache = make_cache_fn(model, cfg, args.batch, device)

    print(f"\n--- {label} ---")

    prefill_ms = bench_prefill(model, tokens, make_cache, args.warmup, args.iters)
    ttft = statistics.median(prefill_ms)
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
    print(f"  Per-token latency  : {per_tok:.3f} ms")
    print(f"  Decode throughput  : {1000 / per_tok:.1f} tok/s")
    print(f"  Peak VRAM          : {peak:.0f} MB")



def main():
    p = argparse.ArgumentParser(description="Prefill / decode benchmark")
    p.add_argument("--ckpt", default="checkpoints/ckpt_final.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--prompt_len", type=int, default=100)
    p.add_argument("--num_tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--compare", action="store_true",
                   help="also benchmark the pytorch backend with the same weights")
    args = p.parse_args()

    assert args.device == "cuda" and torch.cuda.is_available(), \
        "CUDA events need a GPU"

    device = args.device
    ckpt = torch.load(args.ckpt, map_location=device)

    cfg = ModelConfig()
    for k, v in ckpt["model_config"].items():
        setattr(cfg, k, v)

    tcfg = ckpt.get("train_config", {})
    attn_type = tcfg.get("attn_type", "mqa")

    import statistics



    params = sum(p.numel() for p in Transformer(cfg, backend="pytorch").parameters())
    print("Tiny Transformer Inference Benchmark")
    print(f"  params {params/1e6:.1f}M | attn {attn_type} | "
          f"d_model {cfg.d_model} | layers {cfg.num_layers} | "
          f"batch {args.batch} | prompt {args.prompt_len}")

    backend = tcfg.get("backend", "pytorch")
    model = build_model(cfg, ckpt["model"], backend, attn_type, device)
    run_one(f"backend={backend} attn={attn_type}", model, cfg, args, device)

    def decode_latency_at(prompt_len, steps=20):
        # ✅ Change 1 to args.batch so Q matches K/V batch sizes
        tokens = torch.randint(0, cfg.vocab_size, (args.batch, prompt_len), device=device)
        
        # ✅ Keep your factory execution fix
        cache = make_cache_fn(model, cfg, args.batch, device)()
        
        tok = prefill(model, tokens, cache).argmax(-1, keepdim=True)
        times = []
        for _ in range(steps):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            logits = decode_one(model, tok, cache)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
            tok = logits.argmax(-1, keepdim=True)
        return statistics.median(times)


    short = decode_latency_at(8)
    long_ = decode_latency_at(480)
    print(f"decode @ cache~10 : {short:.3f} ms")
    print(f"decode @ cache~490: {long_:.3f} ms")
    print(f"ratio: {long_/short:.2f}x")

    from torch.profiler import profile, ProfilerActivity

    # build a long cache first
    tokens = torch.randint(0, cfg.vocab_size, (32, 480), device=device)
    cache = make_cache_fn(model, cfg, args.batch, device)()
    tok = prefill(model, tokens, cache).argmax(-1, keepdim=True)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            decode_one(model, tok, cache)
            tok = decode_one(model, tok, cache).argmax(-1, keepdim=True)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    if args.compare:
        other = "pytorch" if backend == "cuda" else "cuda"
        try:
            model2 = build_model(cfg, ckpt["model"], other, attn_type, device)
            run_one(f"backend={other} attn={attn_type}", model2, cfg, args, device)
        except Exception as ex:
            print(f"\nskipping backend={other}: {ex}")


if __name__ == "__main__":
    main()
    