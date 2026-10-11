"""Measure prefill, first-token, decode, and end-to-end inference latency."""

import argparse
import math
import statistics
import time

import torch

from inference import load_model
from kv_cache import KVCache_kv


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values, quantile):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


@torch.inference_mode()
def measure(model, cfg, device, batch, prompt_len, decode_steps, warmup, iterations):
    dtype = next(model.parameters()).dtype
    tokens = torch.randint(0, cfg.vocab_size, (batch, prompt_len), device=device)

    def fresh_cache():
        return KVCache_kv(cfg.num_layers, batch, cfg.num_kv_heads,
                          cfg.max_seq_len, cfg.head_dim, dtype, device)

    for _ in range(warmup):
        cache = fresh_cache()
        logits = model(tokens, kv_cache=cache)[:, -1, :]
        token = logits.argmax(-1, keepdim=True)
        for _ in range(decode_steps):
            logits = model(token, kv_cache=cache)[:, -1, :]
            token = logits.argmax(-1, keepdim=True)
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    ttft_ms, decode_ms, total_ms = [], [], []
    for _ in range(iterations):
        cache = fresh_cache()
        sync(device)
        request_start = time.perf_counter()
        logits = model(tokens, kv_cache=cache)[:, -1, :]
        sync(device)
        ttft_ms.append((time.perf_counter() - request_start) * 1000)

        token = logits.argmax(-1, keepdim=True)
        for _ in range(decode_steps):
            sync(device)
            start = time.perf_counter()
            logits = model(token, kv_cache=cache)[:, -1, :]
            sync(device)
            decode_ms.append((time.perf_counter() - start) * 1000)
            token = logits.argmax(-1, keepdim=True)
        sync(device)
        total_ms.append((time.perf_counter() - request_start) * 1000)

    ttft_p50 = statistics.median(ttft_ms)
    decode_p50 = statistics.median(decode_ms) if decode_ms else 0.0
    total_p50 = statistics.median(total_ms)
    return {
        "ttft_p50": ttft_p50,
        "ttft_p90": percentile(ttft_ms, 0.90),
        "decode_p50": decode_p50,
        "decode_p90": percentile(decode_ms, 0.90) if decode_ms else 0.0,
        "total_p50": total_p50,
        "prefill_tps": batch * prompt_len / (ttft_p50 / 1000),
        "decode_tps": 1000 / decode_p50 if decode_p50 else 0.0,
        "generated_tps": batch * decode_steps / (total_p50 / 1000),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", choices=("auto", "cuda", "pytorch"), default="auto")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--sweep", action="store_true",
                        help="run the combinations in --batches and --prompt-lengths")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[16, 64, 128])
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if min(args.batch, args.prompt_len, args.iterations) <= 0 or min(args.decode_steps, args.warmup) < 0:
        parser.error("batch, prompt length and iterations must be positive; other counts nonnegative")
    if args.sweep and (min(args.batches) <= 0 or min(args.prompt_lengths) <= 0):
        parser.error("sweep batch sizes and prompt lengths must be positive")

    model, cfg = load_model(args.ckpt, device, backend=args.backend)
    backend = getattr(model.layers[0].attn, "backend", args.backend)
    workloads = ([(batch, length) for batch in args.batches for length in args.prompt_lengths]
                 if args.sweep else [(args.batch, args.prompt_len)])
    print(f"device={device} backend={backend} dtype={next(model.parameters()).dtype} "
          f"decode_steps={args.decode_steps} warmup={args.warmup} iterations={args.iterations}")
    print("batch prompt  TTFT p50/p90 ms  prefill tok/s  decode p50/p90 ms  decode tok/s  total p50 ms  generated tok/s")

    for batch, prompt_len in workloads:
        if prompt_len + args.decode_steps > cfg.max_seq_len:
            if args.sweep:
                print(f"{batch:>5} {prompt_len:>6}  skipped: exceeds max_seq_len={cfg.max_seq_len}")
                continue
            parser.error("prompt length plus decode steps exceeds model max_seq_len")
        result = measure(model, cfg, device, batch, prompt_len,
                         args.decode_steps, args.warmup, args.iterations)
        print(f"{batch:>5} {prompt_len:>6}  "
              f"{result['ttft_p50']:>6.3f}/{result['ttft_p90']:<6.3f}  "
              f"{result['prefill_tps']:>13.1f}  "
              f"{result['decode_p50']:>6.3f}/{result['decode_p90']:<6.3f}  "
              f"{result['decode_tps']:>12.1f}  {result['total_p50']:>11.3f}  "
              f"{result['generated_tps']:>15.1f}")
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 1024**2
            print(f"      peak CUDA allocation: {peak:.1f} MiB")


if __name__ == "__main__":
    main()
