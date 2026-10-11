"""Measure inference prefill and single-token decode latency for a checkpoint."""

import argparse
import statistics
import time

import torch

from inference import load_model
from kv_cache import KVCache_kv
from models.model_config import ModelConfig


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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

    prefill_ms, decode_ms = [], []
    for _ in range(iterations):
        cache = fresh_cache()
        sync(device)
        start = time.perf_counter()
        logits = model(tokens, kv_cache=cache)[:, -1, :]
        sync(device)
        prefill_ms.append((time.perf_counter() - start) * 1000)

        token = logits.argmax(-1, keepdim=True)
        step_times = []
        for _ in range(decode_steps):
            sync(device)
            start = time.perf_counter()
            logits = model(token, kv_cache=cache)[:, -1, :]
            sync(device)
            step_times.append((time.perf_counter() - start) * 1000)
            token = logits.argmax(-1, keepdim=True)
        decode_ms.extend(step_times)

    return statistics.median(prefill_ms), statistics.median(decode_ms)


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
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if min(args.batch, args.prompt_len, args.iterations) <= 0 or min(args.decode_steps, args.warmup) < 0:
        parser.error("batch, prompt length and iterations must be positive; other counts nonnegative")
    model, cfg = load_model(args.ckpt, device, backend=args.backend)
    if args.prompt_len + args.decode_steps > cfg.max_seq_len:
        parser.error("prompt length plus decode steps exceeds model max_seq_len")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    prefill, decode = measure(model, cfg, device, args.batch, args.prompt_len,
                              args.decode_steps, args.warmup, args.iterations)
    print(f"device={device} backend={args.backend} batch={args.batch} prompt={args.prompt_len}")
    print(f"prefill median: {prefill:.3f} ms ({args.batch * args.prompt_len / (prefill / 1000):.1f} tokens/s)")
    if args.decode_steps:
        print(f"decode median:  {decode:.3f} ms/token ({1000 / decode:.1f} tokens/s)")
    if device.type == "cuda":
        print(f"peak allocated: {torch.cuda.max_memory_allocated(device) / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()
