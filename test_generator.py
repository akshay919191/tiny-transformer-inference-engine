import torch
import time
import torch.nn as nn
import torch.nn.functional as F

from sampling import sample
from models.transformer_block import make_kv_cache, make_kv_cache_


def _build_kv_cache(
    attn_type,
    model,
    config,
    batch_size,
    max_seq_len,
    device,
):
    assert attn_type in ("mha", "mqa"), \
        f"attn_type must be 'mha' or 'mqa', got {attn_type!r}"

    cache_fn = make_kv_cache if attn_type == "mha" else make_kv_cache_

    return cache_fn(
        model,
        config,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        device=device,
    )


@torch.no_grad()
def generate(
    model,
    prompt_tokens,
    max_new_tokens,
    config,
    attn_type="mqa",
):

    model.eval()
    tokens = prompt_tokens

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    kv_cache = _build_kv_cache(
        attn_type,
        model,
        config,
        batch_size=prompt_tokens.shape[0],
        max_seq_len=(
            prompt_tokens.shape[1]
            + max_new_tokens
        ),
        device=prompt_tokens.device,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    logits = model(
        tokens,
        kv_cache=kv_cache,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    ttft_end = time.perf_counter()

    for step in range(max_new_tokens):
        next_token_logits = logits[:, -1, :]
        next_token = sample(
            next_token_logits,
            temperature=0.8,
            top_k=30,
            top_p=0.9,
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1,
        )

        new_token = tokens[:, -1:]
        logits = model(
            new_token,
            kv_cache=kv_cache,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    ttft_ms = (
        ttft_end - start
    ) * 1000

    total_ms = (
        end - start
    ) * 1000

    decode_ms = (
        total_ms - ttft_ms
    )

    tokens_per_sec = (
        max_new_tokens
        / (total_ms / 1000)
    )

    if torch.cuda.is_available():
        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )
    else:
        peak_memory_mb = 0.0

    print("KV CACHE")
    print(f"attn type:     {attn_type}")
    print(f"prompt length: {prompt_tokens.shape[1]}")
    print(f"generate:      {max_new_tokens} tokens")
    print(f"TTFT:          {ttft_ms:.3f} ms")
    print(f"decode time:   {decode_ms:.3f} ms")
    print(f"total:         {total_ms:.3f} ms")
    print(f"tokens/sec:    {tokens_per_sec:.2f}")
    print(f"peak memory:   {peak_memory_mb:.2f} MB")
    print(f"final length:  {tokens.shape[1]}")

    return tokens


@torch.no_grad()
def generate_nocache(
    model,
    prompt_tokens,
    max_new_tokens,
):

    model.eval()
    tokens = prompt_tokens

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()

    for step in range(max_new_tokens):
        logits = model(tokens)
        next_token_logits = logits[:, -1, :]
        next_token = sample(
            next_token_logits,
            temperature=0.8,
            top_k=30,
            top_p=0.9,
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_ms = (
        end - start
    ) * 1000

    tokens_per_sec = (
        max_new_tokens
        / (total_ms / 1000)
    )

    if torch.cuda.is_available():
        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )
    else:
        peak_memory_mb = 0.0

    print("NO KV CACHE")
    print(f"prompt length: {prompt_tokens.shape[1]}")
    print(f"generate:      {max_new_tokens} tokens")
    print(f"total:         {total_ms:.3f} ms")
    print(f"tokens/sec:    {tokens_per_sec:.2f}")
    print(f"peak memory:   {peak_memory_mb:.2f} MB")
    print(f"final length:  {tokens.shape[1]}")

    return tokens

@torch.no_grad()
def test_first_decode(
    model_nocache,
    model_cache,
    prompt_tokens,
    config,
    attn_type="mqa",
):
    
    model_nocache.eval()
    model_cache.eval()

    logits_nc = model_nocache(
        prompt_tokens
    )

    cache = _build_kv_cache(
        attn_type,
        model_cache,
        config,
        batch_size=prompt_tokens.shape[0],
        max_seq_len=prompt_tokens.shape[1] + 1,
        device=prompt_tokens.device,
    )

    logits_c = model_cache(
        prompt_tokens,
        kv_cache=cache,
    )

    prefill_diff = (
        logits_nc - logits_c
    ).abs().max()

    print(
        "Prefill max diff:",
        prefill_diff.item()
    )


    next_token = torch.argmax(
        logits_nc[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    tokens = torch.cat(
        [
            prompt_tokens,
            next_token,
        ],
        dim=1,
    )


    logits_nc = model_nocache(
        tokens
    )

    logits_c = model_cache(
        next_token,
        kv_cache=cache,
    )

    diff = (
        logits_nc[:, -1, :]
        -
        logits_c[:, -1, :]
    ).abs().max()

    print(
        "First decode max diff:",
        diff.item()
    )

    print(
        "No-cache next:",
        torch.argmax(
            logits_nc[:, -1, :],
            dim=-1,
        )
    )

    print(
        "KV-cache next:",
        torch.argmax(
            logits_c[:, -1, :],
            dim=-1,
        )
    )


import argparse
import contextlib
import io


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(fn, *args, **kwargs):
    _sync()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _sync()
    return out, (time.perf_counter() - t0) * 1e3


def main():
    from models.model_config import ModelConfig
    from models.transformer_block import Transformer

    p = argparse.ArgumentParser(description="KV-cache generation demo and correctness check")
    p.add_argument("--ckpt", default="checkpoints/ckpt_final.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt_len", type=int, default=32, help="random prompt length")
    p.add_argument("--prompt_ids", default=None, help="comma-separated token ids (overrides random prompt)")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--attn_type", default=None, help="override checkpoint (mha / mqa)")
    p.add_argument("--backend", default=None, help="override checkpoint (cuda / pytorch)")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_check", action="store_true", help="skip the cache vs no-cache check")
    p.add_argument("--skip_nocache", action="store_true", help="skip the no-cache generation")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    dtype = getattr(torch, args.dtype)

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ModelConfig()
    for k, v in ckpt["model_config"].items():
        setattr(cfg, k, v)

    tcfg = ckpt.get("train_config", {})
    attn_type = args.attn_type or tcfg.get("attn_type", "mqa")
    backend = args.backend or tcfg.get("backend", "pytorch")

    model = Transformer(cfg, attn_type=attn_type, backend=backend).to(device=device, dtype=dtype)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()})
    model.eval()

    if args.prompt_ids:
        ids = [int(t) for t in args.prompt_ids.split(",")]
        prompt = torch.tensor([ids] * args.batch, dtype=torch.long, device=device)
    else:
        prompt = torch.randint(0, cfg.vocab_size, (args.batch, args.prompt_len), device=device)

    assert prompt.shape[1] + args.max_new_tokens <= cfg.max_seq_len, (
        f"prompt ({prompt.shape[1]}) + new tokens ({args.max_new_tokens}) "
        f"must fit in max_seq_len ({cfg.max_seq_len})"
    )

    n_params = sum(t.numel() for t in model.parameters())
    print(f"model: {n_params / 1e6:.1f}M params | attn {attn_type} | backend {backend} | "
          f"dtype {args.dtype} | batch {args.batch} | prompt {prompt.shape[1]} | "
          f"new tokens {args.max_new_tokens}")

    for _ in range(args.warmup):
        with contextlib.redirect_stdout(io.StringIO()):
            generate(model, prompt, 4, cfg, attn_type)

    if not args.skip_check:
        print("\n=== correctness: KV cache vs no cache ===")
        test_first_decode(model, model, prompt, cfg, attn_type)

    # ---- generation with KV cache ----
    print("\n=== generate (KV cache) ===")
    out_c, ms_c = _timed(generate, model, prompt, args.max_new_tokens, cfg, attn_type)
    print(f"generated ids [seq 0]: {out_c[0, prompt.shape[1]:].tolist()}")

    if not args.skip_nocache:
        print("\n=== generate (no cache) ===")
        out_n, ms_n = _timed(generate_nocache, model, prompt, args.max_new_tokens)
        print(f"generated ids [seq 0]: {out_n[0, prompt.shape[1]:].tolist()}")
        print(f"\nwall clock: cache {ms_c:.1f} ms | no cache {ms_n:.1f} ms | "
              f"speedup x{ms_n / ms_c:.2f}")


if __name__ == "__main__":
    main()