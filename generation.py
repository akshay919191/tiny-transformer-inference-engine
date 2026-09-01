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