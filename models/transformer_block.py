import torch
import time
import torch.nn as nn

from mqa import MQA , MQA_Cached
from embedding import TokenEmbedding
from mlp import SwiGLU
from model_config import ModelConfig
from rmsnorm import RMSNorm
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kv_cache import KVCache_kv


class TransformerBlock_Nocache(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.attn_norm = RMSNorm(config.d_model)

        self.attn = MQA(
            config
        )

        self.mlp_norm = RMSNorm(config.d_model)

        self.mlp = SwiGLU(
            config.d_model,
            config.hidden_size,
            config.bias,
        )

    def forward(self, x):

        residual = x

        x = self.attn_norm(x)

        x = self.attn(
            x,
            x,
            x,
            causal=True,
        )

        x = x + residual

        residual = x

        x = self.mlp_norm(x)

        x = self.mlp(x)

        x = x + residual

        return x


class Transformer_nocache(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.embedding = TokenEmbedding(
            config.vocab_size,
            config.d_model,
        )

        self.layers = nn.ModuleList([
            TransformerBlock_Nocache(config)
            for _ in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        for layer in self.layers:

            x = layer(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


class TransformerBlock(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()

        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(config.d_model)

        self.attn = MQA_Cached(
            config
        )

        self.mlp_norm = RMSNorm(config.d_model)

        self.mlp = SwiGLU(
            config.d_model,
            config.hidden_size,
            config.bias,
        )

    def forward(self, x, kv_cache):

        residual = x

        x = self.attn_norm(x)

        x = self.attn(
            x,
            x,
            x,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
            causal=True,
        )

        x = x + residual

        residual = x

        x = self.mlp_norm(x)

        x = self.mlp(x)

        x = x + residual

        return x


class Transformer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.embedding = TokenEmbedding(
            config.vocab_size,
            config.d_model,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                config,
                layer_idx=i,
            )
            for i in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def forward(self, input_ids, kv_cache=None):

        x = self.embedding(input_ids)

        for layer in self.layers:

            x = layer(
                x,
                kv_cache=kv_cache,
            )
        if kv_cache is not None:
            kv_cache.advance(input_ids.shape[1])

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


# def make_kv_cache(model, config, batch_size, max_seq_len, device):

#     head_dim = config.d_model // config.num_heads

#     return KVCache(
#         num_layers=config.num_layers,
#         batch_size=batch_size,
#         num_heads=config.num_heads,
#         max_seq_len=max_seq_len,
#         head_dim=head_dim,
#         dtype=next(model.parameters()).dtype,
#         device=device,
#     )

def make_kv_cache_(model, config, batch_size, max_seq_len, device):

    head_dim = config.d_model // config.num_heads

    return KVCache_kv(
        num_layers=config.num_layers,
        batch_size=batch_size,
        num_heads=config.num_kv_heads,
        max_seq_len=max_seq_len,
        head_dim=head_dim,
        dtype=next(model.parameters()).dtype,
        device=device,
    )


@torch.no_grad()
def generate(
    model,
    prompt_tokens,
    max_new_tokens,
    config,
):

    model.eval()

    tokens = prompt_tokens

    kv_cache = make_kv_cache_(
        model,
        config,
        batch_size=prompt_tokens.shape[0],
        max_seq_len=prompt_tokens.shape[1] + max_new_tokens,
        device=prompt_tokens.device,
    )

    torch.cuda.synchronize()
    start = time.perf_counter()

    logits = model(
        tokens,
        kv_cache=kv_cache,
    )

    torch.cuda.synchronize()
    ttft_end = time.perf_counter()

    for step in range(max_new_tokens):

        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1,
        )


        new_token = tokens[:, -1:]

        cache_length = kv_cache.length

        print(
            f"step {step} → "
            f"model input T={new_token.shape[1]} | "
            f"cache T={cache_length}"
        )

        logits = model(
            new_token,
            kv_cache=kv_cache,
        )

    torch.cuda.synchronize()
    end = time.perf_counter()


    ttft_ms = (
        ttft_end - start
    ) * 1000

    total_ms = (
        end - start
    ) * 1000

    tokens_per_sec = (
        max_new_tokens
        / (total_ms / 1000)
    )

    peak_memory_mb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    print()
    print("========== KV CACHE ==========")
    print(
        f"prompt length: {prompt_tokens.shape[1]}"
    )
    print(
        f"generate:      {max_new_tokens} tokens"
    )
    print()
    print(
        f"TTFT:          {ttft_ms:.3f} ms"
    )
    print(
        f"total:         {total_ms:.3f} ms"
    )
    print(
        f"tokens/sec:    {tokens_per_sec:.2f}"
    )
    print(
        f"peak memory:   {peak_memory_mb:.2f} MB"
    )
    print("==============================")

    return tokens


@torch.no_grad()
def generate_nocache(
    model,
    prompt_tokens,
    max_new_tokens,
):

    model.eval()

    tokens = prompt_tokens

    torch.cuda.synchronize()
    start = time.perf_counter()

    for step in range(max_new_tokens):

        logits = model(tokens)

        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        tokens = torch.cat(
            [tokens, next_token],
            dim=1,
        )

        print(
            f"step {step} → "
            f"model input T={tokens.shape[1]}"
        )

    torch.cuda.synchronize()
    end = time.perf_counter()

    total_ms = (
        end - start
    ) * 1000

    tokens_per_sec = (
        max_new_tokens
        / (total_ms / 1000)
    )

    print()
    print("========== NO KV CACHE ==========")
    print(
        f"prompt length: {prompt_tokens.shape[1]}"
    )
    print(
        f"generate:      {max_new_tokens} tokens"
    )
    print(
        f"total:         {total_ms:.3f} ms"
    )
    print(
        f"tokens/sec:    {tokens_per_sec:.2f}"
    )
    print("=================================")

    return tokens



@torch.no_grad()
def test_first_decode(
    model_nocache,
    model_cache,
    prompt_tokens,
    config,
):

    model_nocache.eval()
    model_cache.eval()

    logits_nc = model_nocache(
        prompt_tokens
    )

    cache = make_kv_cache_(
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


# ============================================================
# LLM GENERATED
# ============================================================
if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    config = ModelConfig()


    model_cache = Transformer(
        config
    ).to(device)

    model_nocache = Transformer_nocache(
        config
    ).to(device)

    # Same weights.
    model_nocache.load_state_dict(
        model_cache.state_dict()
    )


    B = 4
    T = 512
    MAX_NEW_TOKENS = 20

    prompt_tokens = torch.randint(
        0,
        config.vocab_size,
        (B, T),
        device=device,
    )

    # Reset CUDA peak-memory measurement.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


    print("\n\n")
    print("########################################")
    print("#          WITHOUT KV CACHE            #")
    print("########################################")

    generated_nocache = generate_nocache(
        model_nocache,
        prompt_tokens,
        max_new_tokens=MAX_NEW_TOKENS,
    )


    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("\n\n")
    print("########################################")
    print("#           WITH KV CACHE               #")
    print("########################################")

    generated_cache = generate(
        model_cache,
        prompt_tokens,
        max_new_tokens=MAX_NEW_TOKENS,
        config=config,
    )


    print()
    print("Prompt shape:")
    print(prompt_tokens.shape)

    print()
    print("No-cache output:")
    print(generated_nocache.shape)

    print()
    print("KV-cache output:")
    print(generated_cache.shape)

    same = torch.equal(
        generated_nocache,
        generated_cache,
    )

    print()
    print(
        "Outputs identical:",
        same,
    )


    with torch.no_grad():

        logits_nocache = model_nocache(
            prompt_tokens
        )

        prefill_cache = make_kv_cache_(
            model_cache,
            config,
            batch_size=B,
            max_seq_len=T + 1,
            device=device,
        )

        logits_cache = model_cache(
            prompt_tokens,
            kv_cache=prefill_cache,
        )

        diff = (
            logits_nocache
            -
            logits_cache
        ).abs().max()

        print(
            "Prefill max diff:",
            diff.item()
        )

    test_first_decode(
        model_nocache,
        model_cache,
        prompt_tokens,
        config,
    )