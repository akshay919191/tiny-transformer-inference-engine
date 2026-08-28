import torch , time
import torch.nn as nn

from attention import MHA , MHA_CACHED
from embedding import TokenEmbedding
from mlp import SwiGLU
from model_config import ModelConfig
from rmsnorm import RMSNorm

class TransformerBlock_Nocache(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_norm = RMSNorm(config.d_model)

        self.attn = MHA(
            config.num_heads,
            config.d_model,
            config.dropout,
            config.bias,
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

        for i, layer in enumerate(self.layers):

            x = layer(
                x
            )

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_norm = RMSNorm(config.d_model)

        self.attn = MHA_CACHED(
            config.num_heads,
            config.d_model,
            config.dropout,
            config.bias,
        )

        self.mlp_norm = RMSNorm(config.d_model)

        self.mlp = SwiGLU(
            config.d_model,
            config.hidden_size,
            config.bias,
        )

    def forward(self, x, kv_cache=None):

        residual = x

        x = self.attn_norm(x)

        x, new_kv_cache = self.attn(
            x,
            x,
            x,
            kv_cache=kv_cache,
            causal=True,
        )

        x = x + residual

        residual = x

        x = self.mlp_norm(x)

        x = self.mlp(x)

        x = x + residual

        return x, new_kv_cache


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding = TokenEmbedding(
            config.vocab_size,
            config.d_model,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(config)
            for _ in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

    def forward(self, input_ids, kv_cache=None):

        x = self.embedding(input_ids)

        if kv_cache is None:
            kv_cache = [None] * len(self.layers)

        new_kv_cache = []

        for i, layer in enumerate(self.layers):

            x, layer_kv_cache = layer(
                x,
                kv_cache=kv_cache[i],
            )

            new_kv_cache.append(layer_kv_cache)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits, new_kv_cache

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

    total_ms = (end - start) * 1000

    tokens_per_sec = (
        max_new_tokens / (total_ms / 1000)
    )

    print()
    print("========== NO KV CACHE ==========")
    print(f"prompt length: {prompt_tokens.shape[1]}")
    print(f"generate:      {max_new_tokens} tokens")
    print(f"total:         {total_ms:.3f} ms")
    print(f"tokens/sec:    {tokens_per_sec:.2f}")
    print("=================================")

    return tokens

@torch.no_grad()
def generate(
    model,
    prompt_tokens,
    max_new_tokens,
):
    model.eval()

    tokens = prompt_tokens

    kv_cache = [None] * len(model.layers)

    torch.cuda.synchronize()
    start = time.perf_counter()

    logits, kv_cache = model(
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

        print(
            f"step {step} → "
            f"model input T={new_token.shape[1]} | "
            f"cache T={kv_cache[0][0].shape[2]}"
        )

        logits, kv_cache = model(
            new_token,
            kv_cache=kv_cache,
        )

    torch.cuda.synchronize()
    end = time.perf_counter()

    ttft_ms = (ttft_end - start) * 1000
    total_ms = (end - start) * 1000

    generated_tokens_count = max_new_tokens

    tokens_per_sec = (
        generated_tokens_count
        / (total_ms / 1000)
    )

    peak_memory_mb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 2)
    )

    print()
    print("========== Generation Benchmark ==========")
    print(f"prompt length: {prompt_tokens.shape[1]}")
    print(f"generate:      {generated_tokens_count} tokens")
    print()
    print(f"TTFT:          {ttft_ms:.3f} ms")
    print(f"total:         {total_ms:.3f} ms")
    print(f"tokens/sec:    {tokens_per_sec:.2f}")
    print(f"peak memory:   {peak_memory_mb:.2f} MB")
    print("===========================================")

    return tokens

@torch.no_grad()
def test_first_decode(
    model_nocache,
    model_cache,
    prompt_tokens,
):
    model_nocache.eval()
    model_cache.eval()


    logits_nc = model_nocache(prompt_tokens)

    logits_c, cache = model_cache(
        prompt_tokens,
        kv_cache=[None] * len(model_cache.layers),
    )

    print(
        "Prefill diff:",
        (logits_nc - logits_c).abs().max().item()
    )
    next_token = torch.argmax(
        logits_nc[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    tokens = torch.cat(
        [prompt_tokens, next_token],
        dim=1,
    )

    logits_nc = model_nocache(tokens)


    logits_c, cache = model_cache(
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

### main function is LLM generated
if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = ModelConfig()

    model_cache = Transformer(config).to(device)

    model_nocache = Transformer_nocache(config).to(device)

    # Same weights
    model_nocache.load_state_dict(
        model_cache.state_dict()
    )

    B = 4
    T = 512

    prompt_tokens = torch.randint(
        0,
        config.vocab_size,
        (B, T),
        device=device,
    )


    print("\n\n")
    print("########################################")
    print("#          WITHOUT KV CACHE            #")
    print("########################################")

    generated_nocache = generate_nocache(
        model_nocache,
        prompt_tokens,
        max_new_tokens=20,
    )

    print("\n\n")
    print("########################################")
    print("#           WITH KV CACHE               #")
    print("########################################")

    generated_cache = generate(
        model_cache,
        prompt_tokens,
        max_new_tokens=20,
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
    print("Outputs identical:", same)

    with torch.no_grad():

        logits_nocache = model_nocache(
            prompt_tokens
        )

        logits_cache, cache = model_cache(
            prompt_tokens,
            kv_cache=[None] * len(model_cache.layers),
        )

        diff = (
            logits_nocache -
            logits_cache
        ).abs().max()

        print("Prefill max diff:", diff.item())

    test_first_decode(
        model_nocache,
        model_cache,
        prompt_tokens,
    )