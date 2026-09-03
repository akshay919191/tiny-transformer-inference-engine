import argparse
import torch
import torch.nn.functional as F
import tiktoken

from models.transformer_block import Transformer
from models.model_config import ModelConfig
from sampling import sample
from kv_cache import KVCache_kv

enc = tiktoken.get_encoding("gpt2")


def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")


def load_model(ckpt_path, device, attn_type=None, backend=None, causal=None):
    ckpt = torch.load(ckpt_path, map_location=device)

    run_time = ModelConfig()
    for k, v in ckpt["model_config"].items():
        setattr(run_time, k, v)

    train_cfg = ckpt.get("train_config", {})

    resolved_attn_type = attn_type if attn_type is not None else train_cfg.get("attn_type", "mqa")
    resolved_backend = backend if backend is not None else train_cfg.get("backend", "pytorch")

    if causal is not None:
        run_time.causal = causal

    model = Transformer(run_time, attn_type=resolved_attn_type, backend=resolved_backend).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, run_time


def prefill(model, tokens, kv_cache):
    with torch.no_grad():
        logits = model(tokens, kv_cache=kv_cache)  # [B, T, vocab]
    return logits[:, -1, :]  # [B, vocab]


def decode_one(model, next_token, kv_cache):
    with torch.no_grad():
        logits = model(next_token, kv_cache=kv_cache)  # [B, 1, vocab]
    return logits[:, -1, :]  # [B, vocab]


def generate(model, run_time, device, prompt, max_new_tokens=100, temperature=0.9, top_k=20, top_p=1.0):
    ids = enc.encode_ordinary(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  

    kv_heads = getattr(run_time, "num_kv_heads", run_time.num_heads)
    head_dim = run_time.d_model // run_time.num_heads

    kv_cache = KVCache_kv(
        num_layers=run_time.num_layers,
        batch_size=ids.shape[0],
        num_heads=kv_heads,
        max_seq_len=run_time.max_seq_len,
        head_dim=head_dim,
        dtype=torch.float32, ## custom kernel are fp16 based but internal conversion is used 
        device=device,
    )

    logits = prefill(model, ids, kv_cache)
    next_id = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)  

    ids = torch.cat([ids, next_id], dim=1)
    yield enc.decode([next_id.item()])

    for _ in range(max_new_tokens - 1):
        logits = decode_one(model, next_id, kv_cache)
        next_id = sample(logits, temperature=temperature, top_k=top_k, top_p=top_p)  

        ids = torch.cat([ids, next_id], dim=1)
        yield enc.decode([next_id.item()])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_final.pt")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--token", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--attn_type", type=str, default=None, choices=[None, "mqa", "mha"])
    p.add_argument("--backend", type=str, default=None, choices=[None, "cuda", "pytorch"])
    p.add_argument("--causal", type=str2bool, default=None)
    args = p.parse_args()

    model, run_time = load_model(
        args.ckpt,
        args.device,
        attn_type=args.attn_type,
        backend=args.backend,
        causal=args.causal,
    )

    for piece in generate(
        model, run_time, args.device, args.prompt,
        args.token, args.temperature, args.top_k, args.top_p,
    ):
        print(piece, end="", flush=True)
    print()