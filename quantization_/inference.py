import torch, argparse, tiktoken
import torch.nn as nn
import torch.nn.functional as F
from .quant_int8 import dequantize_

from models.transformer_block import Transformer
from models.model_config import ModelConfig
from kernels.capability import resolve_backend
from sampling import sample
from kv_cache import KVCache_kv

enc = tiktoken.get_encoding("gpt2")


class ManualQuantLinear(nn.Module):
    def __init__(self, q_weight, scale, quantize_per_, group_size, pad, out_features, in_features):
        super().__init__()
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale)
        self.quantize_per_ = quantize_per_
        self.group_size = group_size
        self.pad = pad
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x):
        w = dequantize_(self.q_weight, self.scale, self.quantize_per_,
                         self.group_size, self.pad)

        if w.shape[0] != self.out_features or w.shape[1] != self.in_features:
            w = w[:self.out_features, :self.in_features]
        w = w.to(dtype=x.dtype, device=x.device)

        return F.linear(x, w, None)


class ManualQuantEmbedding(nn.Module):
    """
    Embedding lookup is a row-gather (w[x]), NOT a matmul (x @ w).
    x here is int64 token ids, so it must go through F.embedding, never F.linear.
    """
    def __init__(self, q_weight, scale, quantize_per_, group_size, pad, num_embeddings, embedding_dim):
        super().__init__()
        self.register_buffer("q_weight", q_weight)
        self.register_buffer("scale", scale)
        self.quantize_per_ = quantize_per_
        self.group_size = group_size
        self.pad = pad
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, x):
        w = dequantize_(self.q_weight, self.scale, self.quantize_per_,
                         self.group_size, self.pad)

        if w.shape[0] != self.num_embeddings or w.shape[1] != self.embedding_dim:
            w = w[:self.num_embeddings, :self.embedding_dim]
        w = w.to(dtype=torch.float32, device=x.device)

        return F.embedding(x, w)


def set_module_by_name(model, name, module):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)

    setattr(parent, parts[-1], module)


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    run_time = ModelConfig()
    for k, v in ckpt["model_config"].items():
        setattr(run_time, k, v)

    train_cfg = ckpt.get("train_config", {})
    attn_type = train_cfg.get("attn_type", "mqa")
    backend = resolve_backend(train_cfg.get("backend", "pytorch"), run_time, attn_type)

    model = Transformer(run_time, attn_type=attn_type, backend=backend)

    quant_metadata = ckpt["quant_metadata"]
    state_dict = ckpt["model"]

    for name, meta in quant_metadata.items():
        q_weight = state_dict[f"{name}.qweight"]
        scale = state_dict[f"{name}.scale"]
        module_path = name[:-len(".weight")]

        module_type = meta.get("module_type", "linear")

        """
        embedding : because its a look up table
        linear    : for projection

        """
        if module_type == "embedding":
            new_module = ManualQuantEmbedding(
                q_weight=q_weight,
                scale=scale,
                quantize_per_=meta["quantize_per_"],
                group_size=meta["group_size"],
                pad=meta["pad"],
                num_embeddings=meta["orig_shape"][0],
                embedding_dim=meta["orig_shape"][1],
            )
        else:
            new_module = ManualQuantLinear(
                q_weight=q_weight,
                scale=scale,
                quantize_per_=meta["quantize_per_"],
                group_size=meta["group_size"],
                pad=meta["pad"],
                out_features=meta["orig_shape"][0],
                in_features=meta["orig_shape"][1],
            )

        set_module_by_name(model, module_path, new_module)

    plain = {k: v for k, v in state_dict.items()
             if not k.endswith(".qweight") and not k.endswith(".scale")}
    missing, unexpected = model.load_state_dict(plain, strict=False)
    print(f"Loaded plain tensors. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model = model.to(device)
    model.eval()
    return model, run_time


def prefill(model, tokens, kv_cache):
    with torch.no_grad():
        logits = model(tokens, kv_cache=kv_cache)  
    return logits[:, -1, :]  


def decode_one(model, next_token, kv_cache):
    with torch.no_grad():
        logits = model(next_token, kv_cache=kv_cache)  
    return logits[:, -1, :]  


def generate(model, run_time, device, prompt, max_new_tokens=100, temperature=0.9, top_k=20, top_p=1.0):
    ids = enc.encode_ordinary(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    kv_heads = getattr(run_time, "num_kv_heads", run_time.num_heads)
    head_dim = run_time.d_model // run_time.num_heads

    model_dtype = next(model.parameters()).dtype

    kv_cache = KVCache_kv(
        num_layers=run_time.num_layers,
        batch_size=ids.shape[0],
        num_heads=kv_heads,
        max_seq_len=run_time.max_seq_len,
        head_dim=head_dim,
        dtype=model_dtype,
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


"""
don't use backend , causal , device and attn type as it will auto align with weights
"""
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_int8.pt")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--token", type=int, default=100)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    model, run_time = load_model(
        args.ckpt,
        args.device
    )

    for piece in generate(
        model, run_time, args.device, args.prompt,
        args.token, args.temperature, args.top_k, args.top_p,
    ):
        print(piece, end="", flush=True)
    print()