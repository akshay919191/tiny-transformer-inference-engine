import argparse
import torch
import torch.nn.functional as F
import tiktoken

from models.transformer_block import Transformer
from models.model_config import ModelConfig

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


def generate(model, run_time, device, prompt, max_new_tokens=100, temperature=1.0):
    ids = enc.encode_ordinary(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        ids_cond = ids[:, -run_time.max_seq_len:]
        with torch.no_grad():
            logits = model(ids_cond)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        yield enc.decode([next_id.item()])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_final.pt")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--temperature", type=float, default=1.0)
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

    for piece in generate(model, run_time, args.device, args.prompt, args.token, args.temperature):
        print(piece, end="", flush=True)
    print()