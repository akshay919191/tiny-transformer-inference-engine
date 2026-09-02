import argparse
import torch
import torch.nn.functional as F
import tiktoken

from models.transformer_block import Transformer
from models.model_config import ModelConfig

device = "cuda"
enc = tiktoken.get_encoding("gpt2")


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)

    run_time = ModelConfig()
    for k, v in ckpt["model_config"].items():
        setattr(run_time, k, v)

    train_cfg = ckpt.get("train_config", {})
    attn_type = train_cfg.get("attn_type", "mqa")
    backend = train_cfg.get("backend", "pytorch")

    model = Transformer(run_time, attn_type=attn_type, backend=backend).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, run_time


model, run_time = load_model("checkpoints/ckpt_final.pt")


def generate(prompt, max_new_tokens=100, temperature=1.0):
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
    args = p.parse_args()
    args = p.parse_args()

    model, run_time = load_model(args.ckpt)

    for piece in generate(args.prompt, args.token, args.temperature):
        print(piece, end="", flush=True)
    print()