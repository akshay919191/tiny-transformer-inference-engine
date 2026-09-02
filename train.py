# train.py
import os
import argparse
from dataclasses import fields

import numpy as np
import torch
import torch.nn.functional as F

from models.transformer_block import Transformer
from models.model_config import ModelConfig


# ------------------------------------------------- argparse setup

def str2bool(v):
    return str(v).lower() in ("1", "true", "yes", "y")

TYPE_MAP = {"int": int, "float": float, "bool": str2bool, "str": str}


def add_model_args(parser):
    """Auto-create one CLI flag per ModelConfig field, e.g. --num_layers 12"""
    group = parser.add_argument_group("model config")
    for f in fields(ModelConfig):
        ftype = TYPE_MAP.get(f.type if isinstance(f.type, str) else f.type.__name__, str)
        group.add_argument(
            f"--{f.name}",
            type=ftype,
            default=f.default,
            help="default: %(default)s",
        )


def build_parser():
    p = argparse.ArgumentParser(description="Train a small transformer")

    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--backend", type=str, default="cuda", choices=["cuda", "pytorch"],
                   help="attention kernel implementation")
    p.add_argument("--attn_type", type=str, default="mqa", choices=["mqa", "mha"],
                   help="multi-query vs multi-head attention")

    add_model_args(p)   
    return p



def get_batch(split, config, device):
    data = np.memmap(f"data/{split}.bin", dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - config.max_seq_len - 1, (config.batch,))
    batch = torch.stack([
        torch.from_numpy(data[i : i + config.max_seq_len + 1].astype(np.int64))
        for i in ix
    ])
    return batch.to(device)


def save_checkpoint(path, model, optimizer, step, model_config, train_args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_config": vars(model_config),
        "train_config": vars(train_args),
    }, path)
    print(f"Saved checkpoint: {path}")


def train(args):
    device = args.device

    run_time = ModelConfig(**{f.name: getattr(args, f.name) for f in fields(ModelConfig)})
    print(run_time)   

    if args.seed is not None:
        torch.manual_seed(args.seed)

        model = Transformer(run_time, attn_type=args.attn_type, backend=args.backend).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.train()

    for step in range(args.max_steps):
        input_ids = get_batch("train", run_time, device)
        x, y = input_ids[:, :-1], input_ids[:, 1:]

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            model.eval()
            with torch.no_grad():
                val_ids = get_batch("val", run_time, device)
                vx, vy = val_ids[:, :-1], val_ids[:, 1:]
                val_logits = model(vx)
                val_loss = F.cross_entropy(val_logits.reshape(-1, val_logits.size(-1)), vy.reshape(-1))
            model.train()
            print(f"step {step} | train loss {loss.item():.4f} | val loss {val_loss.item():.4f}")

        if step % args.save_every == 0 and step > 0:
            save_checkpoint(os.path.join(args.out_dir, f"ckpt_step{step}.pt"),
                            model, optimizer, step, run_time, args)

    save_checkpoint(os.path.join(args.out_dir, "ckpt_final.pt"),
                    model, optimizer, args.max_steps, run_time, args)


if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)