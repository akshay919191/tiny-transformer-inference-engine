import argparse
import torch
import tiktoken

from serving.paged_model import load_paged_model
from serving.batch import build_batch_step
from serving.allocator import BlockAllocator
from serving.kv_pool import KVPool
from serving.sequence import Sequence
from sampling import sample

enc = tiktoken.get_encoding("gpt2")


@torch.no_grad()
def generate_batched_text(model, cfg, prompts, max_new_tokens, eos_token_id,
                           pool, alloc, temperature=1.0, top_k=32, top_p=0.9,
                           block_size=16):
    """Returns a list of full generated-token-id lists, one per prompt (EOS excluded)."""
    device = next(model.parameters()).device
    prompt_ids = [enc.encode_ordinary(p) for p in prompts]
    seqs = [Sequence(i, list(p), block_size) for i, p in enumerate(prompt_ids)]
    for s in seqs:
        s.ensure_blocks(alloc)

    outputs = [[] for _ in seqs]
    finished = [False] * len(seqs)

    def pick(logits_row):
        return int(sample(logits_row.unsqueeze(0), temperature=temperature, top_k=top_k, top_p=top_p))

    batch = build_batch_step([(s, s.token_ids) for s in seqs], device)
    logits = model.forward_batch(batch, pool)
    next_tokens = {i: pick(logits[i]) for i in range(len(seqs))}

    for i, t in next_tokens.items():
        if t == eos_token_id:
            finished[i] = True
        else:
            outputs[i].append(t)

    for _ in range(max_new_tokens - 1):
        if all(finished):
            break
        active = [i for i in range(len(seqs)) if not finished[i]]
        for i in active:
            seqs[i].append_token(next_tokens[i])
            seqs[i].ensure_blocks(alloc)

        batch = build_batch_step([(seqs[i], [next_tokens[i]]) for i in active], device)
        logits = model.forward_batch(batch, pool)

        step_tokens = {}
        for pos, i in enumerate(active):
            t = pick(logits[pos])
            step_tokens[i] = t
            if t == eos_token_id:
                finished[i] = True
            else:
                outputs[i].append(t)
        next_tokens = step_tokens

    for s in seqs:
        s.release(alloc)
    return outputs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/ckpt_step20000.pt")
    p.add_argument("--prompts", nargs="+", default=[
        "Once upon a time", "The little dog", "In the forest there was"
    ])
    p.add_argument("--token", type=int, default=45)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_gpu_blocks", type=int, default=256)
    args = p.parse_args()

    model, cfg = load_paged_model(args.ckpt, args.device)
    alloc = BlockAllocator(args.num_gpu_blocks)
    pool = KVPool(cfg.num_layers, 16, args.num_gpu_blocks, cfg.num_kv_heads,
                  cfg.d_model // cfg.num_heads, torch.float32, args.device)

    outputs = generate_batched_text(
        model, cfg, args.prompts, args.token, cfg.eos_token_id, pool, alloc,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
    )

    print()
    for prompt, out_ids in zip(args.prompts, outputs):
        text = enc.decode(out_ids)
        print(f"[{prompt}]")
        print(prompt + text)
        print()