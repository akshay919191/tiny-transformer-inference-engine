import torch
import tiktoken

from .sequence import Sequence
from .batch import build_batch_step
from sampling import sample

enc = tiktoken.get_encoding("gpt2")


@torch.no_grad()
def generate_batched_text(model, cfg, prompts, max_new_tokens, eos_token_id,
                          pool, alloc, temperature=1.0, top_k=32, top_p=0.9,
                          block_size=16):
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if not prompts:
        return []
    if temperature <= 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid sampling parameters")
    device = next(model.parameters()).device
    prompt_ids = [enc.encode_ordinary(p) for p in prompts]
    if any(not ids for ids in prompt_ids):
        raise ValueError("every prompt must encode to at least one token")
    if any(len(ids) + max_new_tokens > cfg.max_seq_len for ids in prompt_ids):
        raise ValueError("prompt plus requested generation exceeds max_seq_len")
    if max_new_tokens == 0:
        return [[] for _ in prompts]

    seqs = [Sequence(i, list(ids), block_size) for i, ids in enumerate(prompt_ids)]
    outputs = [[] for _ in seqs]
    try:
        for seq in seqs:
            seq.ensure_blocks(alloc)

        def pick(row):
            return int(sample(row.unsqueeze(0), temperature=temperature,
                              top_k=top_k, top_p=top_p).item())

        batch = build_batch_step([(seq, seq.token_ids) for seq in seqs], device)
        logits = model.forward_batch(batch, pool)
        next_tokens = {i: pick(logits[i]) for i in range(len(seqs))}
        finished = [False] * len(seqs)
        for i, token in next_tokens.items():
            finished[i] = token == eos_token_id
            if not finished[i]:
                outputs[i].append(token)

        for _ in range(max_new_tokens - 1):
            if all(finished):
                break
            active = [i for i, done in enumerate(finished) if not done]
            for i in active:
                seqs[i].append_token(next_tokens[i])
                seqs[i].ensure_blocks(alloc)
            batch = build_batch_step([(seqs[i], [next_tokens[i]]) for i in active], device)
            logits = model.forward_batch(batch, pool)
            current = {}
            for row, i in enumerate(active):
                token = pick(logits[row])
                current[i] = token
                finished[i] = token == eos_token_id
                if not finished[i]:
                    outputs[i].append(token)
            next_tokens = current
        return outputs
    finally:
        for seq in seqs:
            seq.release(alloc)

@torch.no_grad()
def prefill_seq(model, seq, pool, alloc):
    seq.ensure_blocks(alloc)
    device = next(model.parameters()).device
    ids = torch.tensor([seq.token_ids], dtype=torch.long, device=device)
    logits = model(ids, pool, seq.block_table, start=0)
    return int(logits[0, -1].argmax())


@torch.no_grad()
def decode_step(model, seq, pool, alloc, token_id):
    seq.append_token(token_id)
    seq.ensure_blocks(alloc)          
    device = next(model.parameters()).device
    x = torch.tensor([[token_id]], dtype=torch.long, device=device)
    start = seq.num_tokens - 1        
    logits = model(x, pool, seq.block_table, start=start)
    return int(logits[0, -1].argmax())


@torch.no_grad()
def generate_paged(model, prompt_ids, max_new_tokens, pool, alloc, block_size=16):
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if not prompt_ids:
        raise ValueError("prompt must contain at least one token")
    if max_new_tokens == 0:
        return []
    seq = Sequence(0, list(prompt_ids), block_size)
    try:
        next_token = prefill_seq(model, seq, pool, alloc)
        out = []
        for _ in range(max_new_tokens):
            out.append(next_token)
            if len(out) == max_new_tokens:
                break
            next_token = decode_step(model, seq, pool, alloc, next_token)
        return out
    finally:
        seq.release(alloc)

@torch.no_grad()
def generate_batched(model, prompts_ids, max_new_tokens, pool, alloc, block_size=16):
    """prompts_ids: list of token-id lists, one per sequence.
    Returns: list of generated-token lists, one per sequence, same order."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if not prompts_ids:
        return []
    if any(not prompt for prompt in prompts_ids):
        raise ValueError("every prompt must contain at least one token")
    if max_new_tokens == 0:
        return [[] for _ in prompts_ids]
    device = next(model.parameters()).device
    seqs = [Sequence(i, list(p), block_size) for i, p in enumerate(prompts_ids)]
    outputs = [[] for _ in seqs]
    try:
        for s in seqs:
            s.ensure_blocks(alloc)
        batch = build_batch_step([(s, s.token_ids) for s in seqs], device)
        logits = model.forward_batch(batch, pool)
        next_tokens = [int(logits[i].argmax()) for i in range(len(seqs))]
        for i, t in enumerate(next_tokens):
            outputs[i].append(t)

        for _ in range(max_new_tokens - 1):
            for s, t in zip(seqs, next_tokens):
                s.append_token(t)
                s.ensure_blocks(alloc)
            batch = build_batch_step([(s, [t]) for s, t in zip(seqs, next_tokens)], device)
            logits = model.forward_batch(batch, pool)
            next_tokens = [int(logits[i].argmax()) for i in range(len(seqs))]
            for i, t in enumerate(next_tokens):
                outputs[i].append(t)
        return outputs
    finally:
        for s in seqs:
            s.release(alloc)

