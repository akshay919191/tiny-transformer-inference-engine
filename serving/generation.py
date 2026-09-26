import torch , time
import torch.nn.functional as F

from .paged_model import PagedTransformer
from .sequence import Sequence
from models.model_config import ModelConfig
from .batch import build_batch_step

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
    seq = Sequence(0, list(prompt_ids), block_size)

    next_token = prefill_seq(model, seq, pool, alloc)

    out = []
    for _ in range(max_new_tokens):
        out.append(next_token)
        if len(out) == max_new_tokens:
            break
        next_token = decode_step(model, seq, pool, alloc, next_token)

    seq.release(alloc)
    return out

@torch.no_grad()
def generate_batched(model, prompts_ids, max_new_tokens, pool, alloc, block_size=16):
    """prompts_ids: list of token-id lists, one per sequence.
    Returns: list of generated-token lists, one per sequence, same order."""
    device = next(model.parameters()).device
    seqs = [Sequence(i, list(p), block_size) for i, p in enumerate(prompts_ids)]
    for s in seqs:
        s.ensure_blocks(alloc)

    outputs = [[] for _ in seqs]

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
        next_tokens = [int(logits[i].argmax()) for i in range(len(seqs))]   # <- this was missing
        for i, t in enumerate(next_tokens):
            outputs[i].append(t)

    for s in seqs:
        s.release(alloc)
    return outputs

