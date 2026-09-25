import torch
import time

from .paged_model import PagedTransformer
from .sequence import Sequence
from models.model_config import ModelConfig


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
    seq.ensure_blocks(alloc)          # blocks must exist before the write — same rule as before
    device = next(model.parameters()).device
    x = torch.tensor([[token_id]], dtype=torch.long, device=device)
    start = seq.num_tokens - 1        # position of the token just appended
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