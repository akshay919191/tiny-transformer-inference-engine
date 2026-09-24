import torch , time
from .paged_model import PagedTransformer
from .sequence import Sequence
from models.model_config import ModelConfig


@torch.no_grad()
def generate_paged(model, prompt_ids, max_new_tokens, pool, alloc, block_size=16):
    device = next(model.parameters()).device
    seq = Sequence(0, list(prompt_ids), block_size)
    seq.ensure_blocks(alloc)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits = model(ids, pool, seq.block_table, start=0)         # prefill

    out = []
    for _ in range(max_new_tokens):
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if len(out) == max_new_tokens:
            break
        seq.append_token(nxt)
        seq.ensure_blocks(alloc)                                # before the write
        x = torch.tensor([[nxt]], dtype=torch.long, device=device)
        logits = model(x, pool, seq.block_table, start=seq.num_tokens - 1)
    seq.release(alloc)
    return out