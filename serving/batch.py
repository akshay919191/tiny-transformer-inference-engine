import torch
from dataclasses import dataclass


@dataclass
class BatchStep:
    input_ids: torch.Tensor        
    positions: list             
    query_start_loc: list         
    block_tables: list           
    seq_lens: list                

    @property
    def num_seqs(self):
        return len(self.block_tables)


def build_batch_step(entries, device):
    """entries: list of (seq, new_token_ids) pairs.
    new_token_ids is the prompt tokens (prefill) or [last_sampled_token] (decode)."""
    input_ids, positions, qsl, block_tables, seq_lens = [], [], [0], [], []
    for seq, new_tokens in entries:
        start = seq.num_tokens - len(new_tokens)
        input_ids.extend(new_tokens)
        positions.extend(range(start, start + len(new_tokens)))
        qsl.append(qsl[-1] + len(new_tokens))
        block_tables.append(seq.block_table)
        seq_lens.append(seq.num_tokens)

    return BatchStep(
        input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
        positions=positions,
        query_start_loc=qsl,
        block_tables=block_tables,
        seq_lens=seq_lens,
    )