import torch
import torch.nn.functional as F

def paged_varlen_attention_test(
    q: torch.Tensor,               # Flattened [T, H, D]
    pool,                          # Your KVPool manager object
    layer_idx: int,
    block_tables: list[list[int]], # e.g. [[1, 2], [4]]
    query_start_loc: list[int],    # e.g. [0, 3, 4]
    positions: list[int],          # e.g. [0, 1, 2, 46]
    num_heads: int,
    num_kv_heads: int
) -> torch.Tensor:
    """
    Pure PyTorch fallback for testing variable-length paged attention.
    """
    T, H, D = q.shape
    out = torch.zeros_like(q)  

    num_seqs = len(block_tables)

    for i in range(num_seqs):
        start_idx = query_start_loc[i]
        end_idx = query_start_loc[i + 1]
        sq_i = end_idx - start_idx

        if sq_i == 0:
            continue

        q_seq = q[start_idx:end_idx].transpose(0, 1).unsqueeze(0)

        seq_len = positions[end_idx - 1] + 1
        block_table = block_tables[i]

        k_all, v_all = pool.gather(layer_idx, block_table, seq_len)

        k_seq = k_all.transpose(0, 1).unsqueeze(0).contiguous()
        v_seq = v_all.transpose(0, 1).unsqueeze(0).contiguous()

        is_causal = (sq_i > 1 and positions[start_idx] == 0)
        assert sq_i == 1 or positions[start_idx] == 0, "chunked prefill needs an explicit mask"

        out_seq = F.scaled_dot_product_attention(
            q_seq, k_seq, v_seq,
            is_causal=is_causal,
            enable_gqa=(num_kv_heads != num_heads)
        )

        out[start_idx:end_idx] = out_seq.squeeze(0).transpose(0, 1)

    return out