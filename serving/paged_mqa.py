import torch
import torch.nn.functional as F
from models.mqa import MQA_Cached 

from .fake_paged_attn import paged_varlen_attention_test


class MQA_Paged(MQA_Cached):
    def forward_paged(self, x, pool, layer_idx, block_table, start):
        B, SQ, _ = x.shape
        assert B == 1

        q = self.q_proj(x).view(B, SQ, self.num_heads, self.headdim).transpose(1, 2)
        k = self.k_proj(x).view(B, SQ, self.num_kv_heads, self.headdim).transpose(1, 2)
        v = self.v_proj(x).view(B, SQ, self.num_kv_heads, self.headdim).transpose(1, 2)

        positions = torch.arange(start, start + SQ, device=x.device)

        q, k = self.rope.forward_at(q, k, positions=positions)

        k_flat = k[0].transpose(0, 1)     
        v_flat = v[0].transpose(0, 1)

        slot_mapping = pool.make_slot_mapping(start, SQ, block_table)
        pool.write(layer_idx, slot_mapping, k_flat, v_flat)

        k_all, v_all = pool.gather(layer_idx, block_table, start + SQ)
        k_all = k_all.transpose(0, 1).unsqueeze(0).contiguous()   
        v_all = v_all.transpose(0, 1).unsqueeze(0).contiguous()

        out = F.scaled_dot_product_attention(
                q, k_all, v_all,
                is_causal=(SQ > 1 and start == 0),
                enable_gqa=(self.num_kv_heads != self.num_heads))
            
        out = out.transpose(1, 2).contiguous().view(B, SQ, self.d_model)

        return self.out(out)

    def forward_paged_batched(self, x, pool, layer_idx, positions, query_start_loc, block_tables):
        T, _ = x.shape
        device = x.device

        q = self.q_proj(x).view(T, self.num_heads, self.headdim)
        k = self.k_proj(x).view(T, self.num_kv_heads, self.headdim)
        v = self.v_proj(x).view(T, self.num_kv_heads, self.headdim)

        positions_tensor = torch.tensor(positions, device=device, dtype=torch.long)

        q4 = q.transpose(0, 1).unsqueeze(0)
        k4 = k.transpose(0, 1).unsqueeze(0)
        q4, k4 = self.rope.forward_at(q4, k4, positions=positions_tensor)
        q = q4.squeeze(0).transpose(0, 1)
        k = k4.squeeze(0).transpose(0, 1)

        slot_mappings = []
        for i in range(len(block_tables)):
            start_idx, end_idx = query_start_loc[i], query_start_loc[i + 1]
            sq_i = end_idx - start_idx
            if sq_i == 0:
                continue
            seq_start_pos = positions[start_idx]
            assert sq_i == 1 or seq_start_pos == 0, "chunked prefill not supported here"
            slot_mappings.append(pool.make_slot_mapping(seq_start_pos, sq_i, block_tables[i]))

        pool.write(layer_idx, torch.cat(slot_mappings), k, v)

        out = paged_varlen_attention_test(
            q=q, pool=pool, layer_idx=layer_idx,
            block_tables=block_tables, query_start_loc=query_start_loc,
            positions=positions, num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
        )
        out = out.contiguous().view(T, self.d_model)
        return self.out(out)