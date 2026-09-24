import torch
import torch.nn.functional as F
from models.mqa import MQA_Cached 


class MQA_Paged(MQA_Cached):
    def forward_paged(self, x, pool, layer_idx, block_table, start):
        """
        x: [1, SQ, d_model]   pool: KVPool   block_table: list[int]
        start: this sequence's position before these tokens (0 for prefill)
        """
        B, SQ, _ = x.shape
        assert B == 1

        q = self.q_proj(x).view(B, SQ, self.num_heads, self.headdim).transpose(1, 2)
        k = self.k_proj(x).view(B, SQ, self.num_kv_heads, self.headdim).transpose(1, 2)
        v = self.v_proj(x).view(B, SQ, self.num_kv_heads, self.headdim).transpose(1, 2)

        positions = torch.arange(start , start + SQ , device = x.device)

        q , k = self.rope.forward_at(q , k , positions = positions)


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
            
        out = out.transpose(1 , 2).contiguous().view(B , SQ , self.d_model)

        return self.out(out)