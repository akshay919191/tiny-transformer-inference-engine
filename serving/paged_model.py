import torch
import torch.nn as nn

from models.embedding import TokenEmbedding
from models.mlp import SwiGLU
from models.model_config import ModelConfig
from models.rmsnorm import RMSNorm
from serving.paged_mqa import MQA_Paged


class PagedBlock(nn.Module):
    def __init__(self, config, layer_idx, backend="pytorch"):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = MQA_Paged(config, backend=backend)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config.d_model, config.hidden_size, config.bias)   # hidden_size = MLP width

    def forward(self, x, pool, block_table, start):
        residual = x
        x = self.attn_norm(x)
        x = self.attn.forward_paged(x, pool, self.layer_idx, block_table, start)
        x = x + residual

        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        return x + residual


class PagedTransformer(nn.Module):
    def __init__(self, config, backend="pytorch"):
        super().__init__()
        self.embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [PagedBlock(config, i, backend) for i in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, pool, block_table, start):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, pool, block_table, start)
        x = x[:, -1:, :]                                   
        return self.lm_head(self.final_norm(x))
        
        
    @torch.no_grad()
    def forward_batch(self, batch, pool):
        x = self.embedding(batch.input_ids.unsqueeze(0)).squeeze(0)   # [T, d_model]
        for layer in self.layers:
            residual = x
            x = layer.attn_norm(x)
            x = layer.attn.forward_paged_batched(
                x, pool, layer.layer_idx, batch.positions, batch.query_start_loc, batch.block_tables
            )
            x = x + residual
            residual = x
            x = layer.mlp_norm(x)
            x = layer.mlp(x)
            x = x + residual

        last_idx = torch.tensor([e - 1 for e in batch.query_start_loc[1:]], device=x.device)
        x_last = x[last_idx]
        return self.lm_head(self.final_norm(x_last))   # [S, vocab]        


def load_paged_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    run_time = ModelConfig.from_checkpoint(ckpt_path)
    model = PagedTransformer(run_time, backend="pytorch")
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)          
    return model.to(device).eval(), run_time