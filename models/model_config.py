from dataclasses import dataclass
import torch

## initially tiny
@dataclass
class ModelConfig:
    vocab_size: int = 50257
    hidden_size: int = 128
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads : int = 1
    max_seq_len: int = 512
    d_model : int  = 256
    casual = True
    dropout = 0.0
    bias = False
    batch = 4
    dtype = torch.float32