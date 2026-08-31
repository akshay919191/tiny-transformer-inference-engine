from dataclasses import dataclass
import torch

## initially tiny
@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 8
    num_kv_heads : int = 1
    max_seq_len: int = 1024
    d_model : int  = 512
    casual = False
    dropout = 0.0
    bias = False
    batch = 2
    dtype = torch.float32