from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    hidden_size: int = 512
    num_layers: int = 34
    num_heads: int = 8
    num_kv_heads: int = 1
    max_seq_len: int = 512
    d_model: int = 512
    causal: bool = True       
    dropout: float = 0.0     
    bias: bool = False        
    batch: int = 4            
    dtype: torch.dtype = torch.float32  
