from dataclasses import dataclass


## initially tiny
@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    max_seq_len: int = 512
    d_model : int  = 512
    casual = False
    dropout = 0.0
    bias = False