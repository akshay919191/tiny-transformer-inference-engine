from dataclasses import dataclass, field
from math import ceil
import torch

def _elem_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 512
    num_layers: int = 34
    num_heads: int = 8
    num_kv_heads: int = 1
    max_seq_len: int = 512          
    causal: bool = True
    dropout: float = 0.0
    bias: bool = False
    dtype: torch.dtype = torch.float32
    eos_token_id: int = 50256       
    rope_theta: float = 10000.0     
    rms_norm_eps: float = 1e-5      

    def __post_init__(self):
        assert self.d_model % self.num_heads == 0
        assert self.num_heads % self.num_kv_heads == 0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    @property
    def hidden_size(self) -> int:          
        return self.d_model

    @classmethod
    def from_checkpoint(cls, path: str) -> "ModelConfig":
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["model_config"]         
        cfg = dict(cfg) if isinstance(cfg, dict) else vars(cfg)
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class CacheConfig:
    block_size: int = 16
    num_gpu_blocks: int | None = None      
    kv_budget_bytes: int = 2 * 1024**3  
    kv_dtype: torch.dtype | None = None    
    enable_prefix_caching: bool = False


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 512
    enable_chunked_prefill: bool = False
    preemption: str = "recompute"
    policy: str = "fcfs"
    cudagraph_batch_sizes: list[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32])

    def __post_init__(self):
        assert self.max_num_batched_tokens >= self.max_num_seqs, \
            "every decode takes 1 token, so the budget must cover all seqs"
        assert self.preemption == "recompute"
        assert max(self.cudagraph_batch_sizes) <= self.max_num_seqs


@dataclass
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    sched: SchedulerConfig = field(default_factory=SchedulerConfig)

    @property
    def kv_dtype(self) -> torch.dtype:
        return self.cache.kv_dtype or self.model.dtype

    @property
    def max_blocks_per_seq(self) -> int:
        return ceil(self.model.max_seq_len / self.cache.block_size)

    @property
    def bytes_per_block(self) -> int:
        m, c = self.model, self.cache
        return (2 * m.num_layers * c.block_size * m.num_kv_heads
                * m.head_dim * _elem_size(self.kv_dtype))

    @property
    def num_blocks(self) -> int:          
        n = self.cache.num_gpu_blocks
        return n if n is not None else self.cache.kv_budget_bytes // self.bytes_per_block

    def __post_init__(self):
        bs = self.cache.block_size
        assert bs & (bs - 1) == 0, "block_size must be a power of two"
        if not self.sched.enable_chunked_prefill:
            assert self.sched.max_num_batched_tokens >= self.model.max_seq_len, \
                "without chunked prefill, a full-length prompt must fit in one step"
        assert self.num_blocks >= self.max_blocks_per_seq + 1, \
            "pool must hold one full-length sequence plus the scratch block"