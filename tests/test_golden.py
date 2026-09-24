import torch
from serving.paged_model import load_paged_model
from serving.generation import generate_paged
from serving.allocator import BlockAllocator
from serving.kv_pool import KVPool

def test_golden():
    dev = "cuda"
    model, cfg = load_paged_model("checkpoints/ckpt_step20000.pt", dev)   # same ckpt as the golden script
    golden = torch.load("tests/golden.pt")
    for entry in golden:
        alloc = BlockAllocator(128)
        pool = KVPool(cfg.num_layers, 16, 128, cfg.num_kv_heads,
                      cfg.d_model // cfg.num_heads, torch.float32, dev)
        out = generate_paged(model, entry["prompt_ids"].tolist(), 50, pool, alloc)
        assert out == entry["generated_ids"].tolist()