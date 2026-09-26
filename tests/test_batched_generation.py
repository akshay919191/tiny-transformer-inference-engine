import torch

from serving.paged_model import load_paged_model
from serving.generation import generate_batched
from serving.allocator import BlockAllocator
from serving.kv_pool import KVPool


def test_batched_generation_matches_golden():
    dev = "cuda"
    model, cfg = load_paged_model("checkpoints/ckpt_step20000.pt", dev)
    golden = torch.load("tests/golden.pt")
    entries = [golden[1], golden[3], golden[7]]   # 3 different lengths, batched together

    alloc = BlockAllocator(256)
    pool = KVPool(cfg.num_layers, 16, 256, cfg.num_kv_heads,
                  cfg.d_model // cfg.num_heads, torch.float32, dev)

    prompts = [e["prompt_ids"].tolist() for e in entries]
    outputs = generate_batched(model, prompts, 50, pool, alloc)

    for i, e in enumerate(entries):
        expected = e["generated_ids"].tolist()
        got = outputs[i]
        mismatch = next((j for j, (a, b) in enumerate(zip(got, expected)) if a != b), None)
        assert mismatch is None, (
            f"seq {i} (len {len(prompts[i])}) diverges at token {mismatch}: "
            f"{got[mismatch]} vs {expected[mismatch]}"
        )