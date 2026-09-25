import torch

from serving.paged_model import load_paged_model
from serving.generation import prefill_seq, decode_step
from serving.allocator import BlockAllocator
from serving.kv_pool import KVPool
from serving.sequence import Sequence


def test_two_sequences_interleaved():
    dev = "cuda"
    model, cfg = load_paged_model("checkpoints/ckpt_step20000.pt", dev)
    golden = torch.load("tests/golden.pt")
    e1, e2 = golden[3], golden[7]         # pick two different lengths

    alloc = BlockAllocator(128)
    pool = KVPool(cfg.num_layers, 16, 128, cfg.num_kv_heads,
                  cfg.d_model // cfg.num_heads, torch.float32, dev)

    s1 = Sequence(1, e1["prompt_ids"].tolist(), 16)
    s2 = Sequence(2, e2["prompt_ids"].tolist(), 16)

    out1 = [prefill_seq(model, s1, pool, alloc)]
    out2 = [prefill_seq(model, s2, pool, alloc)]

    for _ in range(49):
        out1.append(decode_step(model, s1, pool, alloc, out1[-1]))
        out2.append(decode_step(model, s2, pool, alloc, out2[-1]))

    assert out1 == e1["generated_ids"].tolist()
    assert out2 == e2["generated_ids"].tolist()
    assert set(s1.block_table).isdisjoint(s2.block_table)

    s1.release(alloc)
    s2.release(alloc)
    assert alloc.num_free == alloc.num_usable