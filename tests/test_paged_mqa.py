import pytest, torch
from types import SimpleNamespace
from kv_cache import KVCache
from serving.allocator import BlockAllocator
from serving.sequence import Sequence
from serving.kv_pool import KVPool
from serving.paged_mqa import MQA_Paged

@pytest.mark.parametrize("prompt_len", [1, 16, 17, 20])
def test_paged_matches_contiguous(prompt_len):
    torch.manual_seed(0)
    dev = "cuda"
    cfg = SimpleNamespace(d_model=256, num_heads=8, num_kv_heads=1,
                          bias=False, dropout=0.0, max_seq_len=512)
    mqa = MQA_Paged(cfg, backend="pytorch").to(dev).eval()

    alloc = BlockAllocator(64)
    pool  = KVPool(1, 16, 64, 1, 32, torch.float32, dev)
    cache = KVCache(1, 1, 1, 512, 32, torch.float32, dev)
    seq   = Sequence(0, [0] * prompt_len, 16)     
    seq.ensure_blocks(alloc)

    with torch.no_grad():
        # prefill
        x = torch.randn(1, prompt_len, 256, device=dev)
        ref = mqa(x, x, x, kv_cache=cache, layer_idx=0); cache.advance(prompt_len)
        out = mqa.forward_paged(x, pool, 0, seq.block_table, start=0)
        assert torch.allclose(out, ref, atol=1e-5)

        # 20 decode steps
        for _ in range(20):
            x = torch.randn(1, 1, 256, device=dev)
            seq.append_token(0); seq.ensure_blocks(alloc)
            start = seq.num_tokens - 1
            ref = mqa(x, x, x, kv_cache=cache, layer_idx=0); cache.advance(1)
            out = mqa.forward_paged(x, pool, 0, seq.block_table, start=start)
            assert torch.allclose(out, ref, atol=1e-5)

def test_two_interleaved_sequences():
    torch.manual_seed(0)
    dev = "cuda"
    cfg = SimpleNamespace(d_model=256, num_heads=8, num_kv_heads=1,
                          bias=False, dropout=0.0, max_seq_len=512)
    mqa = MQA_Paged(cfg, backend="pytorch").to(dev).eval()

    alloc = BlockAllocator(64)
    pool  = KVPool(1, 16, 64, 1, 32, torch.float32, dev)
    seqs   = [Sequence(i, [0] * 16, 16) for i in range(2)]
    caches = [KVCache(1, 1, 1, 512, 32, torch.float32, dev) for _ in range(2)]
    for s in seqs:
        s.ensure_blocks(alloc)                      # A -> [1], B -> [2]

    with torch.no_grad():
        for s, c in zip(seqs, caches):              # prefill both
            x = torch.randn(1, 16, 256, device=dev)
            ref = mqa(x, x, x, kv_cache=c, layer_idx=0); c.advance(16)
            out = mqa.forward_paged(x, pool, 0, s.block_table, start=0)
            assert torch.allclose(out, ref, atol=1e-5)

        for _ in range(40):                         # decode alternately
            for s, c in zip(seqs, caches):
                x = torch.randn(1, 1, 256, device=dev)
                s.append_token(0); s.ensure_blocks(alloc)
                start = s.num_tokens - 1
                ref = mqa(x, x, x, kv_cache=c, layer_idx=0); c.advance(1)
                out = mqa.forward_paged(x, pool, 0, s.block_table, start=start)
                assert torch.allclose(out, ref, atol=1e-5)

    a, b = seqs
    assert set(a.block_table).isdisjoint(b.block_table)
    assert any(y - x != 1 for x, y in zip(a.block_table, a.block_table[1:]))   # really scattered