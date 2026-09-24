# test_kv_pool.py
import torch
from serving.kv_pool import KVPool  # adjust import to your filename/class name

def make_tiny_pool():
    return KVPool(
        num_layers=2,
        block_size=16,
        num_blocks=16,       # num_gpu_blocks=16, avoids the default 2GB budget
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )

def test_shapes():
    pool = make_tiny_pool()
    print("kv shape:", pool.kv.shape)
    print("flat_k[0] shape:", pool.flat_k[0].shape)
    print("flat_k[1] shape:", pool.flat_k[1].shape)
    assert pool.kv.shape == (2, 2, 16, 16, 2, 4)
    assert pool.flat_k[0].shape == (16 * 16, 2, 4)
    print("shape test passed\n")

def test_slot_mapping():
    pool = make_tiny_pool()
    # matches the worked example from earlier: block_table=[7,3], start=14, n=4
    slots = pool.make_slot_mapping(start=14, n=4, block_table=[7, 3])
    print("slots:", slots.tolist())
    assert slots.tolist() == [126, 127, 48, 49]
    print("slot_mapping test passed\n")

def test_roundtrip():
    pool = make_tiny_pool()
    layer = 0
    block_table = [9, 3, 12]   # scattered on purpose, not [1,2,3] — see earlier reasoning
    seq_len = 40               # spans multiple blocks (block_size=16)

    slot_mapping = pool.make_slot_mapping(start=0, n=seq_len, block_table=block_table)

    k = torch.randn(seq_len, pool.num_kv_heads, pool.head_dim)
    v = torch.randn(seq_len, pool.num_kv_heads, pool.head_dim)

    pool.write(layer, slot_mapping, k, v)

    k_out, v_out = pool.gather(layer, block_table, seq_len)

    print("k_out shape:", k_out.shape, "expected:", k.shape)
    assert torch.equal(k, k_out), "K round-trip mismatch"
    assert torch.equal(v, v_out), "V round-trip mismatch"
    print("round-trip test passed\n")

def test_untouched_block_zero():
    # zeros-not-empty check: if we never write to block 0, it should stay all zero
    pool = make_tiny_pool()
    layer = 0
    block_table = [5, 9]  # deliberately excludes block 0
    seq_len = 20

    slot_mapping = pool.make_slot_mapping(start=0, n=seq_len, block_table=block_table)
    k = torch.randn(seq_len, pool.num_kv_heads, pool.head_dim)
    v = torch.randn(seq_len, pool.num_kv_heads, pool.head_dim)
    pool.write(layer, slot_mapping, k, v)

    block0 = pool.kv[layer, 0, 0]  # layer 0, K, physical block 0
    assert torch.all(block0 == 0), "block 0 was touched but shouldn't have been"
    print("untouched-block-zero test passed\n")

if __name__ == "__main__":
    test_shapes()
    test_slot_mapping()
    test_roundtrip()
    test_untouched_block_zero()
    print("All tests passed.")