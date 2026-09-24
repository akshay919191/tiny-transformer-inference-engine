import pytest
from serving.allocator import BlockAllocator

def test_alloc_all_then_free_all():
    a = BlockAllocator(8)
    blocks = a.allocate_n(a.num_usable)
    assert a.num_free == 0 and 0 not in blocks and len(set(blocks)) == 7
    a.free(blocks)
    assert a.num_free == 7

def test_out_of_blocks():
    a = BlockAllocator(4)
    assert not a.can_allocate(4)
    a.allocate_n(3)
    with pytest.raises(RuntimeError):
        a.allocate()

def test_double_free_and_block_zero():
    a = BlockAllocator(4)
    b = a.allocate()
    a.free([b])
    with pytest.raises(ValueError):
        a.free([b])
    with pytest.raises(ValueError):
        a.free([0])

def test_failed_free_changes_nothing():
    a = BlockAllocator(4)
    b = a.allocate()
    before = a.num_free
    with pytest.raises(ValueError):
        a.free([b, 0])
    assert a.num_free == before