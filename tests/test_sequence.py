import pytest
from serving.allocator import BlockAllocator
from serving.sequence import Sequence, blocks_for

@pytest.mark.parametrize("n,expected", [(0, 0), (1, 1), (15, 1), (16, 1), (17, 2), (512, 32)])
def test_blocks_for(n, expected):
    assert blocks_for(n, 16) == expected

def test_growth_at_boundary():
    a = BlockAllocator(10)
    s = Sequence(0, list(range(16)), block_size=16)
    s.ensure_blocks(a)
    assert len(s.block_table) == 1
    s.append_token(99)                      # token 17
    assert s.num_new_blocks_needed() == 1
    s.ensure_blocks(a)
    assert len(s.block_table) == 2

def test_release_and_no_shared_blocks():
    a = BlockAllocator(10)
    s1 = Sequence(1, list(range(20)), 16); s1.ensure_blocks(a)
    s2 = Sequence(2, list(range(20)), 16); s2.ensure_blocks(a)
    assert not set(s1.block_table) & set(s2.block_table)
    s1.release(a); s2.release(a)
    assert a.num_free == a.num_usable