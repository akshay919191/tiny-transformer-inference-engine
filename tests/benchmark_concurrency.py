import torch
from serving.paged_model import load_paged_model
from serving.allocator import BlockAllocator
from serving.sequence import Sequence

dev = "cuda"
model, cfg = load_paged_model("checkpoints/ckpt_step20000.pt", dev)

block_size = 16
head_dim = cfg.d_model // cfg.num_heads
bytes_per_elem = 4  # fp32
bytes_per_block = 2 * cfg.num_layers * block_size * cfg.num_kv_heads * head_dim * bytes_per_elem
print(f"bytes/block = {bytes_per_block}")

blocks_per_full_seq = -(-cfg.max_seq_len // block_size)  # ceil
print(f"a full-length sequence needs {blocks_per_full_seq} blocks")

# pick a budget that fits a handful of FULL-LENGTH sequences under contiguous allocation —
# small enough that paging's advantage is visible, big enough to actually allocate blocks
num_full_seqs_budget = 4
num_blocks = blocks_per_full_seq * num_full_seqs_budget
print(f"num_blocks = {num_blocks}  (budget = {num_blocks * bytes_per_block / 1024:.1f} KB)")

contiguous_capacity = num_blocks // blocks_per_full_seq
print(f"contiguous: fits {contiguous_capacity} concurrent sequences (each reserves "
      f"{blocks_per_full_seq} blocks = full max_seq_len={cfg.max_seq_len})")

# ---- paged capacity: simulate real-length sequences ----
import random
random.seed(0)
lengths = [random.randint(20, 100) for _ in range(500)]
alloc = BlockAllocator(num_blocks + 1)  # +1 for the reserved scratch block

fitted = 0
seqs = []
for L in lengths:
    seq = Sequence(fitted, [0] * L, block_size)
    needed = seq.num_blocks_needed()
    if not alloc.can_allocate(needed):
        break
    seq.ensure_blocks(alloc)
    seqs.append(seq)
    fitted += 1

avg_len = sum(lengths[:fitted]) / fitted if fitted else 0
print(f"paged: fit {fitted} concurrent sequences (avg len ~{avg_len:.0f}) before running out of blocks")
print(f"paged/contiguous ratio: {fitted / max(contiguous_capacity, 1):.1f}x more concurrent requests")

for s in seqs:
    s.release(alloc)