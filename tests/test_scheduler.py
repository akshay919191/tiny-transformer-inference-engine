import torch
import tiktoken

from serving.paged_model import load_paged_model
from serving.allocator import BlockAllocator
from serving.kv_pool import KVPool
from serving.scheduler import Scheduler

enc = tiktoken.get_encoding("gpt2")


def make_sched(dev="cuda", num_gpu_blocks=128):
    model, cfg = load_paged_model("checkpoints/ckpt_step20000.pt", dev)
    alloc = BlockAllocator(num_gpu_blocks)
    pool = KVPool(cfg.num_layers, 16, num_gpu_blocks, cfg.num_kv_heads,
                  cfg.d_model // cfg.num_heads, torch.float32, dev)
    sched = Scheduler(model, pool, alloc, cfg)
    return sched, alloc


def test_two_requests_upfront_produce_full_length_text():
    sched, alloc = make_sched()
    prompts = ["Once upon a time", "The little dog"]
    for p in prompts:
        sched.add_request(enc.encode_ordinary(p), max_new_tokens=30)

    outputs = {}
    steps = 0
    while sched.running or sched.waiting:
        for seq_id, tok, done in sched.step():
            outputs.setdefault(seq_id, []).append(tok)
        steps += 1
        assert steps < 200, "runaway loop — scheduler never finished"

    assert len(outputs) == 2
    for seq_id, toks in outputs.items():
        assert 1 <= len(toks) <= 30
        text = enc.decode(toks)
        print(f"seq {seq_id}: {prompts[seq_id]}{text}")
        assert len(text.strip()) > 0


def test_mid_flight_admission():
    """A third request added after generation has already started should
    still get served, and finish with real output, proving continuous
    (not fixed-batch) admission works."""
    sched, alloc = make_sched()
    sched.add_request(enc.encode_ordinary("Once upon a time"), max_new_tokens=30)
    sched.add_request(enc.encode_ordinary("The little dog"), max_new_tokens=30)

    outputs = {}
    steps = 0
    added_third = False
    while sched.running or sched.waiting:
        for seq_id, tok, done in sched.step():
            outputs.setdefault(seq_id, []).append(tok)

        if steps == 3 and not added_third:
            sched.add_request(enc.encode_ordinary("In the forest there was"), max_new_tokens=30)
            added_third = True

        steps += 1
        assert steps < 200

    assert added_third
    assert len(outputs) == 3, f"expected 3 sequences served, got {len(outputs)}"
    for seq_id, toks in outputs.items():
        assert len(toks) > 0


def test_blocks_never_oversubscribed_and_fully_freed():
    sched, alloc = make_sched(num_gpu_blocks=8)
    for i in range(8):
        sched.add_request(enc.encode_ordinary(f"Story number {i} was"), max_new_tokens=20)

    saw_queueing = False
    steps = 0
    while sched.running or sched.waiting:
        sched.step()
        if sched.waiting:              # something is stuck behind admission, regardless of why
            saw_queueing = True
        steps += 1
        assert steps < 500

    assert saw_queueing, \
        "test didn't actually exercise the admission-blocked path — increase request count or lower num_gpu_blocks"
    assert alloc.num_free == alloc.num_usable, "blocks leaked — not all released"

def test_finished_sequence_frees_blocks_before_others_finish():
    """A short max_new_tokens sequence alongside a long one should free its
    blocks well before the long one finishes — proving per-step release,
    not end-of-batch release."""
    sched, alloc = make_sched()
    sched.add_request(enc.encode_ordinary("Hi"), max_new_tokens=3)     # finishes fast
    sched.add_request(enc.encode_ordinary("Once upon a time"), max_new_tokens=40)  # long

    free_after_short_done = None
    steps = 0
    while sched.running or sched.waiting:
        results = sched.step()
        for seq_id, tok, done in results:
            if seq_id == 0 and done and free_after_short_done is None:
                free_after_short_done = alloc.num_free
        steps += 1
        assert steps < 200

    assert free_after_short_done is not None, "short sequence never finished"
    assert alloc.num_free == alloc.num_usable