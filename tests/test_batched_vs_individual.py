import torch
from serving.paged_model import load_paged_model
from serving.batch import build_batch_step
from serving.allocator import BlockAllocator
from serving.kv_pool import KVPool
from serving.sequence import Sequence
from serving.generation import prefill_seq, decode_step


def test_batched_matches_individual():
    dev = "cuda"
    model, cfg = load_paged_model("checkpoints/ckpt_step20000.pt", dev)
    golden = torch.load("tests/golden.pt")
    e1, e2 = golden[1], golden[3]   # short prompt, longer prompt — different lengths on purpose

    def make_pool_alloc():
        alloc = BlockAllocator(128)
        pool = KVPool(cfg.num_layers, 16, 128, cfg.num_kv_heads,
                      cfg.d_model // cfg.num_heads, torch.float32, dev)
        return pool, alloc

    # --- reference: run each sequence individually through the already-verified path ---
    pool_ref, alloc_ref = make_pool_alloc()
    s1_ref = Sequence(1, e1["prompt_ids"].tolist(), 16)
    s2_ref = Sequence(2, e2["prompt_ids"].tolist(), 16)
    t1_ref = prefill_seq(model, s1_ref, pool_ref, alloc_ref)
    t2_ref = prefill_seq(model, s2_ref, pool_ref, alloc_ref)
    # one more decode step each, individually
    t1_ref2 = decode_step(model, s1_ref, pool_ref, alloc_ref, t1_ref)
    t2_ref2 = decode_step(model, s2_ref, pool_ref, alloc_ref, t2_ref)

    # --- batched: same two sequences, fresh pool, run together ---
    pool_b, alloc_b = make_pool_alloc()
    s1 = Sequence(1, e1["prompt_ids"].tolist(), 16)
    s2 = Sequence(2, e2["prompt_ids"].tolist(), 16)
    s1.ensure_blocks(alloc_b)
    s2.ensure_blocks(alloc_b)

    # step 1: both prefill together
    batch = build_batch_step([(s1, s1.token_ids), (s2, s2.token_ids)], dev)
    logits = model.forward_batch(batch, pool_b)
    tok1 = int(logits[0].argmax())
    tok2 = int(logits[1].argmax())
    assert tok1 == t1_ref and tok2 == t2_ref, f"prefill mismatch: {tok1},{tok2} vs {t1_ref},{t2_ref}"

    # step 2: both decode one token together
    s1.append_token(tok1); s1.ensure_blocks(alloc_b)
    s2.append_token(tok2); s2.ensure_blocks(alloc_b)
    batch2 = build_batch_step([(s1, [tok1]), (s2, [tok2])], dev)
    logits2 = model.forward_batch(batch2, pool_b)
    tok1_2 = int(logits2[0].argmax())
    tok2_2 = int(logits2[1].argmax())
    assert tok1_2 == t1_ref2 and tok2_2 == t2_ref2, f"decode mismatch: {tok1_2},{tok2_2} vs {t1_ref2},{t2_ref2}"