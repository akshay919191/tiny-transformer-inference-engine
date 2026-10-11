import pytest
import torch
from sampling import greedy, apply_top_k, apply_top_p, sample


def test_top_k_one_selects_highest_logit():
    logits = torch.tensor([[0.0, 8.0, 2.0]])
    assert sample(logits, top_k=1).item() == 1


def test_greedy_matches_argmax():
    logits = torch.randn(4, 100)
    assert torch.equal(greedy(logits).squeeze(-1), torch.argmax(logits, dim=-1))


def test_top_k_keeps_exactly_k_when_no_ties():
    logits = torch.arange(10, dtype=torch.float32).unsqueeze(0)
    out = apply_top_k(logits.clone(), 3)
    assert (out > float("-inf")).sum().item() == 3
    kept = out[out > float("-inf")]
    assert torch.equal(torch.sort(kept, descending=True).values,
                        torch.tensor([9., 8., 7.]))


def test_top_k_zero_is_noop():
    logits = torch.randn(2, 50)
    assert torch.equal(apply_top_k(logits.clone(), 0), logits)


def test_top_p_keeps_smallest_set_covering_p():
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    out = apply_top_p(logits.clone(), top_p=0.5)
    assert (out > float("-inf")).sum().item() == 1


def test_top_p_one_is_noop():
    logits = torch.randn(2, 50)
    assert torch.equal(apply_top_p(logits.clone(), 1.0), logits)


def test_sample_top_k_1_is_deterministic_argmax():
    torch.manual_seed(0)
    logits = torch.randn(8, 200)
    tok = sample(logits, temperature=1.0, top_k=1, top_p=1.0)
    assert torch.equal(tok.squeeze(-1), torch.argmax(logits, dim=-1))


@pytest.mark.parametrize("kwargs", [
    {"temperature": 0}, {"top_k": -1}, {"top_p": 0}, {"top_p": 1.1},
])
def test_sampling_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        sample(torch.zeros(1, 4), **kwargs)


def test_sampling_rejects_nonfinite_logits():
    with pytest.raises(ValueError, match="NaN or infinity"):
        sample(torch.tensor([[0.0, float("nan")]]))
