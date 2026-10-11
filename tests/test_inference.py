from types import SimpleNamespace

import pytest
import torch

from inference import generate_ids


class FixedLogitModel(torch.nn.Module):
    def __init__(self, token_id, vocab_size=8):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.token_id = token_id
        self.vocab_size = vocab_size

    def forward(self, input_ids, kv_cache=None):
        logits = torch.full((*input_ids.shape, self.vocab_size), -100.0,
                            device=input_ids.device)
        logits[..., self.token_id] = 100.0
        return logits


def config(max_seq_len=8, eos_token_id=7):
    return SimpleNamespace(num_layers=1, num_heads=1, num_kv_heads=1,
                           head_dim=4, max_seq_len=max_seq_len,
                           eos_token_id=eos_token_id)


def test_generation_stops_at_eos():
    model = FixedLogitModel(token_id=7)
    assert generate_ids(model, config(), [1, 2], 5, temperature=1,
                       top_k=1) == [7]


def test_generation_respects_requested_length():
    model = FixedLogitModel(token_id=3)
    assert generate_ids(model, config(), [1], 3, temperature=1,
                       top_k=1) == [3, 3, 3]


@pytest.mark.parametrize("prompt, count", [([], 1), ([1, 2], 7)])
def test_generation_rejects_empty_or_overlong_context(prompt, count):
    with pytest.raises(ValueError):
        generate_ids(FixedLogitModel(1), config(), prompt, count)


def test_zero_generation_does_not_need_a_valid_prompt():
    assert generate_ids(FixedLogitModel(1), config(), [], 0) == []
