import pytest
import torch

from quantization_.quant_int8 import dequantize_, quantize_


@pytest.mark.parametrize("mode, group_size", [
    ("tensor", None), ("channel", None), ("group", 4),
])
def test_quantize_dequantize_is_finite_and_shape_preserving(mode, group_size):
    weights = torch.randn(5, 7)
    q, scale, pad = quantize_(weights, mode, group_size)
    restored = dequantize_(q, scale, mode, group_size, pad)
    assert q.dtype == torch.int8
    assert restored.shape == weights.shape
    assert torch.isfinite(restored).all()
    assert (restored - weights).abs().max() < 0.1


@pytest.mark.parametrize("weights", [torch.empty(0, 3), torch.tensor([[float("nan")]])])
def test_quantization_rejects_empty_or_nonfinite_weights(weights):
    with pytest.raises(ValueError):
        quantize_(weights, "channel")


def test_group_quantization_requires_positive_group_size():
    with pytest.raises(ValueError, match="group_size"):
        quantize_(torch.ones(2, 3), "group", 0)
