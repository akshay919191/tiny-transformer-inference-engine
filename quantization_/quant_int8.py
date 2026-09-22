import torch

QMAX = 127  


def quantize_(weights: torch.Tensor,
              quantize_per_: str,
              group_size: int = None):
    """
    Symmetric INT8 quantization.

    Args:
        weights: float tensor to quantize, shape [out_features, in_features]
        quantize_per_: "tensor" | "channel" | "group"
        group_size: required when quantize_per_ == "group"

    Returns:
        q: int8 tensor, same shape as `weights`
        scale: float32 tensor
            - "tensor":  scalar
            - "channel": shape [out_features, 1]
            - "group":   shape [out_features, in_features // group_size]
        pad: int, only meaningful for "group" (0 if in_features divided evenly).
             Needed by dequantize_ to crop the padding back off.
    """
    if quantize_per_ == "tensor":
        max_abs = torch.clamp(weights.abs().max(), min=1e-8)
        scale = max_abs / QMAX
        q = torch.clamp(torch.round(weights / scale), -QMAX, QMAX).to(torch.int8)
        return q, scale, 0

    elif quantize_per_ == "channel":
        max_abs = torch.clamp(weights.abs().max(dim=1, keepdim=True)[0], min=1e-8)
        scale = max_abs / QMAX
        q = torch.clamp(torch.round(weights / scale), -QMAX, QMAX).to(torch.int8)
        return q, scale, 0

    elif quantize_per_ == "group":
        if group_size is None:
            raise ValueError("group_size is required when quantize_per_='group'")

        out_features, in_features = weights.shape
        pad = (-in_features) % group_size  # amount needed to reach a multiple of group_size
        w = torch.nn.functional.pad(weights, (0, pad)) if pad else weights

        n_groups = w.shape[1] // group_size
        w = w.view(out_features, n_groups, group_size)

        max_abs = torch.clamp(w.abs().max(dim=-1, keepdim=True)[0], min=1e-8)
        scale = max_abs / QMAX 

        q = torch.clamp(torch.round(w / scale), -QMAX, QMAX).to(torch.int8)
        q = q.view(out_features, n_groups * group_size)  # still padded length
        scale = scale.view(out_features, n_groups)

        return q, scale, pad

    else:
        raise ValueError(f"unknown quantize_per_={quantize_per_!r}, "
                          f"expected 'tensor' | 'channel' | 'group'")


def dequantize_(q_weights: torch.Tensor,
                scale: torch.Tensor,
                quantize_per_: str,
                group_size: int = None,
                pad: int = 0):
    """
    Inverse of quantize_. Returns a float32 tensor with the original (unpadded) shape.
    """
    if quantize_per_ == "tensor":
        return q_weights.to(torch.float32) * scale

    elif quantize_per_ == "channel":
        return q_weights.to(torch.float32) * scale 

    elif quantize_per_ == "group":
        if group_size is None:
            raise ValueError("group_size is required when quantize_per_='group'")

        out_features, padded_in = q_weights.shape
        n_groups = padded_in // group_size

        w = q_weights.to(torch.float32).view(out_features, n_groups, group_size)
        s = scale.view(out_features, n_groups, 1)
        w = (w * s).view(out_features, padded_in)

        if pad:
            w = w[:, :-pad]
        return w

    else:
        raise ValueError(f"unknown quantize_per_={quantize_per_!r}, "
                          f"expected 'tensor' | 'channel' | 'group'")

