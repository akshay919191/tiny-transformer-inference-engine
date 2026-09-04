import torch
import torch.nn as nn


class KVCache(nn.Module):

    def __init__(
        self,
        num_layers,
        batch_size,
        num_heads,
        max_seq_len,
        head_dim,
        dtype,
        device,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.length = 0

        self.kcache = [
            torch.empty(
                batch_size,
                num_heads,
                max_seq_len,
                head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

        self.vcache = [
            torch.empty(
                batch_size,
                num_heads,
                max_seq_len,
                head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

    def update(self, layer_idx, k, v):

        seq_len = k.shape[2]

        start = self.length
        end = start + seq_len

        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: "
                f"{end} > {self.max_seq_len}"
            )

        self.kcache[layer_idx][
            :, :, start:end, :
        ] = k

        self.vcache[layer_idx][
            :, :, start:end, :
        ] = v

        return (
            self.kcache[layer_idx][:, :, :end, :],
            self.vcache[layer_idx][:, :, :end, :],
        )

    def get(self, layer_idx):

        return (
            self.kcache[layer_idx][
                :, :, :self.length, :
            ],
            self.vcache[layer_idx][
                :, :, :self.length, :
            ],
        )

    def advance(self, num_tokens):

        self.length += num_tokens

        if self.length > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: "
                f"{self.length} > {self.max_seq_len}"
            )

    def reset(self):

        self.length = 0



class KVCache_kv(nn.Module):

    def __init__(
        self,
        num_layers,
        batch_size,
        num_heads,
        max_seq_len,
        head_dim,
        dtype,
        device,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.length = 0

        self.kcache = [
            torch.empty(
                batch_size,
                num_heads,
                max_seq_len,
                head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

        self.vcache = [
            torch.empty(
                batch_size,
                num_heads,
                max_seq_len,
                head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

    def update(self, layer_idx, k, v):
        if not (0 <= layer_idx < self.num_layers):
            raise ValueError(f"layer_idx={layer_idx} out of range [0, {self.num_layers})")

        T_new = k.shape[2]
        if T_new <= 0:
            raise ValueError("Cannot update KV cache with zero or negative sequence length")

        expected_shape = self.kcache[layer_idx].shape  # [B, H, max_seq_len, D]

        if k.shape[0] != expected_shape[0]:
            raise ValueError(f"Batch size mismatch: got {k.shape[0]}, expected {expected_shape[0]}")
        if k.shape[1] != expected_shape[1]:
            raise ValueError(f"Num KV heads mismatch: got {k.shape[1]}, expected {expected_shape[1]}")
        if k.shape[3] != expected_shape[3]:
            raise ValueError(f"Head dim mismatch: got {k.shape[3]}, expected {expected_shape[3]}")
        if k.dtype != self.kcache[layer_idx].dtype:
            raise ValueError(f"Dtype mismatch: got {k.dtype}, expected {self.kcache[layer_idx].dtype}")
        if k.device != self.kcache[layer_idx].device:
            raise ValueError(f"Device mismatch: got {k.device}, expected {self.kcache[layer_idx].device}")
        if v.shape != k.shape:
            raise ValueError(f"k and v shape mismatch: k={k.shape}, v={v.shape}")

        start = self.length
        end = start + T_new
        if end > self.max_seq_len:
            raise ValueError(f"KV cache overflow: end={end}, max_seq_len={self.max_seq_len}")

        self.kcache[layer_idx][:, :, start:end, :] = k
        self.vcache[layer_idx][:, :, start:end, :] = v

        return (
            self.kcache[layer_idx][:, :, :end, :],
            self.vcache[layer_idx][:, :, :end, :],
        )

    def get(self, layer_idx):

        return (
            self.kcache[layer_idx][
                :, :, :self.length, :
            ],
            self.vcache[layer_idx][
                :, :, :self.length, :
            ],
        )

    def advance(self, num_tokens):

        self.length += num_tokens

        if self.length > self.max_seq_len:
            raise RuntimeError(
                f"KV cache overflow: "
                f"{self.length} > {self.max_seq_len}"
            )

    def reset(self):

        self.length = 0