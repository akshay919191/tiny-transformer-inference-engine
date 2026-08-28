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