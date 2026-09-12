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

        self.dtype = torch.float16

        self.kcache = [
            torch.empty(
                batch_size,
                num_heads,
                max_seq_len,
                head_dim,
                dtype=self.dtype,
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
                dtype=self.dtype,
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

        k_half = k.half() if k.dtype != self.dtype else k
        v_half = v.half() if v.dtype != self.dtype else v

        self.kcache[layer_idx][
            :, :, start:end, :
        ] = k_half

        self.vcache[layer_idx][
            :, :, start:end, :
        ] = v_half

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
        dtype=torch.float16,
        device="cuda",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.length = 0
        self.dtype = dtype

        self.kcache = [
            torch.empty(
                batch_size,
                num_heads,
                max_seq_len,
                head_dim,
                dtype=self.dtype,
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
                dtype=self.dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

    def update(self, layer_idx, k, v):
        if not (0 <= layer_idx < self.num_layers):
            raise ValueError(f"layer_idx={layer_idx} out of range [0, {self.num_layers})")

        seq_len = k.shape[2]
        end = self.length + seq_len

        if end > self.max_seq_len:
            raise ValueError("KV cache overflow")
        if k.shape[3] != self.kcache[0].shape[3]:
            raise ValueError("Head dim mismatch")
        if k.device != self.kcache[0].device:
            raise ValueError("Device mismatch")

        self.kcache[layer_idx][:, :, self.length:end, :] = k
        self.vcache[layer_idx][:, :, self.length:end, :] = v

        return (
            self.kcache[layer_idx][:, :, :end, :],
            self.vcache[layer_idx][:, :, :end, :],
        )

    def get(self, layer_idx):
        return (
            self.kcache[layer_idx][:, :, :self.length, :],
            self.vcache[layer_idx][:, :, :self.length, :],
        )

    def advance(self, num_tokens=1):
        self.length += num_tokens
        if self.length > self.max_seq_len:
            raise RuntimeError(f"KV cache overflow: {self.length} > {self.max_seq_len}")

    def reset(self):
        self.length = 0