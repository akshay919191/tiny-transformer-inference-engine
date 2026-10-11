"""Single command-line entry point for text generation and checkpoint quantization."""

import argparse
from pathlib import Path

import torch
import tiktoken
import torch.nn.functional as F
from torch import nn

from kv_cache import KVCache_kv
from models.model_config import ModelConfig
from sampling import sample


def _load_checkpoint(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # Older PyTorch versions do not support weights_only.
        return torch.load(path, map_location=map_location)


def _model_config(ckpt):
    raw = ckpt.get("model_config")
    if raw is None:
        raise ValueError("checkpoint is missing model_config")
    values = dict(raw) if isinstance(raw, dict) else vars(raw)
    return ModelConfig(**{k: v for k, v in values.items()
                          if k in ModelConfig.__dataclass_fields__})


class _QuantLinear(nn.Module):
    def __init__(self, qweight, scale, meta):
        super().__init__()
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.meta = meta

    def forward(self, x):
        from quantization_.quant_int8 import dequantize_
        w = dequantize_(self.qweight, self.scale, self.meta["quantize_per_"],
                        self.meta.get("group_size"), self.meta.get("pad", 0))
        expected = self.meta["orig_shape"]
        if list(w.shape) != list(expected):
            raise ValueError(f"invalid quantized weight shape: expected {expected}, got {list(w.shape)}")
        return F.linear(x, w.to(device=x.device, dtype=x.dtype))


class _QuantEmbedding(_QuantLinear):
    def forward(self, ids):
        from quantization_.quant_int8 import dequantize_
        w = dequantize_(self.qweight, self.scale, self.meta["quantize_per_"],
                        self.meta.get("group_size"), self.meta.get("pad", 0))
        if list(w.shape) != list(self.meta["orig_shape"]):
            raise ValueError("invalid quantized embedding shape")
        return F.embedding(ids, w.to(device=ids.device,
                                     dtype=getattr(self, "activation_dtype", torch.float32)))


def _set_module(model, path, module):
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], module)


def load_model(ckpt_path, device="cpu", *, attn_type=None, backend=None,
               causal=None, quantized=False):
    device = torch.device(device)
    ckpt = _load_checkpoint(ckpt_path)
    cfg = _model_config(ckpt)
    if causal is not None:
        cfg.causal = causal
    train_cfg = ckpt.get("train_config", {})
    attn_type = attn_type or train_cfg.get("attn_type", "mqa")
    requested_backend = backend if backend is not None else train_cfg.get("backend", "pytorch")
    if device.type != "cuda" and backend is None:
        requested_backend = "pytorch"
    if device.type != "cuda" and requested_backend == "cuda":
        raise ValueError("the CUDA backend requires a CUDA device")

    from kernels.capability import resolve_backend
    from models.transformer_block import Transformer
    backend = resolve_backend(requested_backend, cfg, attn_type)
    model = Transformer(cfg, attn_type=attn_type, backend=backend)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}

    if quantized or "quant_metadata" in ckpt:
        metadata = ckpt.get("quant_metadata")
        if not metadata:
            raise ValueError("--quantized requested, but checkpoint has no quant_metadata")
        for weight_name, meta in metadata.items():
            prefix = weight_name[:-len(".weight")]
            qweight, scale = state.pop(f"{weight_name}.qweight", None), state.pop(f"{weight_name}.scale", None)
            if qweight is None or scale is None:
                raise ValueError(f"quantized checkpoint is missing tensors for {weight_name}")
            module = _QuantEmbedding(qweight, scale, meta) if meta.get("module_type") == "embedding" else _QuantLinear(qweight, scale, meta)
            _set_module(model, prefix, module)
        missing, unexpected = model.load_state_dict(state, strict=False)
        expected_missing = {f"{name[:-len('.weight')]}.{key}"
                            for name in metadata for key in ("qweight", "scale")}
        if set(missing) != expected_missing or unexpected:
            raise ValueError(
                "quantized checkpoint does not match the model structure "
                f"(missing={missing}, unexpected={unexpected})"
            )
    else:
        model.load_state_dict(state, strict=True)

    dtype = torch.float16 if device.type == "cuda" and backend == "cuda" else None
    if dtype:
        model = model.to(device=device, dtype=dtype)
    else:
        model = model.to(device=device)
    for module in model.modules():
        if isinstance(module, _QuantEmbedding):
            module.activation_dtype = next(model.parameters()).dtype
    model.eval()
    return model, cfg


@torch.inference_mode()
def generate_ids(model, cfg, prompt_ids, max_new_tokens, *, temperature=1.0,
                 top_k=0, top_p=1.0):
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    if max_new_tokens == 0:
        return []
    if not prompt_ids:
        raise ValueError("prompt must encode to at least one token")
    if len(prompt_ids) > cfg.max_seq_len:
        raise ValueError(f"prompt has {len(prompt_ids)} tokens; model limit is {cfg.max_seq_len}")
    if len(prompt_ids) + max_new_tokens > cfg.max_seq_len:
        raise ValueError("prompt plus requested generation exceeds max_seq_len")
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cache = KVCache_kv(
        num_layers=cfg.num_layers, batch_size=1,
        num_heads=getattr(cfg, "num_kv_heads", cfg.num_heads),
        max_seq_len=cfg.max_seq_len, head_dim=cfg.head_dim,
        dtype=next(model.parameters()).dtype, device=device,
    )
    logits = model(ids, kv_cache=cache)[:, -1, :]
    generated = []
    for step in range(max_new_tokens):
        token = int(sample(logits, temperature, top_k, top_p).item())
        generated.append(token)
        if token == cfg.eos_token_id or step + 1 == max_new_tokens:
            break
        logits = model(torch.tensor([[token]], device=device), kv_cache=cache)[:, -1, :]
    return generated


def quantize_checkpoint(source, destination, mode="channel", group_size=None):
    from quantization_.quant_int8 import quantize_
    if mode not in ("tensor", "channel", "group"):
        raise ValueError("mode must be tensor, channel, or group")
    if mode == "group" and (group_size is None or group_size <= 0):
        raise ValueError("group_size must be positive for group quantization")
    ckpt = _load_checkpoint(source)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model, _ = load_model(source, "cpu")
    modules = {}
    tied = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            key = f"{name}.weight"
            ptr = module.weight.data_ptr()
            modules[key] = ("embedding" if isinstance(module, nn.Embedding) else "linear", ptr)
            tied.setdefault(ptr, key)
    quant_state, metadata = {}, {}
    for name, tensor in state.items():
        kind_ptr = modules.get(name)
        if tensor.ndim == 2 and kind_ptr and tied[kind_ptr[1]] == name:
            q, scale, pad = quantize_(tensor.float(), mode, group_size)
            quant_state[f"{name}.qweight"] = q
            quant_state[f"{name}.scale"] = scale
            metadata[name] = {"quantize_per_": mode, "group_size": group_size,
                              "pad": pad, "orig_shape": list(tensor.shape),
                              "module_type": kind_ptr[0]}
        else:
            quant_state[name] = tensor
    output = {"model": quant_state, "quant_metadata": metadata,
              "model_config": ckpt["model_config"],
              "train_config": ckpt.get("train_config", {})}
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, destination)


def _run_serving(args, device, tokenizer):
    if args.quantized:
        raise ValueError("paged serving currently supports standard checkpoints only")
    if args.attn_type not in (None, "mqa"):
        raise ValueError("paged serving supports MQA attention only")
    from serving.allocator import BlockAllocator
    from serving.batch import build_batch_step
    from serving.generation import generate_batched_text
    from serving.kv_pool import KVPool
    from serving.paged_model import load_paged_model

    model, cfg = load_paged_model(args.ckpt, device)
    block_size = args.block_size
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if args.num_gpu_blocks <= 0:
        raise ValueError("num_gpu_blocks must be positive")
    alloc = BlockAllocator(args.num_gpu_blocks)
    pool = KVPool(cfg.num_layers, block_size, args.num_gpu_blocks,
                  cfg.num_kv_heads, cfg.head_dim, next(model.parameters()).dtype, device)
    prompts = args.prompts or [args.prompt]
    output_ids = generate_batched_text(
        model, cfg, prompts, args.tokens, cfg.eos_token_id, pool, alloc,
        args.temperature, args.top_k, args.top_p, block_size,
    )
    for prompt, ids in zip(prompts, output_ids):
        print(f"{prompt}{tokenizer.decode(ids)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="checkpoints/ckpt_step20000.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--prompts", nargs="*")
    parser.add_argument("--tokens", "--token", dest="tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", "--top_k", type=int, default=0)
    parser.add_argument("--top-p", "--top_p", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-type", choices=("mqa", "mha"))
    parser.add_argument("--backend", choices=("cuda", "pytorch", "auto"))
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--quantized", action="store_true", help="load an INT8 checkpoint")
    parser.add_argument("--quantize", metavar="OUTPUT", help="quantize --ckpt and save to OUTPUT")
    parser.add_argument("--quant-mode", choices=("tensor", "channel", "group"), default="channel")
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--serving", action="store_true", help="run paged batched serving")
    parser.add_argument("--num-gpu-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()
    if args.tokens < 0 or args.temperature <= 0 or args.top_k < 0 or not 0 < args.top_p <= 1:
        parser.error("tokens/top-k must be nonnegative, temperature positive, and top-p in (0, 1]")
    if args.quantize:
        if args.quant_mode == "group" and (args.group_size is None or args.group_size <= 0):
            parser.error("--group-size must be positive when --quant-mode=group")
        quantize_checkpoint(args.ckpt, args.quantize, args.quant_mode, args.group_size)
        print(f"Saved quantized checkpoint to {args.quantize}")
        return
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    tokenizer = tiktoken.get_encoding("gpt2")
    if args.serving:
        _run_serving(args, device, tokenizer)
        return
    model, cfg = load_model(args.ckpt, device, attn_type=args.attn_type,
                            backend=args.backend, causal=args.causal,
                            quantized=args.quantized)
    ids = tokenizer.encode_ordinary(args.prompt)
    generated = generate_ids(model, cfg, ids, args.tokens,
                             temperature=args.temperature, top_k=args.top_k,
                             top_p=args.top_p)
    print(args.prompt + tokenizer.decode(generated))


if __name__ == "__main__":
    main()
