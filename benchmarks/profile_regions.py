# benchmarks/profile_regions.py
# Run with:  python -m benchmarks.profile_regions --batch 32
#            python -m benchmarks.profile_regions --batch 32 --prefix rope_
import argparse
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

# Ensure project root is in sys.path BEFORE importing custom modules
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kv_cache import KVCache_kv
from models.model_config import ModelConfig
from models.transformer_block import Transformer

# classes whose defining module has a PROFILE_REGIONS flag
LABELED_CLASSES = ("SwiGLU", "MQA_Cached", "MQA")


def build_model(cfg, state_dict, backend, attn_type, device):
    model = Transformer(cfg, attn_type=attn_type, backend=backend).to(device)
    model = model.to(device=device, dtype=torch.float16)
    unwrapped_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(unwrapped_state_dict)
    model.eval()
    return model


def make_cache_fn(model, cfg, batch_size, device):
    head_dim = cfg.d_model // cfg.num_heads

    def make():
        return KVCache_kv(
            num_layers=cfg.num_layers,
            batch_size=batch_size,
            max_seq_len=cfg.max_seq_len,
            num_heads=cfg.num_kv_heads,
            head_dim=head_dim,
            device=device,
            dtype=next(model.parameters()).dtype,
        )

    return make


@torch.inference_mode()
def prefill(model, tokens, kv_cache):
    logits = model(tokens, kv_cache=kv_cache)
    return logits[:, -1, :]


@torch.inference_mode()
def decode_one(model, next_token, kv_cache):
    logits = model(next_token, kv_cache=kv_cache)
    return logits[:, -1, :]


def labeled_modules(model):
    """Python modules (files) that define the labeled classes and have the flag."""
    mods = {}
    for m in model.modules():
        if type(m).__name__ in LABELED_CLASSES:
            mod = sys.modules[type(m).__module__]
            if hasattr(mod, "PROFILE_REGIONS"):
                mods[mod.__name__] = mod
    return list(mods.values())


def dev_us(e):  # attribute name differs across PyTorch versions
    return getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)


def collect(prof, prefixes):
    """{region: {'kernel': (us_total, calls), 'span': (us_total, calls)}}

    DeviceType.CPU rows carry the summed kernel time launched inside the range;
    DeviceType.CUDA rows are the GPU-timeline span (kernels + idle gaps).
    """
    out = {}
    for e in prof.key_averages():
        if not e.key.startswith(prefixes):
            continue
        dt = str(getattr(e, "device_type", "?"))
        kind = "span" if "CUDA" in dt else "kernel"
        out.setdefault(e.key, {})[kind] = (dev_us(e), e.count)
    return out


def main():
    p = argparse.ArgumentParser(description="Profile labeled regions during decode")
    p.add_argument("--ckpt", default="checkpoints/ckpt_final.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--prompt_len", type=int, default=100)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--backend", default=None, help="override backend (cuda / pytorch)")
    p.add_argument("--prefix", nargs="+", default=["mlp_", "rope_"],
                   help="region name prefixes to report")
    p.add_argument("--trace", default=None, help="optional path to write a chrome trace json")
    args = p.parse_args()

    assert args.device == "cuda" and torch.cuda.is_available(), "needs a GPU"
    device = args.device

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ModelConfig()
    for k, v in ckpt["model_config"].items():
        setattr(cfg, k, v)

    tcfg = ckpt.get("train_config", {})
    attn_type = tcfg.get("attn_type", "mqa")
    backend = args.backend or tcfg.get("backend", "pytorch")

    assert args.prompt_len + args.warmup + args.steps <= cfg.max_seq_len, "exceeds max_seq_len"

    model = build_model(cfg, ckpt["model"], backend, attn_type, device)
    mods = labeled_modules(model)
    assert mods, "no module with PROFILE_REGIONS found in the model"

    tokens = torch.randint(0, cfg.vocab_size, (args.batch, args.prompt_len), device=device)
    cache = make_cache_fn(model, cfg, args.batch, device)()
    tok = prefill(model, tokens, cache).argmax(-1, keepdim=True)

    # warmup (labels off) so first-call overhead isn't in the profile
    for _ in range(args.warmup):
        tok = decode_one(model, tok, cache).argmax(-1, keepdim=True)
    torch.cuda.synchronize()

    for mod in mods:
        mod.PROFILE_REGIONS = True
    with torch.inference_mode():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.steps):
                tok = decode_one(model, tok, cache).argmax(-1, keepdim=True)
            torch.cuda.synchronize()
    for mod in mods:
        mod.PROFILE_REGIONS = False

    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"wrote chrome trace to {args.trace}")

    print(f"\nbackend={backend} | batch {args.batch} | cache~{args.prompt_len + args.warmup} "
          f"| {args.steps} decode steps | {cfg.num_layers} layers (eager)")

    prefixes = tuple(args.prefix)
    regions = collect(prof, prefixes)
    if not regions:
        print("no matching ranges found; are the labels wired into the modules this model uses?")
        return

    for prefix in prefixes:
        names = sorted(
            (n for n in regions if n.startswith(prefix)),
            key=lambda n: -regions[n].get("kernel", (0, 0))[0],
        )
        if not names:
            continue
        print(f"\n--- {prefix}* regions ---")
        print(f"{'region':22s} {'calls':>5s} | {'kernel us/call':>14s} | {'span us/call':>12s} | "
              f"{'gap':>6s} | {'kernel ms/step':>14s}")
        tot_kernel = 0.0
        for n in names:
            kt, kc = regions[n].get("kernel", (0.0, 0))
            st, sc = regions[n].get("span", (0.0, 0))
            calls = kc or sc
            k_us = kt / max(kc, 1)
            s_us = st / max(sc, 1)
            gap = (s_us - k_us) if sc else float("nan")
            step_ms = kt / 1e3 / args.steps
            tot_kernel += step_ms
            print(f"{n:22s} {calls:5d} | {k_us:14.1f} | {s_us:12.1f} | {gap:6.1f} | {step_ms:14.3f}")
        print(f"{'total':22s} {'':5s} | {'':14s} | {'':12s} | {'':6s} | {tot_kernel:14.3f}")

    print("\nkernel = summed GPU kernel time inside the region (what a CUDA graph replays)")
    print("span   = first kernel start to last kernel end (includes eager launch gaps)")


if __name__ == "__main__":
    main()