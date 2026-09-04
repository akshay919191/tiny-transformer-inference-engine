
import torch


def cuda_available():

    if not torch.cuda.is_available():
        return False

    try:
        import kernels.kernel
    except ImportError:
        return False

    return True


def kernel_supports(config, attn_type):

    head_dim = config.d_model // config.num_heads

    if head_dim % 2 != 0:
        return False, f"head_dim={head_dim} is odd; RoPE kernel requires even head_dim"

    # GQA/MQA: num_heads must be divisible by kv_heads
    kv_heads = getattr(config, "num_kv_heads", config.num_heads)
    if config.num_heads % kv_heads != 0:
        return False, (
            f"num_heads={config.num_heads} not divisible by "
            f"num_kv_heads={kv_heads} — invalid GQA/MQA grouping"
        )

    # attn_type support — adjust as you actually implement/test each
    SUPPORTED_ATTN_TYPES = {"mqa", "mha"}
    if attn_type not in SUPPORTED_ATTN_TYPES:
        return False, f"attn_type={attn_type!r} not supported by the CUDA backend yet"

    MAX_SUPPORTED_SEQ_LEN = 8192  # TODO: adjust to your kernel's actual limit
    if config.max_seq_len > MAX_SUPPORTED_SEQ_LEN:
        return False, (
            f"max_seq_len={config.max_seq_len} exceeds CUDA kernel limit "
            f"of {MAX_SUPPORTED_SEQ_LEN}"
        )

    return True, "ok"


def resolve_backend(requested_backend, config, attn_type):
    if requested_backend not in ("cuda", "pytorch", "auto"):
        raise ValueError(f"Unknown backend '{requested_backend}', expected cuda/pytorch/auto")

    if requested_backend == "pytorch":
        return "pytorch"

    cuda_ok = cuda_available()
    kernel_ok, reason = (False, "cuda unavailable") if not cuda_ok else kernel_supports(config, attn_type)

    if requested_backend == "cuda":
        if not cuda_ok:
            raise RuntimeError(
                "Backend 'cuda' was explicitly requested, but CUDA / custom "
                "kernel extensions are not available on this machine. "
                "Use --backend pytorch or --backend auto instead."
            )
        if not kernel_ok:
            raise RuntimeError(
                f"Backend 'cuda' was explicitly requested, but the CUDA kernel "
                f"does not support this configuration: {reason}. "
                f"Use --backend pytorch or --backend auto instead."
            )
        return "cuda"

    if cuda_ok and kernel_ok:
        print("[backend] auto -> cuda (capability check passed)")
        return "cuda"
    else:
        print(f"[backend] auto -> pytorch (cuda unavailable or unsupported: {reason})")
        return "pytorch"