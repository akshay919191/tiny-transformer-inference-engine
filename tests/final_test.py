import inspect
import sys
from pathlib import Path
import pytest
import torch
import tiktoken

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kv_cache import KVCache_kv
from models.transformer_block import Transformer
from models.mqa import MQA_Cached
from models.model_config import ModelConfig
from sampling import sample

CHECKPOINT = "checkpoints/ckpt_final.pt"
enc = tiktoken.get_encoding("gpt2")



@pytest.fixture(scope="module")
def checkpoint():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for tests")
    return torch.load(CHECKPOINT, map_location="cuda", weights_only=False)


@pytest.fixture(scope="module")
def config(checkpoint):
    saved_config = dict(checkpoint["model_config"])
    valid_fields = set(inspect.signature(ModelConfig).parameters)
    saved_config = {k: v for k, v in saved_config.items() if k in valid_fields}
    return ModelConfig(**saved_config)


@pytest.fixture(scope="module")
def full_model(config, checkpoint):
    train_cfg = checkpoint.get("train_config", {})
    attn_type = train_cfg.get("attn_type", "mqa")
    backend = "cuda"

    model = Transformer(config, attn_type=attn_type, backend=backend)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device="cuda", dtype=torch.float16)
    model.eval()
    return model


def test_full_text_generation(full_model, config):
    """
    Runs actual generation with KV cache and prints output live.
    """
    print("LIVE TEXT GENERATION TEST")

    prompt = "Once upon a time"
    print(f"Prompt: {prompt!r}\nGenerated: {prompt}", end="", flush=True)

    ids = enc.encode_ordinary(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device="cuda").unsqueeze(0)

    kv_heads = getattr(config, "num_kv_heads", config.num_heads)
    head_dim = config.d_model // config.num_heads

    cache = KVCache_kv(
        num_layers=config.num_layers,
        batch_size=ids.shape[0],
        num_heads=kv_heads,
        max_seq_len=config.max_seq_len,
        head_dim=head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    # 1. Prefill
    with torch.no_grad():
        logits = full_model(ids, kv_cache=cache)[:, -1, :]
    next_id = sample(logits, temperature=0.8, top_k=20)
    
    generated_tokens = [next_id.item()]
    print(enc.decode([next_id.item()]), end="", flush=True)

    # 2. Decode steps
    for _ in range(25):
        with torch.no_grad():
            logits = full_model(next_id, kv_cache=cache)[:, -1, :]
        next_id = sample(logits, temperature=0.8, top_k=20)
        generated_tokens.append(next_id.item())
        print(enc.decode([next_id.item()]), end="", flush=True)

    print("\n" + "=" * 50)
    assert len(generated_tokens) == 26
    assert not torch.isnan(logits).any(), "Logits contained NaN values!"


def test_transformer_prefill_and_decode(full_model, config):
    """
    Verifies KV cache shape progression and numerical sanity.
    """
    batch_size = 2
    prompt_len = 16
    tokens = torch.randint(0, config.vocab_size, (batch_size, prompt_len), device="cuda")

    kv_heads = getattr(config, "num_kv_heads", config.num_heads)
    head_dim = config.d_model // config.num_heads

    cache = KVCache_kv(
        num_layers=config.num_layers,
        batch_size=batch_size,
        num_heads=kv_heads,
        max_seq_len=config.max_seq_len,
        head_dim=head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    with torch.no_grad():
        # Prefill
        logits = full_model(tokens, kv_cache=cache)
        assert logits.shape == (batch_size, prompt_len, config.vocab_size)
        assert cache.length == prompt_len

        # Decode Step
        next_tok = torch.randint(0, config.vocab_size, (batch_size, 1), device="cuda")
        decode_logits = full_model(next_tok, kv_cache=cache)
        assert decode_logits.shape == (batch_size, 1, config.vocab_size)
        assert cache.length == prompt_len + 1


def test_kv_cache_overflow(config):
    head_dim = config.d_model // config.num_heads

    cache = KVCache_kv(
        num_layers=config.num_layers,
        batch_size=1,
        num_heads=config.num_kv_heads,
        max_seq_len=4,
        head_dim=head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    k = torch.randn(1, config.num_kv_heads, 4, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(1, config.num_kv_heads, 4, head_dim, device="cuda", dtype=torch.float16)

    cache.update(0, k, v)
    cache.advance(4)

    k_new = torch.randn(1, config.num_kv_heads, 1, head_dim, device="cuda", dtype=torch.float16)
    v_new = torch.randn(1, config.num_kv_heads, 1, head_dim, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="KV cache overflow"):
        cache.update(0, k_new, v_new)


def test_invalid_layer_index(config):
    head_dim = config.d_model // config.num_heads

    cache = KVCache_kv(
        num_layers=config.num_layers,
        batch_size=1,
        num_heads=config.num_kv_heads,
        max_seq_len=config.max_seq_len,
        head_dim=head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    k = torch.randn(1, config.num_kv_heads, 1, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(1, config.num_kv_heads, 1, head_dim, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="layer_idx"):
        cache.update(config.num_layers, k, v)


def test_wrong_head_dim(config):
    expected_head_dim = config.d_model // config.num_heads
    wrong_head_dim = expected_head_dim + 1

    cache = KVCache_kv(
        num_layers=config.num_layers,
        batch_size=1,
        num_heads=config.num_kv_heads,
        max_seq_len=config.max_seq_len,
        head_dim=expected_head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    k = torch.randn(1, config.num_kv_heads, 1, wrong_head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(1, config.num_kv_heads, 1, wrong_head_dim, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="Head dim mismatch"):
        cache.update(0, k, v)


def test_wrong_device(config):
    head_dim = config.d_model // config.num_heads

    cache = KVCache_kv(
        num_layers=config.num_layers,
        batch_size=1,
        num_heads=config.num_kv_heads,
        max_seq_len=config.max_seq_len,
        head_dim=head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    k = torch.randn(1, config.num_kv_heads, 1, head_dim, device="cpu", dtype=torch.float16)
    v = torch.randn(1, config.num_kv_heads, 1, head_dim, device="cpu", dtype=torch.float16)

    with pytest.raises(ValueError, match="Device mismatch"):
        cache.update(0, k, v)



def test_checkpoint_architecture_mismatch(checkpoint, config):
    saved_config = dict(checkpoint["model_config"])
    valid_fields = set(inspect.signature(ModelConfig).parameters)
    saved_config = {k: v for k, v in saved_config.items() if k in valid_fields}

    bad_config_dict = dict(saved_config)
    bad_config_dict["num_layers"] = bad_config_dict.get("num_layers", 8) + 1

    bad_config = ModelConfig(**bad_config_dict)
    model = Transformer(bad_config, backend="pytorch")

    with pytest.raises(RuntimeError):
        model.load_state_dict(checkpoint["model"], strict=True)


def test_unsupported_attention_type(config):
    with pytest.raises((AssertionError, ValueError)):
        MQA_Cached(config, backend="invalid_backend")



if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))