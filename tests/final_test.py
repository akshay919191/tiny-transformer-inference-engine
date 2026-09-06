import inspect

import pytest
import torch

from kv_cache import KVCache_kv
from models.mqa import MQA_Cached
from models.model_config import ModelConfig


CHECKPOINT = "checkpoints/ckpt_final.pt"


@pytest.fixture
def checkpoint():

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    return torch.load(
        CHECKPOINT,
        map_location="cuda",
        weights_only=False,
    )


@pytest.fixture
def config(checkpoint):

    saved_config = dict(
        checkpoint["model_config"]
    )

    valid_fields = set(
        inspect.signature(
            ModelConfig
        ).parameters
    )

    saved_config = {
        key: value
        for key, value in saved_config.items()
        if key in valid_fields
    }

    return ModelConfig(
        **saved_config
    )


@pytest.fixture
def model(config, checkpoint):

    model = MQA_Cached(
        config,
        backend="cuda",
    ).cuda().half()

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()

    return model


def test_kv_cache_overflow(config):

    head_dim = (
        config.d_model
        // config.num_heads
    )

    cache = KVCache_kv(
        num_layers=config.num_layers,
        batch_size=1,
        num_heads=config.num_kv_heads,
        max_seq_len=4,
        head_dim=head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    k = torch.randn(
        1,
        config.num_kv_heads,
        4,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )

    v = torch.randn(
        1,
        config.num_kv_heads,
        4,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )

    cache.update(0, k, v)
    cache.advance(4)

    k_new = torch.randn(
        1,
        config.num_kv_heads,
        1,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )

    v_new = torch.randn(
        1,
        config.num_kv_heads,
        1,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )

    with pytest.raises(
        ValueError,
        match="KV cache overflow",
    ):
        cache.update(
            0,
            k_new,
            v_new,
        )


def test_invalid_layer_index(config):

    head_dim = (
        config.d_model
        // config.num_heads
    )

    cache = KVCache_kv(
        num_layers=config.num_layers,
        batch_size=1,
        num_heads=config.num_kv_heads,
        max_seq_len=config.max_seq_len,
        head_dim=head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    k = torch.randn(
        1,
        config.num_kv_heads,
        1,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )

    v = torch.randn(
        1,
        config.num_kv_heads,
        1,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )

    with pytest.raises(
        ValueError,
        match="layer_idx",
    ):
        cache.update(
            config.num_layers,
            k,
            v,
        )


def test_wrong_head_dim(config):

    expected_head_dim = (
        config.d_model
        // config.num_heads
    )

    wrong_head_dim = (
        expected_head_dim + 1
    )

    cache = KVCache_kv(
        num_layers=config.num_layers,
        batch_size=1,
        num_heads=config.num_kv_heads,
        max_seq_len=config.max_seq_len,
        head_dim=expected_head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    k = torch.randn(
        1,
        config.num_kv_heads,
        1,
        wrong_head_dim,
        device="cuda",
        dtype=torch.float16,
    )

    v = torch.randn(
        1,
        config.num_kv_heads,
        1,
        wrong_head_dim,
        device="cuda",
        dtype=torch.float16,
    )

    with pytest.raises(
        ValueError,
        match="Head dim mismatch",
    ):
        cache.update(
            0,
            k,
            v,
        )


def test_wrong_device(config):

    head_dim = (
        config.d_model
        // config.num_heads
    )

    cache = KVCache_kv(
        num_layers=config.num_layers,
        batch_size=1,
        num_heads=config.num_kv_heads,
        max_seq_len=config.max_seq_len,
        head_dim=head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    k = torch.randn(
        1,
        config.num_kv_heads,
        1,
        head_dim,
        device="cpu",
        dtype=torch.float16,
    )

    v = torch.randn(
        1,
        config.num_kv_heads,
        1,
        head_dim,
        device="cpu",
        dtype=torch.float16,
    )

    with pytest.raises(
        ValueError,
        match="Device mismatch",
    ):
        cache.update(
            0,
            k,
            v,
        )


def test_checkpoint_architecture_mismatch(
    checkpoint,
    config,
):

    saved_config = dict(
        checkpoint["model_config"]
    )

    valid_fields = set(
        inspect.signature(
            ModelConfig
        ).parameters
    )

    saved_config = {
        key: value
        for key, value in saved_config.items()
        if key in valid_fields
    }

    bad_config_dict = dict(
        saved_config
    )

    original_num_heads = (
        bad_config_dict["num_heads"]
    )

    bad_num_heads = (
        original_num_heads + 1
    )

    while (
        bad_config_dict["d_model"]
        % bad_num_heads
        == 0
    ):
        bad_num_heads += 1

    bad_config_dict[
        "num_heads"
    ] = bad_num_heads

    if (
        "num_kv_heads"
        in bad_config_dict
    ):
        bad_config_dict[
            "num_kv_heads"
        ] = 1

    bad_config = ModelConfig(
        **bad_config_dict
    )

    model = MQA_Cached(
        bad_config,
        backend="pytorch",
    )

    with pytest.raises(RuntimeError):
        model.load_state_dict(
            checkpoint["model"],
            strict=True,
        )


def test_cuda_backend_unavailable():

    if torch.cuda.is_available():
        pytest.skip(
            "CUDA is available"
        )

    config = ModelConfig()

    with pytest.raises(
        (
            RuntimeError,
            ValueError,
            AssertionError,
        )
    ):
        MQA_Cached(
            config,
            backend="cuda",
        )


def test_unsupported_attention_type(
    config,
):

    with pytest.raises(
        (
            AssertionError,
            ValueError,
        )
    ):
        MQA_Cached(
            config,
            backend="invalid_backend",
        )