# tiny-transformer-inference

A GPT-style decoder-only transformer in PyTorch, trained on TinyStories with the GPT-2 tokenizer. It includes MHA/MQA attention, reference and CUDA backends, KV caching, weight-only INT8 checkpoints, and paged batched serving.

## Setup

```bash
pip install -r requirements.txt
```

Training data is expected as tokenized `uint16` files at `data/train.bin` and `data/val.bin`.

## Train

```bash
python train.py --max_steps 5000 --lr 3e-4 --backend cuda --attn_type mqa
```

Checkpoints include model configuration and training settings. Inference reads these settings by default and selects CPU automatically when CUDA is unavailable.

## Inference

All inference and quantization commands use `inference.py`.

```bash
python inference.py --ckpt checkpoints/ckpt_final.pt \
    --prompt "Once upon a time" --tokens 100 --device auto
```

Sampling controls:

```bash
python inference.py --ckpt checkpoints/ckpt_final.pt \
    --prompt "A small story" --tokens 80 \
    --temperature 0.8 --top-k 40 --top-p 0.95
```

`--tokens` must be nonnegative. The prompt and requested continuation must fit the checkpoint's `max_seq_len`; generation stops early on EOS. `--device` accepts `auto`, `cpu`, or `cuda`. Optional `--backend` and `--attn-type` override checkpoint training settings.

### INT8 weight quantization

Quantize a standard checkpoint and then load the resulting file:

```bash
python inference.py --ckpt checkpoints/ckpt_final.pt \
    --quantize checkpoints/ckpt_int8.pt --quant-mode channel
python inference.py --ckpt checkpoints/ckpt_int8.pt --quantized \
    --prompt "Once upon a time" --tokens 100
```

`--quant-mode` accepts `tensor`, `channel` (default), or `group`; group mode also needs `--group-size N`. Weights are dequantized for ordinary floating point operations during inference, so this reduces stored weight size but does not provide integer matrix multiplication. KV cache and activations remain floating point.

### Paged batched serving

```bash
python inference.py --ckpt checkpoints/ckpt_final.pt --serving \
    --prompts "Once upon a time" "A little dog" \
    --tokens 80 --num-gpu-blocks 256 --block-size 16
```

Paged serving currently supports standard checkpoints. Adjust the GPU block count to fit available memory and the desired number/context length of active sequences.

## Tests and benchmarks

Run the repository's tests from the project root with `pytest`. Some integration tests use CUDA and checkpoint/golden artifacts; the focused sampling and inference safety tests run on CPU.

```bash
pytest tests/test_sampling.py tests/test_quantization.py tests/test_inference.py
```

Measure prefill and decode separately on a checkpoint:

```bash
python -m benchmarks.bench_inference --ckpt checkpoints/ckpt_final.pt \
    --device auto --batch 1 --prompt-len 128 --decode-steps 32
```

The benchmark reports median prefill latency, median per-token decode latency, throughput, and CUDA peak allocation when applicable. Compare results only with the same model, backend, device, dtype, and settings. See [`docs/benchmarking.md`](docs/benchmarking.md) for correctness and measurement guidance.

## Architecture and cache notes

See [`docs/architecture.md`](docs/architecture.md), [`docs/kv_cache.md`](docs/kv_cache.md), and [`docs/inference.md`](docs/inference.md).
