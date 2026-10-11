# Tiny Transformer Inference Engine

A decoder-only GPT-style transformer in PyTorch. The project supports MHA and MQA attention, KV-cached generation, a PyTorch attention backend, optional CUDA kernels, INT8 weight-only checkpoints, and paged batched serving.

## Setup

```bash
pip install -r requirements.txt
```

## Data and training

Download TinyStories, tokenize it with the GPT-2 tokenizer, then train. The tokenizer writes `data/train.bin` and `data/val.bin` in the format used by the trainer.

```bash
mkdir -p data checkpoints
curl -L --fail --retry 3 -o data/TinyStories-train.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
curl -L --fail --retry 3 -o data/TinyStories-valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
python tokenizer.py
```

Start with a short run, then increase steps and model size as your hardware allows:

```bash
python train.py --max_steps 20 --eval_interval 10 --batch_size 4 \
  --d_model 128 --num_layers 4 --num_heads 4 --num_kv_heads 1 \
  --max_seq_len 128 --backend auto --device cuda

python train.py --max_steps 5000 --eval_interval 100 --batch_size 8 \
  --d_model 256 --num_layers 8 --num_heads 8 --num_kv_heads 1 \
  --max_seq_len 256 --backend auto --device cuda
```

Training saves `checkpoints/ckpt_final.pt` and periodic step checkpoints. The checkpoint stores the model configuration used for inference.

## Inference

```bash
python inference.py --ckpt checkpoints/ckpt_final.pt \
  --prompt "Once upon a time" --tokens 100 --device auto
```

Sampling options include `--temperature`, `--top-k`, and `--top-p`. The prompt and generated tokens must fit within the checkpoint's `max_seq_len`; generation stops at EOS. Use `--backend` and `--attn-type` to override the checkpoint settings.

### INT8 weight-only quantization

```bash
python inference.py --ckpt checkpoints/ckpt_final.pt \
  --quantize checkpoints/ckpt_int8.pt --quant-mode channel
python inference.py --ckpt checkpoints/ckpt_int8.pt --quantized \
  --prompt "Once upon a time" --tokens 100
```

Quantization stores weights as INT8 with scales. Inference dequantizes them for floating-point operations; it does not use integer matrix multiplication. Choose `tensor`, `channel`, or `group` mode; group mode also requires `--group-size`.

### Paged batched serving

```bash
python inference.py --ckpt checkpoints/ckpt_final.pt --serving \
  --prompts "Once upon a time" "A little dog" \
  --tokens 80 --num-gpu-blocks 256 --block-size 16
```

Paged serving uses the checkpoint's MQA model and a paged KV cache. The block count controls KV-cache capacity.

## Performance

Generation has two distinct costs: prefill processes the prompt, and decode processes one new token at a time using the KV cache. Run the benchmark on your own checkpoint and hardware to measure both:

```bash
python -m benchmarks.bench_inference --ckpt checkpoints/ckpt_final.pt \
  --device auto --batch 1 --prompt-len 128 --decode-steps 32
```

It reports median prefill latency, median decode latency per token, throughput, and peak CUDA allocation when applicable. Results depend on the model, backend, device, and settings; this repository does not claim a fixed speedup.

## Tests

Run the focused tests for sampling, quantization, and generation input checks:

```bash
pytest -q tests/test_sampling.py tests/test_quantization.py tests/test_inference.py
```

See [`docs/architecture.md`](docs/architecture.md), [`docs/kv_cache.md`](docs/kv_cache.md), [`docs/inference.md`](docs/inference.md), and [`docs/benchmarking.md`](docs/benchmarking.md) for implementation details.
