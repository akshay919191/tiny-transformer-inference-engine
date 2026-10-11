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

It reports model-only TTFT, p50/p90 decode latency, prefill and decode throughput, end-to-end generation time, generated-token throughput, and peak CUDA allocation when applicable. TTFT and total time exclude tokenization and KV-cache allocation. Results depend on the model, backend, device, and settings; this repository does not claim a fixed speedup.

### Local CPU measurement

These are CPU results measured on an Intel Core i5-13450HX with `checkpoints/ckpt_final.pt` (`d_model=128`, 4 layers, 4 heads), PyTorch 2.7.1, and 10 CPU threads. The PyTorch backend was used with 3 warmup runs and 10 measured iterations per row. Each request generated 8 tokens.

| Batch | Prompt tokens | TTFT p50/p90 (ms) | Prefill tok/s | Decode p50/p90 (ms/token) | Decode tok/s | Total p50 (ms) | Generated tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 1.974 / 2.166 | 8,106.9 | 1.425 / 2.037 | 702.0 | 15.393 | 519.7 |
| 1 | 64 | 4.934 / 5.149 | 12,970.3 | 1.667 / 2.485 | 599.9 | 18.834 | 424.8 |
| 1 | 112 | 6.910 / 7.071 | 16,207.9 | 1.485 / 1.958 | 673.4 | 20.021 | 399.6 |
| 2 | 16 | 2.920 / 3.318 | 10,957.6 | 1.585 / 2.263 | 631.0 | 17.333 | 923.1 |
| 2 | 64 | 7.944 / 8.876 | 16,112.1 | 1.924 / 2.726 | 519.8 | 24.977 | 640.6 |
| 2 | 112 | 12.266 / 12.657 | 18,261.9 | 2.048 / 4.611 | 488.3 | 32.152 | 497.6 |
| 4 | 16 | 4.157 / 4.417 | 15,396.7 | 1.578 / 2.317 | 633.7 | 19.028 | 1,681.8 |
| 4 | 64 | 13.253 / 13.292 | 19,316.8 | 1.936 / 5.085 | 516.6 | 32.253 | 992.2 |
| 4 | 112 | 23.044 / 23.280 | 19,440.8 | 1.928 / 6.552 | 518.6 | 43.933 | 728.4 |

Reproduce the measurement with:

```bash
python -m benchmarks.bench_inference --ckpt checkpoints/ckpt_final.pt \
  --device cpu --backend pytorch --sweep --batches 1 2 4 \
  --prompt-lengths 16 64 112 --decode-steps 8 --warmup 3 --iterations 10
```

TTFT measures the prefill forward pass and excludes tokenization and KV-cache allocation. These results describe this CPU and checkpoint only, not a general performance claim.

## Tests

Run the focused tests for sampling, quantization, and generation input checks:

```bash
pytest -q tests/test_sampling.py tests/test_quantization.py tests/test_inference.py
```

In this workspace, `pytest` was not installed, so the pytest files were not run. Thirteen manual assertions covering sampling validation, quantization round trips, and generation stopping/context limits passed. An 8-token CPU generation smoke test also completed. CUDA was unavailable, so no GPU result is reported.

See [`docs/architecture.md`](docs/architecture.md), [`docs/kv_cache.md`](docs/kv_cache.md), [`docs/inference.md`](docs/inference.md), and [`docs/benchmarking.md`](docs/benchmarking.md) for implementation details.
