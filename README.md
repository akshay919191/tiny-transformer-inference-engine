# tiny-transformer-inference

A GPT-style decoder-only transformer in PyTorch, trained on TinyStories with the GPT-2 tokenizer. It includes MHA/MQA attention, reference and CUDA backends, KV caching, weight-only INT8 checkpoints, and paged batched serving.

## Setup

```bash
pip install -r requirements.txt
```

Training data is expected as tokenized `uint16` files at `data/train.bin` and `data/val.bin`.

## Train

Download the original TinyStories train/validation text files from the
[Hugging Face dataset](https://huggingface.co/datasets/roneneldan/TinyStories),
then convert them to this repo's GPT-2-tokenized `uint16` format:

```bash
mkdir -p data checkpoints
curl -L --fail --retry 3 \
  -o data/TinyStories-train.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt
curl -L --fail --retry 3 \
  -o data/TinyStories-valid.txt \
  https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt
python tokenizer.py
```

The Hugging Face page currently lists the training text file at about 1.92 GB.
Keep additional disk space free for tokenized files and checkpoints. The
training loader reads `data/train.bin` and `data/val.bin`.

Start with a short run to confirm the data, model, and checkpoint path work:

```bash
python train.py --max_steps 20 --eval_interval 10 --batch_size 4 \
    --d_model 128 --num_layers 4 --num_heads 4 --num_kv_heads 1 \
    --max_seq_len 128 --backend auto --device cuda
```

Then train longer, for example:

```bash
python train.py --max_steps 5000 --eval_interval 100 --batch_size 8 \
    --d_model 256 --num_layers 8 --num_heads 8 --num_kv_heads 1 \
    --max_seq_len 256 --backend auto --device cuda
```

Checkpoints include model configuration and training settings. Inference reads
these settings by default. `--backend auto` chooses a backend based on CUDA
availability and kernel support. The model modules import custom kernel
extensions at import time, so selecting the PyTorch backend does not guarantee
the project can run on a machine where those extensions are absent. Use the
short training command first to check your local build.

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

`--tokens` must be nonnegative. The prompt and requested continuation must fit
the checkpoint's `max_seq_len`; generation stops early on EOS. `--device`
accepts `auto`, `cpu`, or `cuda`; `auto` chooses CUDA when PyTorch reports it
available, otherwise CPU. Optional `--backend` and `--attn-type` override
checkpoint training settings. The custom kernel import requirement described
above also applies to inference.

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

Run the focused tests from the project root with `pytest`. These cover sampling,
quantization helpers, and generation guards without loading a real checkpoint.

```bash
pytest tests/test_sampling.py tests/test_quantization.py tests/test_inference.py
```

`pytest -q` discovers additional integration tests. Several expect CUDA and
specific artifacts, including `checkpoints/ckpt_step20000.pt` and
`tests/golden.pt`; a short training run does not create those artifacts, so the
full suite is not expected to pass from a fresh checkout.

Measure prefill and decode separately on a checkpoint:

```bash
python -m benchmarks.bench_inference --ckpt checkpoints/ckpt_final.pt \
    --device auto --batch 1 --prompt-len 128 --decode-steps 32
```

The benchmark reports median prefill latency, median per-token decode latency,
throughput, and CUDA peak allocation when applicable. It requires a trained
checkpoint and a working model/kernel installation. Results depend on the
machine and settings; the repository does not claim a fixed speedup. See
[`docs/benchmarking.md`](docs/benchmarking.md) for measurement details.

## Architecture and cache notes

See [`docs/architecture.md`](docs/architecture.md), [`docs/kv_cache.md`](docs/kv_cache.md), and [`docs/inference.md`](docs/inference.md).
