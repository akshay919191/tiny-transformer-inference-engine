# tiny-transformer-inference

A GPT-style decoder-only transformer built from scratch in PyTorch, with optional Triton CUDA attention kernels.

Trained on TinyStories using the GPT-2 tokenizer.

## Architecture

* Decoder-only transformer with RoPE, RMSNorm, and SwiGLU MLPs
* Attention: MHA (multi-head) or MQA (multi-query, 1 shared KV head)
* Two interchangeable backends:

  * `pytorch` — reference implementation
  * `cuda` — cpp extension kernels
* KV cache + streaming generation

| Config         |         Value |
| -------------- | ------------: |
| `vocab_size`   | 50257 (GPT-2) |
| `d_model`      |           256 |
| `num_layers`   |             8 |
| `num_heads`    |             8 |
| `num_kv_heads` |       1 (MQA) |
| `max_seq_len`  |           512 |

## Setup

```bash
pip install -r requirements.txt
```

## Data

Tokenized TinyStories as `uint16` binaries, produced with `tiktoken` (GPT-2 encoding):

```text
data/
├── train.bin
└── val.bin
```

## Training

```bash
python train.py --max_steps 5000 --lr 3e-4 --backend cuda --attn_type mqa
```

| Flag             | Default | Description                         |
| ---------------- | ------- | ----------------------------------- |
| `--max_steps`    | `5000`  | Training steps                      |
| `--lr`           | `3e-4`  | Learning rate (AdamW)               |
| `--weight_decay` | `0.1`   | Weight decay                        |
| `--device`       | `cuda`  | Device                              |
| `--backend`      | `cuda`  | `cuda` (cpp extension) or `pytorch` |
| `--attn_type`    | `mqa`   | `mqa` or `mha`                      |

Checkpoints are saved to `checkpoints/` and store both the model config and training flags, so inference reproduces the exact setup.

## Generation

```bash
python generation.py \
    --ckpt checkpoints/ckpt_final.pt \
    --prompt "Once upon a time" \
    --token 100
```

| Flag            | Default                     | Description          |
| --------------- | --------------------------- | -------------------- |
| `--ckpt`        | `checkpoints/ckpt_final.pt` | Checkpoint path      |
| `--prompt`      | `"Once upon a time"`        | Prompt text          |
| `--temperature` | `1.0`                       | Sampling temperature |
| `--token`       | `100`                       | Tokens to generate   |

Tokens are printed to the terminal as they are generated (streaming).

## Benchmarks

Run the prefill and decode benchmark with both backends:

```bash
python -m benchmarks.bench_decode_prefill --batch 32 --compare
```

### Performance Summary

**Model Profile:** 27.7M parameters | MQA Attention | `d_model` 256 | 8 Layers | Batch 32 | Prompt Length 100

| Metric                    | `cuda` (cpp extension) Backend | `pytorch` Backend |         Improvement |
| :------------------------ | -----------------------------: | ----------------: | ------------------: |
| **Prefill Latency (p50)** |                  **12.143 ms** |         12.717 ms |    **~4.5% faster** |
| **Prefill Throughput**    |            **263,535.2 tok/s** |   251,622.0 tok/s | **+11,913.2 tok/s** |
| **Decode Latency (p50)**  |                   **0.577 ms** |          0.723 ms |   **~20.2% faster** |
| **Decode Throughput**     |              **1,734.6 tok/s** |     1,382.3 tok/s |    **+352.3 tok/s** |
| **Peak VRAM Memory**      |                     **708 MB** |            794 MB |    **Saving 86 MB** |

### KV Cache Scaling

The CUDA backend maintains nearly constant decode latency as the KV cache grows:

| KV Cache Length | Decode Latency |
| --------------: | -------------: |
|             ~10 |       1.880 ms |
|            ~490 |       1.923 ms |
|           Ratio |      **1.02x** |

This demonstrates that MQA + KV caching keeps decode performance relatively stable as the cached sequence grows.

### CUDA Profiler

The CUDA backend was profiled during inference to identify the major GPU execution costs.

| Operation               |  CUDA Time | % of CUDA Time |
| ----------------------- | ---------: | -------------: |
| `aten::mm` / GEMM       |   4.386 ms |         48.90% |
| PyTorch Flash Attention |   1.996 ms |         22.26% |
| RMSNorm CUDA kernel     | 937.739 µs |         10.46% |
| `aten::copy_`           | 395.489 µs |          4.41% |
| RoPE CUDA kernel        | 324.585 µs |          3.62% |
| `aten::add`             | 313.798 µs |          3.50% |

**Total profiled CUDA time:** 8.968 ms

The profiling results show that matrix multiplications are currently the largest GPU execution cost, followed by attention and RMSNorm.

## Project Structure

```text
tiny-transformer-inference/

│
├── .vscode/
│
├── benchmarks/
│   ├── __init__.py
│   ├── benchmark_decode.py
│   ├── benchmark_prefill.py
│   ├── latency.py
│   ├── memory.py
│   └── throughput.py
│
├── checkpoints/
│   ├── ckpt_final.pt
│   ├── ckpt_step0.pt
│   └── ckpt_step4500.pt
│
├── configs/
│   ├── __init__.py
│   └── benchmark_config.py
│
├── data/
│   ├── TinyStories-train.txt
│   ├── TinyStories-valid.txt
│   ├── train.bin
│   └── val.bin
│
├── docs/
│   ├── __init__.py
│   ├── architecture.md
│   ├── benchmarking.md
│   ├── inference.md
│   └── kv_cache.md
│
├── kernels/
│   ├── common/
│   ├── cuda-kSAMPLING/
│   ├── flashattn/
│   ├── rmsnorm_kernel/
│   ├── rope_kernel/
│   └── kernel.py
│
├── models/
│   ├── __init__.py
│   ├── attention.py
│   ├── embedding.py
│   ├── mlp.py
│   ├── model_config.py
│   ├── mqa.py
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── test.py
│   └── transformer_block.py
│
├── tests/
│   └── ...
│
├── config.py
├── generation.py
├── kv_cache.py
├── modeltest.py
├── README.md
├── requirements.txt
├── sampling.py
├── test_generator.py
├── tokenizer.py
└── train.py
```
