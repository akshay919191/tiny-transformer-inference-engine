````markdown
# tiny-transformer-inference

A GPT-style decoder-only transformer built from scratch in PyTorch, with optional Triton CUDA attention kernels.

Trained on TinyStories using the GPT-2 tokenizer.

## Architecture

- Decoder-only transformer with RoPE, RMSNorm, and SwiGLU MLPs
- Attention: MHA (multi-head) or MQA (multi-query, 1 shared KV head)
- Two interchangeable backends:
  - `pytorch` — reference implementation
  - `cuda` — Triton kernels
- KV cache + streaming generation (in progress)

| Config | Value |
|---|---:|
| `vocab_size` | 50257 (GPT-2) |
| `d_model` | 256 |
| `num_layers` | 8 |
| `num_heads` | 8 |
| `num_kv_heads` | 1 (MQA) |
| `max_seq_len` | 512 |

## Setup

```bash
pip install -r requirements.txt
````

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

| Flag             | Default | Description                  |
| ---------------- | ------- | ---------------------------- |
| `--max_steps`    | `5000`  | Training steps               |
| `--lr`           | `3e-4`  | Learning rate (AdamW)        |
| `--weight_decay` | `0.1`   | Weight decay                 |
| `--device`       | `cuda`  | Device                       |
| `--backend`      | `cuda`  | `cuda` (Triton) or `pytorch` |
| `--attn_type`    | `mqa`   | `mqa` or `mha`               |

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
│   ├── softmax_kernel/
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
