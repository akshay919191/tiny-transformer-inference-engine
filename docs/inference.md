# Inference

Use the root `inference.py` entry point for standard generation, checkpoint quantization, and paged batched serving.

```bash
python inference.py --ckpt checkpoints/ckpt_final.pt \
    --prompt "Once upon a time" --tokens 100 --device auto
```

The loader reads model configuration and attention/backend choices from the checkpoint. Use `--backend pytorch` or `--backend cuda` to override the backend, and `--attn-type mha|mqa` to override attention type. Explicit CUDA device requests fail clearly when CUDA is unavailable. `auto` uses CUDA when available and CPU otherwise.

The prompt must encode to at least one token. Its length plus `--tokens` must fit `max_seq_len`. Sampling accepts `--temperature`, `--top-k`, and `--top-p`; non-finite logits and invalid sampling parameters are rejected. Generation stops at EOS.

INT8 weights are supported through `--quantize OUTPUT` and `--quantized`. This is weight-only storage quantization: weights are dequantized to floating point for each operation, while activations and the KV cache stay floating point. Quantization rejects non-finite weights and invalid group sizes.

Paged batch mode is selected with `--serving --prompts ...`. It accepts standard checkpoints and uses `--num-gpu-blocks` and power-of-two `--block-size` to size the KV pool.
