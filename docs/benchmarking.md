# Correctness and performance

## Focused CPU checks

```bash
pytest tests/test_sampling.py tests/test_quantization.py tests/test_inference.py
```

These cover sampler argument/logit validation, top-k behavior, finite quantization round trips, EOS stopping, generation length, and context-limit checks. The full suite also contains GPU, checkpoint, golden-output, attention, allocator, and scheduler integration coverage; those cases require the corresponding hardware and artifacts.

## Checkpoint benchmark

```bash
python -m benchmarks.bench_inference --ckpt checkpoints/ckpt_final.pt \
    --device auto --batch 1 --prompt-len 128 --decode-steps 32 \
    --warmup 3 --iterations 10
```

The script warms up the model, creates a fresh KV cache for every measured run, and reports median prefill and per-token decode latency. CUDA work is synchronized around wall-clock timings. It also reports token throughput and peak allocated CUDA memory.

Use the same checkpoint, backend, device, batch size, prompt length, decode length, warmup, and iteration count when comparing results. Report hardware, PyTorch/CUDA versions, and backend alongside any published number. Do not compare these measurements with values from a different setup.

The paged attention implementation has additional accuracy and timing coverage in `tests/test.py`; it compares paged outputs with PyTorch SDPA over prefill, decode, chunked, and mixed shapes. Run that GPU-specific script only where its Triton/CUDA requirements are available.
