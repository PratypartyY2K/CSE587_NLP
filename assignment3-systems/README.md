# Assignment 3 Systems

This repository contains a pure PyTorch reference implementation and a Triton forward kernel for FlashAttention-2, along with a benchmarking script for comparing that implementation against regular PyTorch attention.

## FlashAttention Files

- `systems/flash_attention.py`
  - `FlashAttentionPytorchFunction`
  - `FlashAttentionTritonFunction`
  - Triton forward kernel for FlashAttention-2
  - PyTorch backward recomputation path using `torch.compile`
- `systems/attention_benchmark.py`
  - `triton.testing.do_bench` benchmark for PyTorch vs Triton FlashAttention
- `tests/adapters.py`
  - adapter hooks used by the assignment tests

## Implemented Features

- Pure PyTorch tiled FlashAttention-2 forward pass
- Triton FlashAttention-2 forward kernel
- Optional causal masking flag `is_causal=False` on both autograd functions
- Backward pass implemented in PyTorch using the FlashAttention recomputation equations
- Benchmark script that reports forward, backward, and end-to-end latency

## Running Tests

PyTorch forward:

```bash
uv run pytest -k test_flash_forward_pass_pytorch
```

Triton forward:

```bash
uv run pytest -k test_flash_forward_pass_triton
```

Backward:

```bash
uv run pytest -k test_flash_backward
```

Run all attention tests:

```bash
uv run pytest -k test_attention
```

## Running The Benchmark

The benchmark is intended for a single CUDA GPU and uses `triton.testing.do_bench`.

Default assignment-style run:

```bash
uv run python -m systems.attention_benchmark
```

This benchmark:

- uses batch size `1`
- enables causal masking
- sweeps sequence lengths from `128` to `65536` by powers of 2
- sweeps embedding dimensions `16, 32, 64, 128`
- sweeps `torch.bfloat16` and `torch.float32`
- reports `forward_ms`, `backward_ms`, and `end_to_end_ms`

## Benchmark Summary

From the collected benchmark results on GPU:

- Triton forward latency is consistently lower than regular PyTorch forward latency.
- Backward latency is typically higher for the Triton variant because the backward pass is still implemented in PyTorch rather than Triton.
- End-to-end latency is usually close between the two implementations, and Triton becomes more competitive as sequence length increases.
- The Triton implementation avoids several large `float32` OOM cases where the regular PyTorch baseline runs out of memory at sequence length `65536`.

## Notes

- The Triton path requires CUDA and Triton support, so it should be developed or benchmarked on Linux/Colab rather than macOS.
- A cuBLAS warning about initializing the CUDA context may appear on first backward execution; this did not affect correctness in testing.
