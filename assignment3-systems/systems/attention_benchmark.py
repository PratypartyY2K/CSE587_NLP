from __future__ import annotations

import argparse
import timeit

import torch

from basics.basics.model import scaled_dot_product_attention


DEFAULT_D_MODELS = (16, 32, 64, 128)
DEFAULT_SEQUENCE_LENGTHS = (256, 1024, 4096, 8192, 16384)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-head attention over a grid of embedding dims and sequence lengths."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--include-compiled",
        action="store_true",
        help="Also benchmark a torch.compile version of the attention function.",
    )
    parser.add_argument(
        "--d-models",
        type=int,
        nargs="+",
        default=list(DEFAULT_D_MODELS),
        help="Embedding dimensions to benchmark.",
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEQUENCE_LENGTHS),
        help="Sequence lengths to benchmark.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kwargs = {
        "size": (batch_size, sequence_length, d_model),
        "device": device,
        "dtype": dtype,
        "requires_grad": requires_grad,
    }
    q = torch.randn(**kwargs)
    k = torch.randn(**kwargs)
    v = torch.randn(**kwargs)
    return q, k, v


def causal_mask(batch_size: int, sequence_length: int, device: torch.device) -> torch.Tensor:
    seq = torch.arange(sequence_length, device=device)
    base_mask = seq.unsqueeze(0) >= seq.unsqueeze(1)
    return base_mask.unsqueeze(0).expand(batch_size, -1, -1)


def benchmark_forward(
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup_steps: int,
    steps: int,
    attention_fn,
) -> float:
    q, k, v = make_inputs(batch_size, sequence_length, d_model, device, dtype, requires_grad=False)
    mask = causal_mask(batch_size, sequence_length, device)

    for _ in range(warmup_steps):
        _ = attention_fn(q, k, v, mask)
        synchronize(device)

    durations = []
    for _ in range(steps):
        start = timeit.default_timer()
        _ = attention_fn(q, k, v, mask)
        synchronize(device)
        durations.append(timeit.default_timer() - start)
    return sum(durations) / len(durations)


def benchmark_backward(
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup_steps: int,
    steps: int,
    attention_fn,
) -> tuple[float, int]:
    mask = causal_mask(batch_size, sequence_length, device)

    def backward_iteration() -> tuple[float, int]:
        q, k, v = make_inputs(batch_size, sequence_length, d_model, device, dtype, requires_grad=True)
        out = attention_fn(q, k, v, mask)
        loss = out.sum()
        synchronize(device)
        memory_before_backward = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
        start = timeit.default_timer()
        loss.backward()
        synchronize(device)
        elapsed = timeit.default_timer() - start
        del q, k, v, out, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return elapsed, memory_before_backward

    for _ in range(warmup_steps):
        backward_iteration()

    durations = []
    memory_before_backward_values = []
    for _ in range(steps):
        duration, memory_before_backward = backward_iteration()
        durations.append(duration)
        memory_before_backward_values.append(memory_before_backward)

    return sum(durations) / len(durations), max(memory_before_backward_values)


def format_bytes_as_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.2f} GiB"


def eager_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return scaled_dot_product_attention(q, k, v, mask)


def main() -> None:
    args = parse_args()
    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)

    if device.type != "cuda":
        raise ValueError("This benchmark is intended for CUDA runs.")

    attention_variants: list[tuple[str, object]] = [("eager", eager_attention)]
    if args.include_compiled:
        attention_variants.append(("compiled", torch.compile(eager_attention)))

    print("| variant | d_model | seq_len | forward_time_s | backward_time_s | memory_before_backward | status |")
    print("| --- | --- | --- | --- | --- | --- | --- |")

    for variant_name, attention_fn in attention_variants:
        for d_model in args.d_models:
            for sequence_length in args.sequence_lengths:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                try:
                    forward_time = benchmark_forward(
                        batch_size=args.batch_size,
                        sequence_length=sequence_length,
                        d_model=d_model,
                        device=device,
                        dtype=dtype,
                        warmup_steps=args.warmup_steps,
                        steps=args.steps,
                        attention_fn=attention_fn,
                    )
                    backward_time, memory_before_backward = benchmark_backward(
                        batch_size=args.batch_size,
                        sequence_length=sequence_length,
                        d_model=d_model,
                        device=device,
                        dtype=dtype,
                        warmup_steps=args.warmup_steps,
                        steps=args.steps,
                        attention_fn=attention_fn,
                    )
                    print(
                        f"| {variant_name} | {d_model} | {sequence_length} | {forward_time:.6f} | "
                        f"{backward_time:.6f} | {format_bytes_as_gib(memory_before_backward)} | ok |"
                    )
                except torch.OutOfMemoryError:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    print(
                        f"| {variant_name} | {d_model} | {sequence_length} | - | - | - | oom |"
                    )


if __name__ == "__main__":
    main()
