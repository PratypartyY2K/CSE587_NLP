from __future__ import annotations

import argparse
import math
import statistics
import timeit

import torch

from basics.basics.model import BasicsTransformerLM
from basics.basics.nn_utils import cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark BasicsTransformerLM forward and backward passes."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--warmup-steps", "-w", type=int, default=10)
    parser.add_argument("--steps", "-n", type=int, default=50)
    parser.add_argument(
        "--mode",
        choices=("forward", "forward-backward"),
        default="forward-backward",
        help="Whether to benchmark only the forward pass or both forward and backward.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, for example 'cuda', 'cuda:0', or 'cpu'.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Floating-point dtype for model parameters and activations.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def validate_args(args: argparse.Namespace) -> None:
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads.")

    if args.device == "cpu" and args.dtype == torch.float16:
        raise ValueError("float16 benchmarking is not supported on CPU; use float32 or bfloat16.")


def make_random_batch(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    return inputs, targets


def run_step(
    model: BasicsTransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    device: torch.device,
) -> None:
    if mode == "forward":
        with torch.no_grad():
            _ = model(inputs)
    else:
        model.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()

    synchronize_if_needed(device)


def benchmark(
    model: BasicsTransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    warmup_steps: int,
    steps: int,
    mode: str,
    device: torch.device,
) -> list[float]:
    for _ in range(warmup_steps):
        run_step(model, inputs, targets, mode=mode, device=device)

    durations = []
    for _ in range(steps):
        start = timeit.default_timer()
        run_step(model, inputs, targets, mode=mode, device=device)
        end = timeit.default_timer()
        durations.append(end - start)
    return durations


def format_results(durations: list[float], batch_size: int, context_length: int) -> str:
    total = sum(durations)
    avg = statistics.mean(durations)
    std = statistics.stdev(durations) if len(durations) > 1 else 0.0
    tokens_per_step = batch_size * context_length
    steps_per_second = math.inf if total == 0 else len(durations) / total
    return "\n".join(
        [
            f"total_time_s: {total:.6f}",
            f"mean_step_time_s: {avg:.6f}",
            f"std_step_time_s: {std:.6f}",
            f"steps_per_second: {steps_per_second:.4f}",
            f"tokens_per_second: {steps_per_second * tokens_per_step:.4f}",
            f"min_step_time_s: {min(durations):.6f}",
            f"max_step_time_s: {max(durations):.6f}",
        ]
    )


def main() -> None:
    args = parse_args()
    args.dtype = resolve_dtype(args.dtype)
    validate_args(args)

    device = torch.device(args.device)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device=device, dtype=args.dtype)
    model.train(args.mode == "forward-backward")

    inputs, targets = make_random_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        device=device,
    )

    synchronize_if_needed(device)
    durations = benchmark(
        model=model,
        inputs=inputs,
        targets=targets,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        mode=args.mode,
        device=device,
    )

    print(f"device: {device}")
    print(f"dtype: {args.dtype}")
    print(f"mode: {args.mode}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
    print(format_results(durations, batch_size=args.batch_size, context_length=args.context_length))


if __name__ == "__main__":
    main()
