from __future__ import annotations

import argparse
import math
import statistics
import timeit
from contextlib import nullcontext

import torch

from basics.basics.model import BasicsTransformerLM
from basics.basics.optimizer import AdamW
from basics.basics.nn_utils import cross_entropy


MODEL_SPECS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def nvtx_range(name: str):
    if torch.cuda.is_available():
        return torch.cuda.nvtx.range(name)
    return nullcontext()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark BasicsTransformerLM forward and backward passes."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument(
        "--model-size",
        choices=tuple(MODEL_SPECS),
        help="Named model preset from Table 1. Overrides d-model/d-ff/num-layers/num-heads.",
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--warmup-steps", "-w", type=int, default=10)
    parser.add_argument("--steps", "-n", type=int, default=50)
    parser.add_argument(
        "--mode",
        choices=("forward", "forward-backward", "train-step"),
        default="train-step",
        help="Benchmark inference only, forward+backward, or a full training step including optimizer.step().",
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

    if str(args.device).startswith("cpu") and args.dtype == torch.float16:
        raise ValueError("float16 benchmarking is not supported on CPU; use float32 or bfloat16.")


def apply_model_preset(args: argparse.Namespace) -> None:
    if args.model_size is None:
        return

    for key, value in MODEL_SPECS[args.model_size].items():
        setattr(args, key, value)


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
    optimizer: AdamW | None,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    device: torch.device,
) -> None:
    if mode == "forward":
        with nvtx_range("measured_step"), torch.no_grad():
            with nvtx_range("forward_pass"):
                _ = model(inputs)
    elif mode == "forward-backward":
        with nvtx_range("measured_step"):
            model.zero_grad(set_to_none=True)
            with nvtx_range("forward_pass"):
                logits = model(inputs)
            with nvtx_range("loss"):
                loss = cross_entropy(logits, targets)
            with nvtx_range("backward_pass"):
                loss.backward()
    else:
        if optimizer is None:
            raise ValueError("optimizer is required when mode=train-step")

        with nvtx_range("measured_step"):
            optimizer.zero_grad(set_to_none=True)
            with nvtx_range("forward_pass"):
                logits = model(inputs)
            with nvtx_range("loss"):
                loss = cross_entropy(logits, targets)
            with nvtx_range("backward_pass"):
                loss.backward()
            with nvtx_range("optimizer_step"):
                optimizer.step()
    if mode == "forward-backward":
        model.zero_grad(set_to_none=True)

    synchronize_if_needed(device)


def benchmark(
    model: BasicsTransformerLM,
    optimizer: AdamW | None,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    warmup_steps: int,
    steps: int,
    mode: str,
    device: torch.device,
) -> list[float]:
    with nvtx_range("warmup"):
        for _ in range(warmup_steps):
            run_step(model, optimizer, inputs, targets, mode=mode, device=device)

    durations = []
    with nvtx_range("benchmark"):
        for _ in range(steps):
            start = timeit.default_timer()
            run_step(model, optimizer, inputs, targets, mode=mode, device=device)
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
    apply_model_preset(args)
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
    model.train(args.mode != "forward")
    optimizer = AdamW(model.parameters()) if args.mode == "train-step" else None

    inputs, targets = make_random_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        device=device,
    )

    synchronize_if_needed(device)
    durations = benchmark(
        model=model,
        optimizer=optimizer,
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
    if args.model_size is not None:
        print(f"model_size: {args.model_size}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
    print(format_results(durations, batch_size=args.batch_size, context_length=args.context_length))


if __name__ == "__main__":
    main()
