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
    parser.add_argument(
        "--autocast-dtype",
        choices=("none", "float16", "bfloat16"),
        default="none",
        help="Optional autocast dtype for mixed precision. Keeps model parameters in --dtype.",
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


def maybe_autocast_context(device: torch.device, autocast_dtype: torch.dtype | None):
    if autocast_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=autocast_dtype)


def validate_args(args: argparse.Namespace) -> None:
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model must be divisible by --num-heads.")

    if str(args.device).startswith("cpu") and args.dtype == torch.float16:
        raise ValueError("float16 benchmarking is not supported on CPU; use float32 or bfloat16.")
    if args.autocast_dtype is not None and args.dtype != torch.float32:
        raise ValueError("Mixed precision autocast expects FP32 model parameters; set --dtype=float32.")
    if args.autocast_dtype is not None and not str(args.device).startswith("cuda"):
        raise ValueError("Autocast benchmarking is only supported on CUDA for this script.")


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
    autocast_dtype: torch.dtype | None,
) -> dict[str, float]:
    timings: dict[str, float] = {}

    def time_block(name: str, fn) -> torch.Tensor | None:
        synchronize_if_needed(device)
        start = timeit.default_timer()
        result = fn()
        synchronize_if_needed(device)
        timings[name] = timeit.default_timer() - start
        return result

    if mode == "forward":
        with nvtx_range("measured_step"), torch.no_grad():
            with maybe_autocast_context(device, autocast_dtype):
                with nvtx_range("forward_pass"):
                    time_block("forward", lambda: model(inputs))
    elif mode == "forward-backward":
        with nvtx_range("measured_step"):
            model.zero_grad(set_to_none=True)
            with maybe_autocast_context(device, autocast_dtype):
                with nvtx_range("forward_pass"):
                    logits = time_block("forward", lambda: model(inputs))
                with nvtx_range("loss"):
                    loss = time_block("loss", lambda: cross_entropy(logits, targets))
            with nvtx_range("backward_pass"):
                time_block("backward", lambda: loss.backward())
    else:
        if optimizer is None:
            raise ValueError("optimizer is required when mode=train-step")

        with nvtx_range("measured_step"):
            optimizer.zero_grad(set_to_none=True)
            with maybe_autocast_context(device, autocast_dtype):
                with nvtx_range("forward_pass"):
                    logits = time_block("forward", lambda: model(inputs))
                with nvtx_range("loss"):
                    loss = time_block("loss", lambda: cross_entropy(logits, targets))
            with nvtx_range("backward_pass"):
                time_block("backward", lambda: loss.backward())
            with nvtx_range("optimizer_step"):
                time_block("optimizer_step", lambda: optimizer.step())
    if mode == "forward-backward":
        model.zero_grad(set_to_none=True)
    timings["total"] = sum(timings.values())
    return timings


def benchmark(
    model: BasicsTransformerLM,
    optimizer: AdamW | None,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    warmup_steps: int,
    steps: int,
    mode: str,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> list[dict[str, float]]:
    with nvtx_range("warmup"):
        for _ in range(warmup_steps):
            run_step(
                model,
                optimizer,
                inputs,
                targets,
                mode=mode,
                device=device,
                autocast_dtype=autocast_dtype,
            )

    durations = []
    with nvtx_range("benchmark"):
        for _ in range(steps):
            durations.append(
                run_step(
                    model,
                    optimizer,
                    inputs,
                    targets,
                    mode=mode,
                    device=device,
                    autocast_dtype=autocast_dtype,
                )
            )
    return durations


def summarize_timings(durations: list[dict[str, float]], key: str) -> tuple[float, float, float, float]:
    values = [step[key] for step in durations if key in step]
    total = sum(values)
    avg = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return total, avg, std, min(values), max(values)


def format_results(durations: list[dict[str, float]], batch_size: int, context_length: int) -> str:
    total, avg, std, min_value, max_value = summarize_timings(durations, "total")
    tokens_per_step = batch_size * context_length
    steps_per_second = math.inf if total == 0 else len(durations) / total
    lines = [
        f"total_time_s: {total:.6f}",
        f"mean_step_time_s: {avg:.6f}",
        f"std_step_time_s: {std:.6f}",
        f"steps_per_second: {steps_per_second:.4f}",
        f"tokens_per_second: {steps_per_second * tokens_per_step:.4f}",
        f"min_step_time_s: {min_value:.6f}",
        f"max_step_time_s: {max_value:.6f}",
    ]
    for key in ("forward", "loss", "backward", "optimizer_step"):
        if any(key in step for step in durations):
            _, mean_value, std_value, min_phase, max_phase = summarize_timings(durations, key)
            lines.extend(
                [
                    f"mean_{key}_time_s: {mean_value:.6f}",
                    f"std_{key}_time_s: {std_value:.6f}",
                    f"min_{key}_time_s: {min_phase:.6f}",
                    f"max_{key}_time_s: {max_phase:.6f}",
                ]
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    apply_model_preset(args)
    args.dtype = resolve_dtype(args.dtype)
    args.autocast_dtype = None if args.autocast_dtype == "none" else resolve_dtype(args.autocast_dtype)
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
        autocast_dtype=args.autocast_dtype,
    )

    print(f"device: {device}")
    print(f"dtype: {args.dtype}")
    print(f"autocast_dtype: {args.autocast_dtype}")
    print(f"mode: {args.mode}")
    if args.model_size is not None:
        print(f"model_size: {args.model_size}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
    print(format_results(durations, batch_size=args.batch_size, context_length=args.context_length))


if __name__ == "__main__":
    main()
