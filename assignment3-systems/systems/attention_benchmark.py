from __future__ import annotations

import argparse
import math

import torch

from systems.flash_attention import FlashAttentionTritonFunction

try:
    import triton.testing as triton_testing
except ImportError:  # pragma: no cover - benchmark only runs in CUDA/Triton environments.
    triton_testing = None


DEFAULT_D_MODELS = (64, 128)
DEFAULT_SEQUENCE_LENGTHS = (128, 256, 512, 1024)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Triton FlashAttention against a regular PyTorch attention implementation."
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--d-models", type=int, nargs="+", default=list(DEFAULT_D_MODELS))
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=list(DEFAULT_SEQUENCE_LENGTHS))
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def make_inputs(
    batch_size: int,
    sequence_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
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


def pytorch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        q_positions = torch.arange(n_queries, device=q.device).unsqueeze(-1)
        k_positions = torch.arange(n_keys, device=q.device).unsqueeze(0)
        scores = torch.where(q_positions >= k_positions, scores, torch.full_like(scores, -1e6))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def triton_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    return FlashAttentionTritonFunction.apply(q, k, v, is_causal)


def benchmark_forward(
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    *,
    warmup: int,
    repetitions: int,
) -> float:
    return triton_testing.do_bench(
        lambda: attention_fn(q, k, v, is_causal),
        warmup=warmup,
        rep=repetitions,
    )


def benchmark_backward(
    attention_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    is_causal: bool,
    *,
    warmup: int,
    repetitions: int,
) -> float:
    out = attention_fn(q, k, v, is_causal)

    def run_backward() -> None:
        q.grad = None
        k.grad = None
        v.grad = None
        out.backward(grad_output, retain_graph=True)

    return triton_testing.do_bench(run_backward, warmup=warmup, rep=repetitions)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    if device.type != "cuda":
        raise ValueError("This benchmark is intended for CUDA runs.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if triton_testing is None:
        raise RuntimeError("This benchmark requires Triton to be installed.")

    variants = (
        ("pytorch", pytorch_attention),
        ("triton_flash", triton_attention),
    )

    print("| variant | mode | batch_size | seq_len | d_model | causal | time_ms |")
    print("| --- | --- | --- | --- | --- | --- | --- |")

    for d_model in args.d_models:
        for sequence_length in args.sequence_lengths:
            q_fwd, k_fwd, v_fwd = make_inputs(
                args.batch_size,
                sequence_length,
                d_model,
                device,
                dtype,
                requires_grad=False,
            )
            q_bwd, k_bwd, v_bwd = make_inputs(
                args.batch_size,
                sequence_length,
                d_model,
                device,
                dtype,
                requires_grad=True,
            )
            grad_output = torch.randn_like(v_bwd)

            for variant_name, attention_fn in variants:
                forward_ms = benchmark_forward(
                    attention_fn,
                    q_fwd,
                    k_fwd,
                    v_fwd,
                    args.causal,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                )
                backward_ms = benchmark_backward(
                    attention_fn,
                    q_bwd,
                    k_bwd,
                    v_bwd,
                    grad_output,
                    args.causal,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                )

                print(
                    f"| {variant_name} | forward | {args.batch_size} | {sequence_length} | "
                    f"{d_model} | {args.causal} | {forward_ms:.3f} |"
                )
                print(
                    f"| {variant_name} | backward | {args.batch_size} | {sequence_length} | "
                    f"{d_model} | {args.causal} | {backward_ms:.3f} |"
                )


if __name__ == "__main__":
    main()
