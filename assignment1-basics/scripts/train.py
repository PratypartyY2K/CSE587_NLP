"""Training script for the TransformerLM model.

Features:
- Configure model and optimizer hyperparameters via command-line args.
- Memory-efficient dataset loading via numpy.memmap (np.load(..., mmap_mode='r')).
- Checkpoint save/load using the impl IO helpers.
- Periodic evaluation and console logging; optional Weights & Biases logging if installed.

Usage (quick):
python scripts/train.py --train-data out/tiny_train_ids.npy --valid-data out/tiny_valid_ids.npy \
    --vocab-size 10000 --context-length 512 --d-model 512 --num-layers 6 --num-heads 8 \
    --batch-size 32 --lr 1e-3 --total-steps 10000 --device cpu --checkpoint out/checkpoint.pt
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.transformer import TransformerLM
from cs336_basics.impl import (
    AdamW,
    run_get_batch_impl,
    run_save_checkpoint_impl,
    run_load_checkpoint_impl,
    run_gradient_clipping_impl,
)


def load_memmap(path: str, dtype: str | None = None) -> np.ndarray:
    """Load a numpy array in memory-mapped read-only mode if possible."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # Use mmap_mode='r' for .npy files
    try:
        arr = np.load(path, mmap_mode="r")
    except Exception:
        # Fall back to normal load (works for plain files)
        arr = np.load(path)
    if dtype is not None and str(arr.dtype) != dtype:
        arr = arr.astype(dtype)
    return arr


def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_eval_batches: int = 10,
):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_eval_batches):
            x, y = run_get_batch_impl(dataset, batch_size, context_length, device)
            logits = model(x)  # (B, L, V)
            B, L, V = logits.shape
            loss = F.cross_entropy(logits.view(B * L, V), y.view(B * L), reduction="mean")
            total_loss += float(loss.item())
    model.train()
    return total_loss / max(1, num_eval_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True, help="Path to train ids .npy memmap")
    parser.add_argument("--valid-data", required=False, help="Path to valid ids .npy memmap")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint output path")

    # Model hyperparameters
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=0, help="If 0, uses 4*d_model")

    # Optimization
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--start-step", type=int, default=0)

    args = parser.parse_args()

    device = args.device
    context_length = args.context_length
    d_ff = args.d_ff if args.d_ff and args.d_ff > 0 else 4 * args.d_model

    print("Loading datasets...")
    train_data = load_memmap(args.train_data)
    valid_data = None
    if args.valid_data:
        valid_data = load_memmap(args.valid_data)

    print("Building model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=d_ff,
        rope_theta=10000.0,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)

    # Attempt to resume from checkpoint if exists
    start_step = int(args.start_step)
    if os.path.exists(args.checkpoint):
        try:
            step = run_load_checkpoint_impl(args.checkpoint, model, optimizer)
            print(f"Loaded checkpoint '{args.checkpoint}' at iteration {step}")
            start_step = int(step)
        except Exception as e:
            print(f"Warning: could not load checkpoint {args.checkpoint}: {e}")

    print(f"Starting training on device={device} from step {start_step} to {args.total_steps}")

    # Training loop
    t0 = time.time()
    report_time = t0
    for it in range(start_step, args.total_steps):
        x, y = run_get_batch_impl(train_data, args.batch_size, context_length, device)

        logits = model(x)  # (B, L, V)
        B, L, V = logits.shape
        loss = F.cross_entropy(logits.view(B * L, V), y.view(B * L), reduction="mean")

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            run_gradient_clipping_impl(model.parameters(), args.grad_clip)
        optimizer.step()

        # simple LR schedule hook (optional): here we don't change lr per step automatically
        if (it + 1) % args.log_interval == 0:
            now = time.time()
            elapsed = now - report_time
            steps = args.log_interval
            print(f"step={it + 1}/{args.total_steps} loss={loss.item():.6f} time_per_{steps}steps={elapsed:.2f}s")
            report_time = now

        if valid_data is not None and (it + 1) % args.eval_interval == 0:
            val_loss = evaluate(model, valid_data, args.batch_size, context_length, device, num_eval_batches=5)
            print(f"[eval] step={it + 1} val_loss={val_loss:.6f}")

        if (it + 1) % args.save_interval == 0:
            # Save checkpoint
            try:
                run_save_checkpoint_impl(model, optimizer, it + 1, args.checkpoint)
                print(f"Saved checkpoint to {args.checkpoint} at step {it + 1}")
            except Exception as e:
                print(f"Error saving checkpoint to {args.checkpoint}: {e}")

    total_time = time.time() - t0
    print(f"Training complete. Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
