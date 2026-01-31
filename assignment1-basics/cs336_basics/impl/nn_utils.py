"""Small utility functions extracted from adapters_impl for modularity.

Exports:
- run_softmax_impl
- run_silu_impl
- run_get_batch_impl
- run_cross_entropy_impl
- run_gradient_clipping_impl
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor


def run_softmax_impl(in_features: Tensor, dim: int) -> Tensor:
    x = torch.as_tensor(in_features)
    max_along_dim = torch.amax(x, dim=dim, keepdim=True)
    x_stable = x - max_along_dim
    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


def run_silu_impl(in_features: Tensor) -> Tensor:
    x = torch.as_tensor(in_features)
    return torch.nn.functional.silu(x)


def run_get_batch_impl(dataset, batch_size: int, context_length: int, device: str):
    """Sample language-modeling batches from a 1D numpy array dataset.
    Returns x,y torch.LongTensor on the specified device with shapes (batch_size, context_length).
    """
    import numpy as np
    import torch

    arr = np.asarray(dataset)
    n = arr.shape[0]
    if n <= context_length:
        raise ValueError("dataset too small for context_length")
    # sample starts uniformly
    starts = np.random.randint(0, n - context_length, size=(batch_size,))
    x_batch = np.stack([arr[s : s + context_length] for s in starts], axis=0)
    y_batch = np.stack([arr[s + 1 : s + 1 + context_length] for s in starts], axis=0)
    x_t = torch.as_tensor(x_batch, dtype=torch.long, device=device)
    y_t = torch.as_tensor(y_batch, dtype=torch.long, device=device)
    return x_t, y_t


def run_cross_entropy_impl(inputs: Tensor, targets: Tensor) -> Tensor:
    x = torch.as_tensor(inputs)
    t = torch.as_tensor(targets, dtype=torch.long)

    vocab_size = x.shape[-1]
    x_flat = x.view(-1, vocab_size)
    t_flat = t.view(-1)

    lse = torch.logsumexp(x_flat, dim=-1)
    x_at_target = x_flat.gather(1, t_flat.unsqueeze(1)).squeeze(1)
    loss_per_example = lse - x_at_target
    return loss_per_example.mean()


def run_gradient_clipping_impl(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """Clip gradients of the provided parameters in-place to have total L2 norm at most max_l2_norm.

    Mirrors torch.nn.utils.clip_grad_norm_. Uses eps=1e-6.
    """
    total_norm_sq = 0.0
    grads = []
    for p in parameters:
        if p is None:
            continue
        if not hasattr(p, "grad"):
            continue
        grad = p.grad
        if grad is None:
            continue
        grad_data = grad.detach()
        total_norm_sq += float(torch.sum(grad_data.float() * grad_data.float()).item())
        grads.append(grad)

    total_norm = math.sqrt(total_norm_sq)
    if total_norm == 0:
        return
    clip_coef = float(max_l2_norm) / (total_norm + eps)
    if clip_coef >= 1.0:
        return
    for g in grads:
        g.mul_(clip_coef)
