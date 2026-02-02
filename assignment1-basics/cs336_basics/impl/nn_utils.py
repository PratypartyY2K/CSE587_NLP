"""Utilities and small NN-layer adapters.

Contains focused implementations used by tests: softmax, SiLU, batch sampling,
cross-entropy, gradient clipping, and small adapter runners (linear, embedding, SwiGLU, RMSNorm).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor

from ..linear import Linear
from ..embedding import Embedding
from ..swiglu import SwiGLU
from ..rmsnorm import RMSNorm


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


def run_linear_impl(d_in: int, d_out: int, weights: Tensor, in_features: Tensor) -> Tensor:
    W = torch.as_tensor(weights)
    model = Linear(
        in_features=d_in,
        out_features=d_out,
        device=W.device if isinstance(W, torch.Tensor) else None,
        dtype=W.dtype if isinstance(W, torch.Tensor) else None,
    )
    with torch.no_grad():
        model.W.copy_(W)
    return model(torch.as_tensor(in_features))


def run_embedding_impl(vocab_size: int, d_model: int, weights: Tensor, token_ids: Tensor) -> Tensor:
    W = torch.as_tensor(weights)
    emb = Embedding(
        num_embeddings=vocab_size,
        embedding_dim=d_model,
        device=W.device if isinstance(W, torch.Tensor) else None,
        dtype=W.dtype if isinstance(W, torch.Tensor) else None,
    )
    with torch.no_grad():
        emb.weight.copy_(W)
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    return emb(ids)


def run_swiglu_impl(
    d_model: int, d_ff: int, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor, in_features: Tensor
) -> Tensor:
    W1 = torch.as_tensor(w1_weight)
    W2 = torch.as_tensor(w2_weight)
    W3 = torch.as_tensor(w3_weight)
    swiglu = SwiGLU(
        d_model=d_model,
        d_ff=d_ff,
        device=W1.device if isinstance(W1, torch.Tensor) else None,
        dtype=W1.dtype if isinstance(W1, torch.Tensor) else None,
    )
    with torch.no_grad():
        swiglu.w1.copy_(W1)
        swiglu.w2.copy_(W2)
        swiglu.w3.copy_(W3)
    return swiglu(torch.as_tensor(in_features))


def run_rmsnorm_impl(d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    w = torch.as_tensor(weights)
    rms = RMSNorm(
        d_model=d_model,
        eps=eps,
        device=w.device if isinstance(w, torch.Tensor) else None,
        dtype=w.dtype if isinstance(w, torch.Tensor) else None,
    )
    with torch.no_grad():
        rms.weight.copy_(w)
    return rms(torch.as_tensor(in_features))
