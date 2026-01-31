"""Neural-network layer adapter functions.

Exports:
- run_linear_impl
- run_embedding_impl
- run_swiglu_impl
- run_rmsnorm_impl
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..linear import Linear
from ..embedding import Embedding
from ..swiglu import SwiGLU
from ..rmsnorm import RMSNorm


def run_linear_impl(d_in: int, d_out: int, weights: Tensor, in_features: Tensor) -> Tensor:
    W = torch.as_tensor(weights)
    model = Linear(in_features=d_in, out_features=d_out, device=W.device if isinstance(W, torch.Tensor) else None, dtype=W.dtype if isinstance(W, torch.Tensor) else None)
    with torch.no_grad():
        model.W.copy_(W)
    return model(torch.as_tensor(in_features))


def run_embedding_impl(vocab_size: int, d_model: int, weights: Tensor, token_ids: Tensor) -> Tensor:
    W = torch.as_tensor(weights)
    emb = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=W.device if isinstance(W, torch.Tensor) else None, dtype=W.dtype if isinstance(W, torch.Tensor) else None)
    with torch.no_grad():
        emb.weight.copy_(W)
    ids = torch.as_tensor(token_ids, dtype=torch.long)
    return emb(ids)


def run_swiglu_impl(d_model: int, d_ff: int, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor, in_features: Tensor) -> Tensor:
    W1 = torch.as_tensor(w1_weight)
    W2 = torch.as_tensor(w2_weight)
    W3 = torch.as_tensor(w3_weight)
    swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=W1.device if isinstance(W1, torch.Tensor) else None, dtype=W1.dtype if isinstance(W1, torch.Tensor) else None)
    with torch.no_grad():
        swiglu.w1.copy_(W1)
        swiglu.w2.copy_(W2)
        swiglu.w3.copy_(W3)
    return swiglu(torch.as_tensor(in_features))


def run_rmsnorm_impl(d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    w = torch.as_tensor(weights)
    rms = RMSNorm(d_model=d_model, eps=eps, device=w.device if isinstance(w, torch.Tensor) else None, dtype=w.dtype if isinstance(w, torch.Tensor) else None)
    with torch.no_grad():
        rms.weight.copy_(w)
    return rms(torch.as_tensor(in_features))
