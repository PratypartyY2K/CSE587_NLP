"""
Implementation module: substantive logic moved here so `tests/adapters.py` remains thin glue.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .linear import Linear
from .embedding import Embedding
from .swiglu import SwiGLU
from .rmsnorm import RMSNorm
from .rope import RotaryPositionalEmbedding


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


def run_scaled_dot_product_attention_impl(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    q = torch.as_tensor(Q)
    k = torch.as_tensor(K)
    v = torch.as_tensor(V)
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2).to(q.dtype)) / math.sqrt(d_k)
    if mask is not None:
        m = torch.as_tensor(mask, dtype=torch.bool)
        while m.dim() < scores.dim():
            m = m.unsqueeze(0)
        scores = scores.masked_fill(~m, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out


def run_multihead_self_attention_impl(d_model: int, num_heads: int, q_proj_weight: Tensor, k_proj_weight: Tensor, v_proj_weight: Tensor, o_proj_weight: Tensor, in_features: Tensor) -> Tensor:
    Wq = torch.as_tensor(q_proj_weight)
    Wk = torch.as_tensor(k_proj_weight)
    Wv = torch.as_tensor(v_proj_weight)
    Wo = torch.as_tensor(o_proj_weight)
    from .multihead_attention import MultiHeadSelfAttention
    mha = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, device=Wq.device if isinstance(Wq, torch.Tensor) else None, dtype=Wq.dtype if isinstance(Wq, torch.Tensor) else None)
    with torch.no_grad():
        mha.q_proj.copy_(Wq)
        mha.k_proj.copy_(Wk)
        mha.v_proj.copy_(Wv)
        mha.o_proj.copy_(Wo)
    x = torch.as_tensor(in_features)
    return mha(x)


def run_rope_impl(d_k: int, theta: float, max_seq_len: int, in_query_or_key: Tensor, token_positions: Tensor) -> Tensor:
    x = torch.as_tensor(in_query_or_key)
    pos = torch.as_tensor(token_positions)
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=x.device)
    return rope(x, pos)


def run_multihead_self_attention_with_rope_impl(d_model: int, num_heads: int, max_seq_len: int, theta: float, q_proj_weight: Tensor, k_proj_weight: Tensor, v_proj_weight: Tensor, o_proj_weight: Tensor, in_features: Tensor, token_positions: Tensor | None = None) -> Tensor:
    Wq = torch.as_tensor(q_proj_weight)
    Wk = torch.as_tensor(k_proj_weight)
    Wv = torch.as_tensor(v_proj_weight)
    Wo = torch.as_tensor(o_proj_weight)
    x = torch.as_tensor(in_features)
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")
    d_k = d_model // num_heads
    q = torch.matmul(x, Wq.t())
    k = torch.matmul(x, Wk.t())
    v = torch.matmul(x, Wv.t())
    *lead_dims, seq_len, _ = q.shape
    q = q.view(*lead_dims, seq_len, num_heads, d_k).permute(*list(range(len(lead_dims))), -2, -3, -1)
    k = k.view(*lead_dims, seq_len, num_heads, d_k).permute(*list(range(len(lead_dims))), -2, -3, -1)
    v = v.view(*lead_dims, seq_len, num_heads, d_k).permute(*list(range(len(lead_dims))), -2, -3, -1)
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=x.device)
    if token_positions is None:
        pos = torch.arange(0, seq_len, device=x.device)
    else:
        pos = torch.as_tensor(token_positions, device=x.device)
    q = rope(q, pos)
    k = rope(k, pos)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    device = scores.device
    causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    while causal.dim() < scores.dim():
        causal = causal.unsqueeze(0)
    scores = scores.masked_fill(~causal, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    out = out.permute(*list(range(len(lead_dims))), -2, -3, -1).contiguous()
    out = out.view(*lead_dims, seq_len, d_model)
    out = torch.matmul(out, Wo.t())
    return out


def run_rmsnorm_impl(d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    w = torch.as_tensor(weights)
    rms = RMSNorm(d_model=d_model, eps=eps, device=w.device if isinstance(w, torch.Tensor) else None, dtype=w.dtype if isinstance(w, torch.Tensor) else None)
    with torch.no_grad():
        rms.weight.copy_(w)
    return rms(torch.as_tensor(in_features))


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
    max_start = n - context_length - 1 + 1  # inclusive end for start selection so that x+1 valid
    # sample starts uniformly
    starts = np.random.randint(0, n - context_length, size=(batch_size,))
    x_batch = np.stack([arr[s : s + context_length] for s in starts], axis=0)
    y_batch = np.stack([arr[s + 1 : s + 1 + context_length] for s in starts], axis=0)
    x_t = torch.as_tensor(x_batch, dtype=torch.long, device=device)
    y_t = torch.as_tensor(y_batch, dtype=torch.long, device=device)
    return x_t, y_t


def run_transformer_block_impl(*args, **kwargs):
    """Placeholder for transformer block implementation. Implemented later in cs336_basics.
    Present so tests can import the symbol from adapters.
    """
    raise NotImplementedError("run_transformer_block_impl is not implemented yet")


def run_transformer_lm_impl(*args, **kwargs):
    """Placeholder for transformer LM forward; implemented later.
    Exposed so tests can import symbol from adapters.
    """
    raise NotImplementedError("run_transformer_lm_impl is not implemented yet")
