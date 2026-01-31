"""Attention helpers: scaled dot-product, multi-head, and RoPE wrappers.

Implements core attention routines used by transformer blocks and tests.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..rope import RotaryPositionalEmbedding


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


def run_multihead_self_attention_impl(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    Wq = torch.as_tensor(q_proj_weight)
    Wk = torch.as_tensor(k_proj_weight)
    Wv = torch.as_tensor(v_proj_weight)
    Wo = torch.as_tensor(o_proj_weight)
    from ..multihead_attention import MultiHeadSelfAttention

    mha = MultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        device=Wq.device if isinstance(Wq, torch.Tensor) else None,
        dtype=Wq.dtype if isinstance(Wq, torch.Tensor) else None,
    )
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


def run_multihead_self_attention_with_rope_impl(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor | None = None,
) -> Tensor:
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
