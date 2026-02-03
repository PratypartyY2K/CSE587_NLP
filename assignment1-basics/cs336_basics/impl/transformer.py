from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from ..rmsnorm import RMSNorm
from ..swiglu import SwiGLU
from .attention import multihead_self_attention_with_rope


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    attn_out_proj_weight: Tensor,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    rms1_weight: Tensor,
    rms2_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    Wq = torch.as_tensor(q_proj_weight)
    Wk = torch.as_tensor(k_proj_weight)
    Wv = torch.as_tensor(v_proj_weight)
    Wo = torch.as_tensor(o_proj_weight)

    x = torch.as_tensor(in_features)

    rms1 = RMSNorm(d_model=d_model, eps=1e-5, device=Wq.device, dtype=Wq.dtype)
    with torch.no_grad():
        rms1.weight.copy_(torch.as_tensor(rms1_weight))

    x_norm = rms1(x)
    attn_out = multihead_self_attention_with_rope(
        d_model=d_model,
        num_heads=num_heads,
        max_seq_len=x.shape[1],
        theta=10000.0,
        q_proj_weight=Wq,
        k_proj_weight=Wk,
        v_proj_weight=Wv,
        o_proj_weight=Wo,
        in_features=x_norm,
        token_positions=None,
    )
    x = x + attn_out

    rms2 = RMSNorm(d_model=d_model, eps=1e-5, device=Wq.device, dtype=Wq.dtype)
    with torch.no_grad():
        rms2.weight.copy_(torch.as_tensor(rms2_weight))

    x_norm2 = rms2(x)

    W1 = torch.as_tensor(w1_weight)
    W2 = torch.as_tensor(w2_weight)
    W3 = torch.as_tensor(w3_weight)
    swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=W1.device, dtype=W1.dtype)
    with torch.no_grad():
        swiglu.w1.copy_(W1)
        swiglu.w2.copy_(W2)
        swiglu.w3.copy_(W3)

    ff_out = swiglu(x_norm2)
    x = x + ff_out
    return x


def transformer_lm(
    vocab_size: int,
    context_length: int,
    num_layers: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    token_emb_weight: Tensor,
    pos_emb_weight: Tensor,
    blocks_weights: Iterable[dict],
    final_rms_weight: Tensor,
    lm_head_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    token_emb = torch.as_tensor(token_emb_weight)
    pos_emb = torch.as_tensor(pos_emb_weight)
    W = torch.as_tensor(in_features)

    x = torch.matmul(W, token_emb.t()) + pos_emb[: W.shape[1], :]

    for i in range(num_layers):
        b = blocks_weights[i]
        x = transformer_block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            q_proj_weight=b["q_proj"],
            k_proj_weight=b["k_proj"],
            v_proj_weight=b["v_proj"],
            o_proj_weight=b["o_proj"],
            attn_out_proj_weight=b.get("attn_out_proj", b["o_proj"]),
            w1_weight=b["w1"],
            w2_weight=b["w2"],
            w3_weight=b["w3"],
            rms1_weight=b["rms1"],
            rms2_weight=b["rms2"],
            in_features=x,
        )

    final_rms = RMSNorm(d_model=d_model, eps=1e-5, device=W.device, dtype=W.dtype)
    with torch.no_grad():
        final_rms.weight.copy_(torch.as_tensor(final_rms_weight))
    x = final_rms(x)

    logits = torch.matmul(x, torch.as_tensor(lm_head_weight).t())
    return logits


def transformer_block_from_weights(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict,
    in_features: Tensor,
) -> Tensor:
    q_proj = weights.get("attn.q_proj.weight")
    k_proj = weights.get("attn.k_proj.weight")
    v_proj = weights.get("attn.v_proj.weight")
    o_proj = weights.get("attn.output_proj.weight", weights.get("attn.o_proj.weight", weights.get("attn.output.weight")))

    w1 = weights.get("ffn.w1.weight")
    w2 = weights.get("ffn.w2.weight")
    w3 = weights.get("ffn.w3.weight")

    rms1 = weights.get("ln1.weight", weights.get("rms1.weight"))
    rms2 = weights.get("ln2.weight", weights.get("rms2.weight"))

    return transformer_block(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        q_proj_weight=q_proj,
        k_proj_weight=k_proj,
        v_proj_weight=v_proj,
        o_proj_weight=o_proj,
        attn_out_proj_weight=o_proj,
        w1_weight=w1,
        w2_weight=w2,
        w3_weight=w3,
        rms1_weight=rms1,
        rms2_weight=rms2,
        in_features=in_features,
    )


def transformer_lm_from_weights(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict,
    in_indices: Tensor,
) -> Tensor:
    token_emb = weights.get("token_embeddings.weight")
    lm_head = weights.get("lm_head.weight")
    if token_emb is None and lm_head is not None:
        token_emb = lm_head
    if token_emb is None:
        token_emb = torch.zeros((vocab_size, d_model), dtype=torch.float32)

    pos_emb = weights.get("pos_embeddings.weight")
    if pos_emb is None:
        pos_emb = torch.zeros((context_length, d_model), dtype=token_emb.dtype)

    ids = torch.as_tensor(in_indices, dtype=torch.long)
    x = torch.as_tensor(token_emb, dtype=torch.float32)[ids]
    seq_len = x.shape[1]
    x = x + pos_emb[:seq_len, :].unsqueeze(0)

    for i in range(num_layers):
        prefix = f"layers.{i}."
        b = {}
        for k, v in weights.items():
            if k.startswith(prefix):
                b[k[len(prefix):]] = v
        x = transformer_block_from_weights(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            max_seq_len=context_length,
            theta=rope_theta,
            weights=b,
            in_features=x,
        )

    final_rms = weights.get("ln_final.weight")
    if final_rms is not None:
        from ..rmsnorm import RMSNorm as _RMS

        rms = _RMS(d_model=d_model)
        with torch.no_grad():
            rms.weight.copy_(torch.as_tensor(final_rms))
        x = rms(x)

    lm_w = lm_head if lm_head is not None else token_emb
    logits = torch.matmul(x, torch.as_tensor(lm_w).t())
    return logits
