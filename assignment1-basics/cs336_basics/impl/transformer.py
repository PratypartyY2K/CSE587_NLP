"""Transformer implementations: run_transformer_block_impl and run_transformer_lm_impl.

This file contains the canonical implementation; any previous `_impl` module re-exports
from here to avoid duplication.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from ..rmsnorm import RMSNorm
from ..swiglu import SwiGLU
from ..rope import RotaryPositionalEmbedding
from ..multihead_attention import MultiHeadSelfAttention
from .attention import run_multihead_self_attention_with_rope_impl


def run_transformer_block_impl(d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, weights: Dict[str, Tensor], in_features: Tensor) -> Tensor:
    """Run a single pre-norm Transformer block using provided weights.

    This mirrors the original implementation in adapters_impl but keeps local imports
    limited to avoid circular dependencies.
    """
    x = torch.as_tensor(in_features)
    seq_len = x.shape[-2]

    # LN1
    ln1_w = weights.get("ln1.weight")

    # create RMSNorm and load weight if present
    if isinstance(ln1_w, torch.Tensor):
        ln1 = RMSNorm(d_model=d_model, eps=1e-5, device=ln1_w.device, dtype=ln1_w.dtype)
        with torch.no_grad():
            ln1.weight.copy_(torch.as_tensor(ln1_w))
    else:
        ln1 = RMSNorm(d_model=d_model, eps=1e-5, device=x.device)

    # Apply pre-norm before attention
    x_ln1 = ln1(x)

    # Attention
    q_w = weights.get("attn.q_proj.weight")
    if q_w is None:
        q_w = weights.get("attn.q.weight")
    k_w = weights.get("attn.k_proj.weight")
    if k_w is None:
        k_w = weights.get("attn.k.weight")
    v_w = weights.get("attn.v_proj.weight")
    if v_w is None:
        v_w = weights.get("attn.v.weight")
    o_w = weights.get("attn.output_proj.weight")
    if o_w is None:
        o_w = weights.get("attn.o_proj.weight")
    if o_w is None:
        o_w = weights.get("attn.out_proj.weight")

    if q_w is not None and k_w is not None and v_w is not None and o_w is not None:
        attn_out = run_multihead_self_attention_with_rope_impl(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            q_proj_weight=q_w,
            k_proj_weight=k_w,
            v_proj_weight=v_w,
            o_proj_weight=o_w,
            in_features=x_ln1,
            token_positions=torch.arange(0, seq_len, device=x.device),
        )
    else:
        attn_out = torch.zeros_like(x)

    y = x + attn_out

    # LN2
    ln2_w = weights.get("ln2.weight")
    if isinstance(ln2_w, torch.Tensor):
        ln2 = RMSNorm(d_model=d_model, eps=1e-5, device=ln2_w.device, dtype=ln2_w.dtype)
        with torch.no_grad():
            ln2.weight.copy_(torch.as_tensor(ln2_w))
    else:
        ln2 = RMSNorm(d_model=d_model, eps=1e-5, device=x.device)

    ffn_in = ln2(y)

    # FFN: prefer SwiGLU (w1,w2,w3). If only w2/w3 present, do SiLU MLP
    w1 = weights.get("ffn.w1.weight")
    w2 = weights.get("ffn.w2.weight")
    w3 = weights.get("ffn.w3.weight")

    if w1 is not None and w2 is not None and w3 is not None:
        # build SwiGLU using weights
        W1 = torch.as_tensor(w1)
        W2 = torch.as_tensor(w2)
        W3 = torch.as_tensor(w3)
        swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=W1.device, dtype=W1.dtype)
        with torch.no_grad():
            swiglu.w1.copy_(W1)
            swiglu.w2.copy_(W2)
            swiglu.w3.copy_(W3)
        ffn_out = swiglu(ffn_in)
    elif w2 is not None and w3 is not None:
        up = torch.matmul(ffn_in, torch.as_tensor(w3).t())
        act = torch.nn.functional.silu(up)
        ffn_out = torch.matmul(act, torch.as_tensor(w2).t())
    else:
        ffn_out = torch.zeros_like(x)

    out = y + ffn_out
    return out


def run_transformer_lm_impl(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: Dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    """Construct a TransformerLM, load provided weights into its parameters, and run a forward pass."""
    from ..transformer import TransformerLM

    model = TransformerLM(vocab_size=vocab_size, context_length=context_length, d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, rope_theta=rope_theta)

    # Load token embeddings
    if "token_embeddings.weight" in weights:
        with torch.no_grad():
            model.token_embeddings.weight.copy_(weights["token_embeddings.weight"])
    # Load lm_head
    if "lm_head.weight" in weights:
        with torch.no_grad():
            model.lm_head.copy_(weights["lm_head.weight"])
    # Load ln_final
    if "ln_final.weight" in weights:
        with torch.no_grad():
            model.ln_final.weight.copy_(weights["ln_final.weight"])

    # Load layer weights
    for i in range(num_layers):
        prefix = f"layers.{i}."
        layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        if not layer_weights:
            continue
        block = model.blocks[i]
        # attention proj
        for name in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "output_proj.weight"]:
            key = prefix + ("attn." + name)
            if key in weights:
                if name == "output_proj.weight":
                    with torch.no_grad():
                        block.o_proj.copy_(weights[key])
                else:
                    param = getattr(block, name.split("_")[0] + "_proj")
                    with torch.no_grad():
                        param.copy_(weights[key])
        # ln1, ln2
        if prefix + "ln1.weight" in weights:
            with torch.no_grad():
                block.ln1.weight.copy_(weights[prefix + "ln1.weight"])
        if prefix + "ln2.weight" in weights:
            with torch.no_grad():
                block.ln2.weight.copy_(weights[prefix + "ln2.weight"])
        # ffn weights
        if prefix + "ffn.w1.weight" in weights:
            with torch.no_grad():
                block.w1.copy_(weights[prefix + "ffn.w1.weight"])
        if prefix + "ffn.w2.weight" in weights:
            with torch.no_grad():
                block.w2.copy_(weights[prefix + "ffn.w2.weight"])
        if prefix + "ffn.w3.weight" in weights:
            with torch.no_grad():
                block.w3.copy_(weights[prefix + "ffn.w3.weight"])

    x = torch.as_tensor(in_indices, dtype=torch.long)
    return model(x)
