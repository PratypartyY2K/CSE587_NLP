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

# Import modularized nn utils
from .impl.nn_utils import (
    run_softmax_impl,
    run_silu_impl,
    run_get_batch_impl,
    run_cross_entropy_impl,
    run_gradient_clipping_impl,
)

# Import attention/rope implementations
from .impl.attention import (
    run_scaled_dot_product_attention_impl,
    run_multihead_self_attention_impl,
    run_rope_impl,
    run_multihead_self_attention_with_rope_impl,
)


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


# The following functions are provided by the modularized nn_utils module (imported above):
# - run_softmax_impl
# - run_silu_impl
# - run_get_batch_impl
# - run_cross_entropy_impl
# - run_gradient_clipping_impl


# Attention functions are imported from impl.attention


def run_transformer_block_impl(d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, weights: dict[str, Tensor], in_features: Tensor) -> Tensor:
    """Run a single pre-norm Transformer block using provided weights.

    This function is intentionally permissive: if attention weights are missing,
    attention is treated as a zero operator (identity residual). For the FFN,
    if SwiGLU weights (w1,w2,w3) are present we use SwiGLU; otherwise if only
    w2 and w3 are present we perform a 2-layer SiLU MLP using w3 as the up-proj
    and w2 as the down-proj.
    """
    import torch
    from torch import nn

    x = torch.as_tensor(in_features)
    seq_len = x.shape[-2]

    # LN1
    ln1_w = weights.get("ln1.weight")
    from .rmsnorm import RMSNorm

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
    # Look for attn weights with common keys (explicit None checks to avoid tensor boolean ambiguity)
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
        # Use RoPE-enabled multihead attention implementation
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
        # No attention weights provided: treat attention as 0 -> residual is identity
        attn_out = torch.zeros_like(x)

    y = x + attn_out

    # LN2
    ln2_w = weights.get("ln2.weight")
    ln2 = RMSNorm(d_model=d_model, eps=1e-5, device=ln2_w.device if isinstance(ln2_w, torch.Tensor) else x.device, dtype=ln2_w.dtype if isinstance(ln2_w, torch.Tensor) else None)
    if ln2_w is not None:
        with torch.no_grad():
            ln2.weight.copy_(torch.as_tensor(ln2_w))

    ffn_in = ln2(y)

    # FFN: prefer SwiGLU (w1,w2,w3). If only w2/w3 present, do SiLU MLP: out = SiLU(ffn_up(x)) @ w2.T
    w1 = weights.get("ffn.w1.weight")
    w2 = weights.get("ffn.w2.weight")
    w3 = weights.get("ffn.w3.weight")

    if w1 is not None and w2 is not None and w3 is not None:
        # Use SwiGLU implementation
        ffn_out = run_swiglu_impl(d_model=d_model, d_ff=d_ff, w1_weight=w1, w2_weight=w2, w3_weight=w3, in_features=ffn_in)
    elif w2 is not None and w3 is not None:
        # Use two-layer MLP: up = x @ w3.T  (d_ff), act=SiLU(up), down = act @ w2.T
        up = torch.matmul(ffn_in, torch.as_tensor(w3).t())
        act = torch.nn.functional.silu(up)
        ffn_out = torch.matmul(act, torch.as_tensor(w2).t())
    else:
        # No FFN weights provided
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
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    """Construct a TransformerLM, load provided weights into its parameters, and run a forward pass."""
    from .transformer import TransformerLM

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
                # map attn.output_proj.weight -> o_proj
                if name == "output_proj.weight":
                    with torch.no_grad():
                        block.o_proj.copy_(weights[key])
                else:
                    # q,k,v
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


# run_cross_entropy_impl and run_gradient_clipping_impl are provided by the imported nn_utils module


def run_get_lr_cosine_schedule_impl(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """Return learning rate at iteration `it` using linear warmup followed by cosine annealing.

    Warmup: 0 -> max_learning_rate over `warmup_iters` (inclusive: at t=warmup_iters lr=max_learning_rate).
    Cosine: after warmup, anneal from max_learning_rate to min_learning_rate over
    `num_cosine_steps = cosine_cycle_iters - warmup_iters` iterations (inclusive: at t=cosine_cycle_iters lr=min_learning_rate).
    After `cosine_cycle_iters`, return min_learning_rate.
    """
    t = int(it)
    alpha_max = float(max_learning_rate)
    alpha_min = float(min_learning_rate)
    Tw = int(warmup_iters)
    Tc = int(cosine_cycle_iters)

    if Tw <= 0:
        # No warmup: start at max and immediately begin cosine from t=0
        if Tc <= 0 or t >= Tc:
            return alpha_min
        # treat whole interval as cosine
        num_cosine_steps = max(1, Tc)
        progress = max(0, t)
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress / num_cosine_steps))
        return alpha_min + (alpha_max - alpha_min) * cos_val

    # Warmup phase (including t == Tw)
    if t <= Tw:
        return alpha_max * (t / Tw)

    # Cosine annealing phase
    if t <= Tc:
        num_cosine_steps = max(1, Tc - Tw)
        progress = t - Tw
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress / num_cosine_steps))
        return alpha_min + (alpha_max - alpha_min) * cos_val

    # After annealing: hold at minimum
    return alpha_min


# run_gradient_clipping_impl provided by imported nn_utils


def run_save_checkpoint_impl(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | "BinaryIO" | "IO[bytes]",
):
    """Serialize model state_dict, optimizer state_dict, and iteration to `out`.

    `out` may be a path or a file-like object. This function uses torch.save.
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    # torch.save accepts path-like or file-like objects
    torch.save(payload, out)


def run_load_checkpoint_impl(
    src: str | os.PathLike | "BinaryIO" | "IO[bytes]",
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Load a checkpoint from `src` and restore state to model and optimizer.

    Returns the iteration number stored in the checkpoint.
    """
    payload = torch.load(src, map_location="cpu")
    # restore model and optimizer
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload["iteration"])
