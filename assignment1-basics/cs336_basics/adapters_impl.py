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

# Import transformer implementations
from .impl.transformer import (
    run_transformer_block_impl,
    run_transformer_lm_impl,
)

# Import tokenizer impls
from .impl.tokenizer import (
    train_bpe as run_train_bpe_impl,
    Tokenizer as ImplTokenizer,
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

# Transformer implementations are provided by the impl.transformer module (imported above)


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
