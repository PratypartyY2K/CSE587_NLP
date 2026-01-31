"""Implementation package facade for cs336_basics.impl.

This module re-exports named implementation functions and classes so callers can
import them from `cs336_basics.impl`.
"""

from .nn_utils import (
    run_softmax_impl,
    run_silu_impl,
    run_get_batch_impl,
    run_cross_entropy_impl,
    run_gradient_clipping_impl,
    run_linear_impl,
    run_embedding_impl,
    run_swiglu_impl,
    run_rmsnorm_impl,
)
from .attention import (
    run_scaled_dot_product_attention_impl,
    run_multihead_self_attention_impl,
    run_rope_impl,
    run_multihead_self_attention_with_rope_impl,
)
from .transformer import (
    run_transformer_block_impl,
    run_transformer_lm_impl,
)
from .tokenizer import train_bpe, Tokenizer
from .optimizer import AdamW, run_get_lr_cosine_schedule_impl
from .io import run_save_checkpoint_impl, run_load_checkpoint_impl

__all__ = [
    "run_softmax_impl",
    "run_silu_impl",
    "run_get_batch_impl",
    "run_cross_entropy_impl",
    "run_gradient_clipping_impl",
    "run_scaled_dot_product_attention_impl",
    "run_multihead_self_attention_impl",
    "run_rope_impl",
    "run_multihead_self_attention_with_rope_impl",
    "run_transformer_block_impl",
    "run_transformer_lm_impl",
    "train_bpe",
    "Tokenizer",
    "AdamW",
    "run_get_lr_cosine_schedule_impl",
    "run_save_checkpoint_impl",
    "run_load_checkpoint_impl",
    # nn layer helpers
    "run_linear_impl",
    "run_embedding_impl",
    "run_swiglu_impl",
    "run_rmsnorm_impl",
]
