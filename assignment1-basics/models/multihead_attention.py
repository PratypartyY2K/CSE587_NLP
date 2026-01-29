"""
Compatibility shim: original implementation moved to `cs336_basics.multihead_attention`.
Re-export for backwards compatibility.
"""
import warnings

warnings.warn("models.multihead_attention has moved to cs336_basics.multihead_attention; import from cs336_basics instead", DeprecationWarning)

from cs336_basics.multihead_attention import MultiHeadSelfAttention

__all__ = ["MultiHeadSelfAttention"]
