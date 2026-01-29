"""
Compatibility shim: original implementation moved to `cs336_basics.rope`.
Re-export for backwards compatibility.
"""
import warnings

warnings.warn("models.rope has moved to cs336_basics.rope; import from cs336_basics instead", DeprecationWarning)

from cs336_basics.rope import RotaryPositionalEmbedding

__all__ = ["RotaryPositionalEmbedding"]
