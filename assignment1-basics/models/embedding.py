"""
Compatibility shim: original implementation moved to `cs336_basics.embedding`.
Re-export for backwards compatibility.
"""
import warnings

warnings.warn("models.embedding has moved to cs336_basics.embedding; import from cs336_basics instead", DeprecationWarning)

from cs336_basics.embedding import Embedding

__all__ = ["Embedding"]
