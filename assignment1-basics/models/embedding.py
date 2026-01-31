"""
Compatibility shim: original implementation moved to `cs336_basics.embedding`.
Re-export for backwards compatibility.
"""

import warnings
from cs336_basics.embedding import Embedding

warnings.warn(
    "models.embedding has moved to cs336_basics.embedding; import from cs336_basics instead",
    DeprecationWarning,
)

__all__ = ["Embedding"]
