"""
Compatibility shim: original implementation moved to `cs336_basics.linear`.
This file re-exports the class for backwards compatibility and emits a deprecation
warning when imported. The substantive implementation now lives in
`cs336_basics/linear.py`.
"""
import warnings

warnings.warn("models.linear has moved to cs336_basics.linear; import from cs336_basics instead", DeprecationWarning)

from cs336_basics.linear import Linear

__all__ = ["Linear"]
