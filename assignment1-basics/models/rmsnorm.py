"""
Compatibility shim: original implementation moved to `cs336_basics.rmsnorm`.
Re-export for backwards compatibility.
"""
import warnings

warnings.warn("models.rmsnorm has moved to cs336_basics.rmsnorm; import from cs336_basics instead", DeprecationWarning)

from cs336_basics.rmsnorm import RMSNorm

__all__ = ["RMSNorm"]
