"""
Compatibility shim: original implementation moved to `cs336_basics.swiglu`.
Re-export for backwards compatibility.
"""
import warnings

warnings.warn("models.swiglu has moved to cs336_basics.swiglu; import from cs336_basics instead", DeprecationWarning)

from cs336_basics.swiglu import SwiGLU

__all__ = ["SwiGLU"]
