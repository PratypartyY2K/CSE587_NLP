"""
Facade shim: re-export implementations from `cs336_basics.impl`.

This module intentionally keeps a stable import path while delegating actual
implementations to the `impl` package.
"""
from __future__ import annotations

# Re-export everything from the impl package
from cs336_basics.impl import *  # noqa: F401,F403

__all__ = [name for name in dir() if not name.startswith("__")]
