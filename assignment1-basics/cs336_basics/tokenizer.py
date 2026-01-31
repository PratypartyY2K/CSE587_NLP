"""Public tokenizer shim that re-exports implementations from cs336_basics.impl.tokenizer.

This file intentionally keeps the public API stable while the implementation
lives in `cs336_basics.impl.tokenizer`.
"""
from .impl.tokenizer import train_bpe, Tokenizer

__all__ = ["train_bpe", "Tokenizer"]
