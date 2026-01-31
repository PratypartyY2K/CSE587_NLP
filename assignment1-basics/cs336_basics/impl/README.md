# cs336_basics.impl

This package contains the concrete implementations used by the `cs336_basics` adapters.
The goal is to collect substantive logic here and keep `cs336_basics.adapters_impl` as a
thin facade that re-exports the implementation functions.

Modules
- `nn_utils.py` - small stateless neural-network utilities (softmax, SiLU, cross-entropy,
  batch sampling, gradient clipping).
- `nn_layers.py` - helpers that wrap small NN layers (Linear, Embedding, SwiGLU, RMSNorm)
  and expose factory-like run_* functions used by the adapter tests.
- `attention.py` - scaled dot-product attention, multi-head self-attention, and RoPE helpers.
- `transformer.py` - Transformer block and Transformer LM helper functions (run_transformer_block_impl,
  run_transformer_lm_impl).
- `tokenizer.py` - BPE `train_bpe` and `Tokenizer` implementation.
- `optimizer.py` - optimizer helpers (a local AdamW implementation and cosine LR schedule).
- `io.py` - checkpoint save/load helpers (run_save_checkpoint_impl, run_load_checkpoint_impl).

Usage
The tests and the adapter shim import names from `cs336_basics.adapters_impl`, which re-exports
all public functions from this package. When editing implementations here, keep the API
(signatures and semantics) and the names stable to avoid breaking the thin adapter shim.

Why this layout
- Keeps domain-specific logic grouped and easier to test and maintain.
- Facilitates incremental refactors: `cs336_basics.adapters_impl` remains the single import
  target for the rest of the codebase and tests.

If you want to add more modules, follow the same pattern: implement functionality here and
export the public names in `cs336_basics.impl.__init__.py` so the adapter shim continues to work.
