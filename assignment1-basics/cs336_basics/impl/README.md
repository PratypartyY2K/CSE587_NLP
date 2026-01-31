# cs336_basics.impl

This package contains the canonical, concrete implementations used by the cs336_basics codebase.
We intentionally keep all substantive logic here so callers (tests and scripts) can import a
stable, single source of truth: `cs336_basics.impl`.

Quick notes
- There are no "shim" modules left at the top-level. Import implementations from `cs336_basics.impl`.
- The impl package is the single place to add or change implementation logic — keep APIs stable.

Modules (canonical)
- `nn_utils.py` — small stateless utilities and light NN-layer adapters. Exports:
  `run_softmax_impl`, `run_silu_impl`, `run_get_batch_impl`, `run_cross_entropy_impl`,
  `run_gradient_clipping_impl`, and small layer helpers `run_linear_impl`, `run_embedding_impl`,
  `run_swiglu_impl`, `run_rmsnorm_impl`.
- `attention.py` — scaled dot-product attention, multi-head self-attention, and RoPE helpers.
  Exports: `run_scaled_dot_product_attention_impl`, `run_multihead_self_attention_impl`,
  `run_rope_impl`, `run_multihead_self_attention_with_rope_impl`.
- `transformer.py` — Transformer block and Transformer LM helpers. Exports:
  `run_transformer_block_impl`, `run_transformer_lm_impl`.
- `tokenizer.py` — Byte-level BPE training (`train_bpe`) and a `Tokenizer` class.
- `optimizer.py` — Optimizer helpers (local `AdamW` class and `run_get_lr_cosine_schedule_impl`).
- `io.py` — Small I/O helpers for checkpoint save/load: `run_save_checkpoint_impl`, `run_load_checkpoint_impl`.
- `__init__.py` — package facade that re-exports the public names above for convenience.

Why this layout
- Single source of truth: implementations live under `cs336_basics.impl` so there is no duplication
  or accidental drift between shim and impl code.
- Tests and scripts import directly from `cs336_basics.impl` (see Usage below).
- Keep `impl` focused on implementation; higher-level orchestration or CLI code stays in `scripts/`.

Usage examples
- From tests or other code, import the implementation functions/classes you need:

```python
from cs336_basics.impl import Tokenizer, train_bpe
from cs336_basics.impl import run_scaled_dot_product_attention_impl
from cs336_basics.impl import AdamW, run_get_lr_cosine_schedule_impl
```

- To train a tokenizer and serialize outputs (example):

```python
vocab, merges = train_bpe('data/tinystories_sample_small.txt', vocab_size=10000, special_tokens=['<|endoftext|>'])
# persist vocab/merges as desired (pickle/json/text)
```

Contributing and making changes
- Add new functionality inside `cs336_basics/impl/` and export it from `impl/__init__.py` if it should be public.
- Avoid adding top-level shims that re-export impl functions — tests and scripts now import `cs336_basics.impl` directly.
- Run the full test suite before committing: `pytest -q`.

Developer tips
- Keep function signatures and names stable; tests call specific hooks (see `tests/adapters.py`).
- If you need to split a large module, prefer grouping related helpers in the same `impl` package (e.g.,
  small utilities together in `nn_utils.py`) rather than creating many tiny files.

Contact / Notes
- If you need help deciding where to add a new helper, put it in `nn_utils.py` or a new clearly named
  module under `impl/` and export it via `impl/__init__.py`.

---
Updated on: 2026-01-31
