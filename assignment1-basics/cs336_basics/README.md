# cs336_basics

This package contains the model-level code used in the assignment: small neural-network
building blocks, RoPE, multi-head attention, and a canonical `TransformerLM` implementation.

Note: Most tests and higher-level utilities import the concrete implementations from
`cs336_basics.impl`. The `cs336_basics` package provides model classes and layer definitions
that are convenient for direct importing, quick experiments, or for use when building models
programmatically.

Contents (high level)
- `linear.py` — lightweight `Linear` module compatible with PyTorch's interface (no bias).
- `embedding.py` — simple `Embedding` implementation used by `TransformerLM`.
- `swiglu.py` — SwiGLU feed-forward block used in the Transformer block.
- `rmsnorm.py` — RMSNorm implementation.
- `rope.py` — Rotary positional embeddings (RoPE) module.
- `multihead_attention.py` — a minimal causal multi-head attention implementation.
- `transformer.py` — `TransformerBlock` and `TransformerLM` classes that wire the pieces together.
- `tokenizer.py` — historical shim (the canonical tokenizer implementation lives in
  `cs336_basics/impl/tokenizer.py` and is re-exported from `cs336_basics.impl`).

Quick usage examples
--------------------
Constructing and running a small Transformer language model (toy example):

```python
import torch
from cs336_basics.transformer import TransformerLM

model = TransformerLM(
    vocab_size=10000,
    context_length=128,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048,
    rope_theta=10000.0,
)
# forward pass with a toy batch of token ids
x = torch.randint(0, 10000, (2, 128), dtype=torch.long)
logits = model(x)  # shape (2, 128, vocab_size)
```

Using `cs336_basics` with the canonical impl helpers
---------------------------------------------------
For tests and most scripts, prefer importing algorithmic helpers from the canonical
`cs336_basics.impl` package (this ensures a single source of truth and keeps adapters and
thin shims out of the code path). Example:

```python
from cs336_basics.impl import Tokenizer, train_bpe
from cs336_basics.transformer import TransformerLM

# build model and use the impl Tokenizer when needed
```

Development notes & recommended workflow
---------------------------------------
- For reproducible environments and locked dependencies, we recommend using `uv` with
  `uv.lock` present in the repo. See the top-level README for `uv install` / `uv run` examples.
- Keep model definitions in `cs336_basics/` and algorithmic implementations or test adapters in
  `cs336_basics/impl/`. This separation avoids duplication and reduces import cycles.
- Run unit tests frequently while developing:

```
uv run pytest -q
```

- Follow the project's code style (Black / Ruff if you use them). Small, focused diffs are
  easier to review than large formatting-only commits.

Where to look next
------------------
- `cs336_basics/impl/README.md` — detailed developer guide and public API for the canonical
  implementations used across tests and scripts.
- `scripts/` — example scripts for training, encoding datasets, and generation.
- `tests/` — unit tests exercising each required functionality.

---
Updated: 2026-02-02
