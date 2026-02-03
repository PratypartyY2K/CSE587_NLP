# cs336_basics.impl

This package contains the canonical, tested implementations used by the assignment repository. The
goal is to keep all substantive algorithmic code (tokenizer, attention, layers, optimizers, I/O
helpers, and small utilities) in a single place so tests and scripts can import a stable API:
`cs336_basics.impl`.

This README documents the public surface, examples of how to use the helpers, recommended
development workflow, and a short reference for contributors.

Checklist (what this module provides)
- Byte-pair encoding (BPE) training and a byte-level Tokenizer implementation
- Core NN building blocks used in the assignment (Linear, Embedding, SwiGLU, RMSNorm)
- Rotary positional embeddings (RoPE) and scaled dot-product attention helpers
- Causal multi-head self-attention and a pre-norm Transformer block + TransformerLM runner
- Utility functions: softmax, cross-entropy, gradient clipping, batching helper (get_batch)
- Minimal AdamW optimizer and a cosine LR schedule helper
- I/O helpers for saving / loading checkpoints
- A small generation helper: sampling with temperature and top-p (nucleus) support

Public API (exported names)
These names are re-exported via `cs336_basics.impl` (see `cs336_basics/impl/__init__.py`). Import
from the impl package in tests and scripts to keep dependencies stable.

- Tokenizer and BPE
  - `train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> (vocab, merges)`
  - `Tokenizer` class: `encode`, `encode_iterable`, `decode`, `from_files`.

- Attention & Transformer helpers
  - `scaled_dot_product_attention`
  - `multihead_self_attention`
  - `rope` / `multihead_self_attention_with_rope`
  - `transformer_block`, `transformer_lm`

- NN utilities and small layer adapters
  - `softmax`, `silu`
  - `get_batch` — sample LM batches from a 1D token array (supports np.memmap)
  - `cross_entropy` — numerically stable cross-entropy
  - `gradient_clipping`
  - `linear`, `embedding`, `swiglu`, `rmsnorm`

- Optimizer & schedule
  - `AdamW` (a torch.optim.Optimizer-compatible minimal AdamW implementation)
  - `get_lr_cosine_schedule` (LR warmup + cosine anneal helper)

- I/O and generation
  - `save_checkpoint`, `load_checkpoint`
  - `generate` — autoregressive sampling with temperature and top-p

Usage examples
---------------
Basic import pattern (recommended from tests and scripts):

```python
from cs336_basics.impl import Tokenizer, train_bpe, get_batch
from cs336_basics.impl import AdamW, get_lr_cosine_schedule
from cs336_basics.impl import save_checkpoint, load_checkpoint, generate
```

Train a small BPE tokenizer and save artifacts:

```python
vocab, merges = train_bpe('data/tinystories_sample_small.txt', vocab_size=10000, special_tokens=['<|endoftext|>'])
# serialize
import pickle
with open('tokenizer_vocab.pkl','wb') as f:
    pickle.dump(vocab, f)
with open('tokenizer_merges.pkl','wb') as f:
    pickle.dump(merges, f)
```

Sampling batches for training (memory-efficient):

```python
import numpy as np
from cs336_basics.impl import get_batch
arr = np.load('out/tiny_train_ids.npy', mmap_mode='r')  # memmap
x, y = get_batch(arr, batch_size=32, context_length=128, device='cpu')
```

Saving and loading checkpoints (recommended pattern):

```python
# Save
save_checkpoint(model, optimizer, iteration, 'out/checkpoint.pt')
# Load
iteration = load_checkpoint('out/checkpoint.pt', model, optimizer)
```

Text generation example (sampling):

```python
from cs336_basics.impl import generate
# model: TransformerLM instance (loaded / constructed)
prompt = [1,2,3]
out_ids = generate(model, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9, device='cpu', eos_token_id=50256)
```

Developer workflow & tests
-------------------------
- Preferred reproducible environment: use the project `uv.lock` + `uv` tooling (see root README).
- Run the unit test suite frequently while making changes:

```bash
uv run pytest -q
# or, without uv (if using an ordinary venv):
pytest -q
```

- Keep the public function signatures stable — tests identify their entry points through the
  adapters/exports in `cs336_basics/impl/__init__.py` and `tests/adapters.py`.


- If you don't want to install Black/Ruff, the tests still run without them. Please avoid large
  style-only diffs in patches – keep changes focused on behavior or small, well-formatted edits.

Troubleshooting
---------------
- If you encounter import cycles, prefer moving helper functions into the `impl/` package to keep
  a single import root; the `impl` package is intentionally the canonical location.
- For memory issues when tokenizing large corpora, use `numpy.memmap` and streaming encoders in
  `scripts/encode_parallel_bin.py` / `scripts/encode_and_memmap.py`.

---
Updated on: 2026-02-02
