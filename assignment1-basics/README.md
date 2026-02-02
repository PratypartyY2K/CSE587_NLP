# Assignment 1 — Basics (tokenizer + transformer)

This repository contains the code and tests for the "assignment1-basics" project (byte-level BPE tokenizer, Transformer building blocks, training utilities, and small example scripts). The repository is organized so the canonical implementations live under `cs336_basics/impl/` and higher-level modules (models, scripts, examples) live at the top-level.

Quick start
- Install dependencies (recommended in a virtualenv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or pip install -U pytest torch numpy pypdf black ruff
```

- Run the unit tests (fast):

```bash
pytest -q
```

- Train code (example):

```bash
python scripts/train.py \
  --train-data out/tiny_train_ids.npy \
  --valid-data out/tiny_valid_ids.npy \
  --checkpoint out/checkpoint.pt \
  --vocab-size 10000 --context-length 512 --d-model 512 \
  --num-layers 6 --num-heads 8 --batch-size 32 --lr 3e-4 --device cpu
```

- Generate from a model (Python example):

```python
from cs336_basics.impl import run_generate_impl
from cs336_basics.transformer import TransformerLM

m = TransformerLM(vocab_size=100, context_length=16, d_model=32, num_layers=2, num_heads=4, d_ff=128, rope_theta=10000.0)
prompt = [1, 2, 3]
out = run_generate_impl(m, prompt, max_new_tokens=20, temperature=1.0, top_p=0.9, device='cpu')
print(out)
```

Project layout (high level)
- `cs336_basics/impl/` — canonical implementations and small adapters used by tests and scripts. Import from `cs336_basics.impl` in code and scripts.
- `cs336_basics/` — small model files and helpers (Transformer, layers, rope, etc.).
- `scripts/` — convenience scripts for training, encoding, and evaluation.
- `tests/` — unit tests that exercise the required functionality. Run with `pytest`.
- `data/` — small sample datasets (tracked in repo).
- `artifacts/`, `out/` — directories used for large tokenizer artifacts and encoded outputs (ignored by git).
