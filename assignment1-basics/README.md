# Assignment 1 — Basics (tokenizer + transformer)

This repository provides a compact, self-contained implementation of the components required
for the assignment: a byte-level BPE tokenizer, Transformer building blocks, training/IO
utilities, and small example scripts. The code is organized so that algorithmic implementations
live under `cs336_basics/impl/` and model primitives live under `cs336_basics/`.

This README covers getting started (recommended: using `uv`), running tests and scripts,
expected workflows for development, and where to find further documentation inside the repo.

Table of contents
- Quick start (recommended: `uv`)
- Running tests
- Typical workflows and examples
  - Train a BPE tokenizer
  - Encode large corpora to `.npy` using memmap
  - Train a small TransformerLM
  - Generate text from a model
- Developer notes (formatting, linting, tests)
- Data, artifacts, and gitignore
- Where to find documentation (module READMEs)

Quick start (recommended: using `uv`)
------------------------------------
We recommend using the `uv` tool which uses `pyproject.toml` and the included `uv.lock` to
create a reproducible environment. `uv` is lightweight and convenient for ensuring everyone
(and CI) uses the same dependency versions.

1) Install `uv` (one-time on your machine):

```bash
python -m pip install --upgrade uv
```

2) Create the environment and install pinned dependencies from the lockfile:

```bash
# this creates a managed environment and installs packages according to uv.lock
uv install
```

3) Run tests and scripts inside the managed environment:

```bash
# run tests
uv run pytest -q

# run a script (example)
uv run python scripts/train.py --help
```

If you choose not to use `uv`, create a virtual environment manually and install the needed
packages (see the `pyproject.toml` for hints). The repository is tested against the locked
versions and `uv` is the easiest way to reproduce that environment.

Running tests
-------------
- Run the full test suite (fast locally):

```bash
uv run pytest -q
```

- If you prefer a plain venv (no `uv`):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install pytest numpy torch regex pypdf
pytest -q
```

Typical workflows and examples
------------------------------
Below are short examples for the main tasks used in the assignment. The scripts directory
contains runnable scripts that implement these workflows with more options.

Train a BPE tokenizer (tiny example)

```python
from cs336_basics.impl import train_bpe
vocab, merges = train_bpe('data/tinystories_sample_small.txt', vocab_size=10000, special_tokens=['<|endoftext|>'])
# serialize as desired (pickle/json)
```

Encode a large corpus to a `.npy` memmap (memory-efficient)

- Use `scripts/encode_parallel_bin.py` for document-level parallelism (large corpora).
- Use `scripts/encode_and_memmap.py` for single-process or debugging.

Example (parallel encoding):

```bash
uv run python scripts/encode_parallel_bin.py \
  --vocab tokenizer_vocab.pkl --merges tokenizer_merges.pkl \
  --input data/owt_train.txt --out out/owt_train_ids.npy --workers 8
```

Train a TransformerLM (toy run)

```bash
uv run python scripts/train.py \
  --train-data out/tiny_train_ids.npy --valid-data out/tiny_valid_ids.npy \
  --checkpoint out/checkpoint.pt --vocab-size 10000 --context-length 512 \
  --d-model 512 --num-layers 6 --num-heads 8 --batch-size 32 --lr 3e-4 --device cpu
```

Generate text (sampling)

```python
from cs336_basics.impl import run_generate_impl
# model is a TransformerLM instance already loaded or instantiated
prompt = [1, 2, 3]
out_ids = run_generate_impl(model, prompt, max_new_tokens=128, temperature=0.8, top_p=0.9, device='cpu', eos_token_id=50256)
```

Developer notes
---------------
- Tests are the primary correctness gate — run them often.
- Formatting & linting: I recommend using Black + Ruff. If you use them locally, run:

```bash
uv run black . --line-length 120
uv run ruff format .
```

- If you prefer not to install these tools, they are optional for tests but helpful for consistent
  style.

- To add a public helper: put algorithmic code in `cs336_basics/impl/` and export it via
  `cs336_basics/impl/__init__.py`. Tests import from `cs336_basics.impl`.

Data and artifacts
------------------
- Small sample data live in `data/` for tests and demonstrations. Large corpora should be kept
  outside the repo and referenced (use `artifacts/` or `out/` for generated, large files).
- `.gitignore` is configured to ignore generated artifacts and common binary outputs. Do not
  commit multi-GB files to the repository.

Where to find more documentation inside the repo
-----------------------------------------------
- `cs336_basics/impl/README.md` — primary developer guide and API reference for the canonical
  implementations used by tests and scripts.
- `cs336_basics/README.md` — overview of the `cs336_basics` module (model primitives and
  transformer classes).
- `scripts/README.md` — details for each script, common flags, and invocation examples.

---
Updated: 2026-02-02
