# Assignment 1 Basics

Minimal implementation of tokenizer + transformer components for the assignment.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run tests

```bash
python -m pytest -q
```

## Data rule

Keep all input datasets inside this repository under `data/`.

## Common commands

```bash
# train tokenizer
python scripts/train_bpe_run.py

# encode text to token IDs
python scripts/encode_parallel_bin.py --vocab tokenizer_vocab.pkl --merges tokenizer_merges.pkl --input data/TinyStoriesV2-GPT4-train.txt --out out/tiny_train_ids.npy --workers 8

# train model
python scripts/train.py --train-data out/tiny_train_ids.npy --valid-data out/tiny_valid_ids.npy --checkpoint out/checkpoint.pt --vocab-size 10000 --context-length 128 --device cpu
```
