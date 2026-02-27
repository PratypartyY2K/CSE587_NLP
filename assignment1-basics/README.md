# Assignment 1 Basics

Repository for Assignment 1 (tokenizer + transformer fundamentals).

## Required setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Required tests

```bash
python -m pytest -q
```

## Required data rule

All input datasets must be inside this repository under `data/`.

Required paths used by scripts:
- `data/TinyStoriesV2-GPT4-train.txt`
- `data/TinyStoriesV2-GPT4-valid.txt`
- `data/owt_train.txt`
- `data/owt_valid.txt`

## Required training pipeline

```bash
# 1) train tokenizer
python scripts/train_bpe_run.py

# 2) encode train/valid text
python scripts/encode_parallel_bin.py --vocab tokenizer_vocab.pkl --merges tokenizer_merges.pkl --input data/TinyStoriesV2-GPT4-train.txt --out out/tiny_train_ids.npy --workers 8
python scripts/encode_parallel_bin.py --vocab tokenizer_vocab.pkl --merges tokenizer_merges.pkl --input data/TinyStoriesV2-GPT4-valid.txt --out out/tiny_valid_ids.npy --workers 8

# 3) train + eval model
python scripts/train.py \
  --train-data out/tiny_train_ids.npy \
  --valid-data out/tiny_valid_ids.npy \
  --checkpoint out/checkpoint.pt \
  --vocab-size 10000 \
  --context-length 128 \
  --d-model 512 \
  --num-layers 6 \
  --num-heads 8 \
  --batch-size 16 \
  --lr 3e-4 \
  --total-steps 5000 \
  --device cuda
```
