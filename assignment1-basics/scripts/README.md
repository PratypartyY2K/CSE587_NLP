# scripts

Core scripts required for the assignment pipeline.

## Required scripts

- `train_bpe_run.py`: train TinyStories BPE tokenizer
- `encode_parallel_bin.py`: encode text files into token-id `.npy` arrays
- `train.py`: train and evaluate Transformer LM

## Required usage

```bash
python scripts/train_bpe_run.py
python scripts/encode_parallel_bin.py --help
python scripts/train.py --help
```

## Required data rule

Script inputs must come from repo-local `data/` files.
