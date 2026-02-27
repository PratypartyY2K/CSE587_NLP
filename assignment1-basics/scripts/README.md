# scripts

Utility scripts for tokenizer training, dataset encoding, and model training.

## Usage

```bash
python scripts/<script>.py --help
```

## Data rule

Use only repo-local dataset files under `data/`.

## Main scripts

- `train_bpe_run.py`: train TinyStories tokenizer
- `encode_parallel_bin.py`: encode text to `.npy` token IDs
- `train.py`: train/evaluate language model
