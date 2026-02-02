# scripts/

This directory contains convenience scripts used to train tokenizers, encode datasets, and run small experiments with the Transformer model in this repository. The scripts are intended as examples and as a practical starting point for experimentation; they are not production-grade runners.

Principles
- Keep scripts small and focused. Each script performs one primary task (train a tokenizer, encode a dataset, run a small training loop, etc.).
- Prefer using the `uv` managed environment (see top-level README) to ensure reproducible dependencies when running scripts.
- Where possible scripts accept arguments via `argparse`; run `<script> --help` to see available flags.

Common invocation patterns
```bash
# inside a uv-managed environment
uv run python scripts/<script>.py --help
# or, if you prefer a plain venv
python scripts/<script>.py --help
```

Scripts index
- `train.py` — A configurable training loop for the Transformer LM.
  - Key flags: `--train-data`, `--valid-data`, `--checkpoint`, `--vocab-size`, `--context-length`, `--d-model`, `--num-layers`, `--num-heads`, `--batch-size`, `--lr`, `--device`, `--total-steps`.
  - Example: run a toy training session on CPU
    ```bash
    uv run python scripts/train.py \
      --train-data out/tiny_train_ids.npy \
      --valid-data out/tiny_valid_ids.npy \
      --checkpoint out/checkpoint.pt \
      --vocab-size 10000 --context-length 128 --d-model 512 --num-layers 6 --num-heads 8 \
      --batch-size 32 --lr 3e-4 --device cpu --total-steps 1000
    ```

- `encode_and_memmap.py` — Single-process encoder: tokenize a text file, write token IDs to a `.npy` memmap.
  - Useful for small datasets or debugging the tokenization pipeline.
  - Example usage: `python scripts/encode_and_memmap.py --vocab vocab.pkl --merges merges.txt --input data/owt_train.txt --out out/owt_train.npy`

- `encode_parallel_bin.py` — Parallel (process-pool) encoder that writes an intermediate binary stream of uint16 token ids and converts it into a `.npy` memmap.
  - Uses multiprocessing worker initialization to avoid reloading big tokenizer objects for every document.
  - Good for encoding large text corpora by document-level parallelism.
  - Example: `python scripts/encode_parallel_bin.py --vocab tokenizer_vocab.pkl --merges tokenizer_merges.pkl --input data/owt_train.txt --out out/owt_train_ids.npy --workers 8`

- `train_bpe_run.py`, `train_owt_small.py` — Example scripts to train BPE tokenizers (TinyStories / OpenWebText) and to profile tokenizer training. See `--help` for tuning options.

- `tokenizer_throughput.py` — Small utility to measure tokenizer encode throughput.

- `compare_owt_tiny.py` — Quick script that loads two tokenizers (TinyStories vs. OpenWebText) and prints a small comparison on sample docs.

- `compute_compression.py` — Helper that computes bytes/token compression statistics for tokenizers; useful for the experiments section of the assignment. It reads serialized tokenizers and computes bytes-per-token for sampled documents.

- `train_bpe_run.py` — Wrapper used in the assignment to run `train_bpe` in a reproducible way (configurable hyperparameters, output paths, and logging).

---
Updated: 2026-02-02
