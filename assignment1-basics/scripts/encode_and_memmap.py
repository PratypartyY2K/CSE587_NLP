#!/usr/bin/env python3
# Ensure numeric libraries and tokenizers don't spawn threads/processes when run
import os

import time
import pickle
import numpy as np
from cs336_basics.impl import Tokenizer
from multiprocessing import Pool, cpu_count
from collections.abc import Iterator


def load_tokenizer(vocab_pkl, merges_pkl, special_tokens=None):
    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_pkl, "rb") as f:
        merges = pickle.load(f)
    return Tokenizer(vocab, merges, special_tokens=special_tokens)


def count_tokens(tokenizer, filepath):
    # Deprecated single-threaded path (kept for reference)
    cnt = 0
    with open(filepath, encoding="utf-8") as f:
        for _ in tokenizer.encode_iterable(f):
            cnt += 1
    return cnt


def doc_stream(filepath: str, delimiter: str = "<|endoftext|>") -> Iterator[str]:
    """Yield documents (including the delimiter) from a large file without loading it all."""
    buf = ""
    with open(filepath, encoding="utf-8") as f:
        while True:
            chunk = f.read(64 * 1024)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(delimiter)
            buf = parts.pop()
            for part in parts:
                yield part + delimiter
    if buf:
        yield buf


# Worker-global tokenizer (initialized in each Pool worker)
_WORKER_TOKENIZER = None


def _init_worker(vocab_pkl: str, merges_pkl: str, special_tokens=None):
    global _WORKER_TOKENIZER
    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_pkl, "rb") as f:
        merges = pickle.load(f)
    _WORKER_TOKENIZER = Tokenizer(vocab, merges, special_tokens)


def _worker_count(doc: str) -> int:
    global _WORKER_TOKENIZER
    ids = _WORKER_TOKENIZER.encode(doc)
    return len(ids)


def _worker_encode(doc: str) -> list[int]:
    global _WORKER_TOKENIZER
    return _WORKER_TOKENIZER.encode(doc)


def write_tokens_memmap(
    tokenizer,
    filepath,
    out_path,
    dtype=np.uint16,
    workers: int = 1,
    vocab_pkl=None,
    merges_pkl=None,
    special_tokens=None,
):
    print(f"Counting tokens for {filepath} (workers={workers})...")
    start = time.time()
    if workers and workers > 1 and vocab_pkl and merges_pkl:
        # parallel counting by documents
        with Pool(workers, initializer=_init_worker, initargs=(vocab_pkl, merges_pkl, special_tokens)) as pool:
            counts = pool.imap(_worker_count, doc_stream(filepath))
            total = 0
            for c in counts:
                total += c
    else:
        total = count_tokens(tokenizer, filepath)

    dur = time.time() - start
    print(f"Token count={total} (counting time={dur:.2f}s)")

    if total == 0:
        print("No tokens found; writing empty file.")
        np.save(out_path, np.array([], dtype=dtype))
        return total

    # prepare memmap
    print(f"Creating memmap {out_path} with {total} entries of type {dtype}...")
    mm = np.memmap(out_path, dtype=dtype, mode="w+", shape=(total,))

    print(f"Writing tokens to memmap (workers={workers})...")
    i = 0
    start = time.time()
    if workers and workers > 1 and vocab_pkl and merges_pkl:
        with Pool(workers, initializer=_init_worker, initargs=(vocab_pkl, merges_pkl, special_tokens)) as pool:
            for ids in pool.imap(_worker_encode, doc_stream(filepath)):
                for tid in ids:
                    if tid >= 2**16:
                        raise ValueError(f"Token id {tid} >= 65536; uint16 overflow. Use uint32.")
                    mm[i] = np.uint16(tid)
                    i += 1
    else:
        with open(filepath, encoding="utf-8") as f:
            for tid in tokenizer.encode_iterable(f):
                if tid >= 2**16:
                    raise ValueError(f"Token id {tid} >= 65536; uint16 overflow. Use uint32.")
                mm[i] = np.uint16(tid)
                i += 1
    mm.flush()
    dur = time.time() - start
    print(f"Wrote {i} tokens to {out_path} (time={dur:.2f}s, {i / dur if dur > 0 else float('inf'):.0f} tokens/s)")
    return total


def main():
    os.makedirs("out", exist_ok=True)

    # use (CPU count - 1) workers by default, at least 1
    workers = max(1, cpu_count() - 1)

    jobs = [
        # (vocab, merges, special_tokens, input, output)
        (
            "tokenizer_vocab.pkl",
            "tokenizer_merges.pkl",
            ["<|endoftext|>"],
            "data/TinyStoriesV2-GPT4-train.txt",
            "out/tiny_train_ids.npy",
        ),
        (
            "tokenizer_vocab.pkl",
            "tokenizer_merges.pkl",
            ["<|endoftext|>"],
            "data/TinyStoriesV2-GPT4-valid.txt",
            "out/tiny_valid_ids.npy",
        ),
        (
            "owt_small_tokenizer_vocab.pkl",
            "owt_small_tokenizer_merges.pkl",
            ["<|endoftext|>"],
            "data/owt_train_100mb.txt",
            "out/owt_train_100mb_ids.npy",
        ),
        (
            "owt_small_tokenizer_vocab.pkl",
            "owt_small_tokenizer_merges.pkl",
            ["<|endoftext|>"],
            "data/owt_valid.txt",
            "out/owt_valid_ids.npy",
        ),
    ]

    for vocab_pkl, merges_pkl, special_tokens, input_path, out_path in jobs:
        print("\n=== JOB: ", input_path, "->", out_path, "===")
        if not os.path.exists(vocab_pkl) or not os.path.exists(merges_pkl):
            print(f"Missing tokenizer files {vocab_pkl} or {merges_pkl}; skipping.")
            continue
        if not os.path.exists(input_path):
            print(f"Input file {input_path} not found; skipping.")
            continue

        tokenizer = load_tokenizer(vocab_pkl, merges_pkl, special_tokens)
        try:
            total = write_tokens_memmap(
                tokenizer,
                input_path,
                out_path,
                dtype=np.uint16,
                workers=workers,
                vocab_pkl=vocab_pkl,
                merges_pkl=merges_pkl,
                special_tokens=special_tokens,
            )
            print(f"Finished encoding {input_path}: total tokens={total}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    main()
