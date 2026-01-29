#!/usr/bin/env python3
import argparse
import os
import time
import pickle
import numpy as np
from cs336_basics.tokenizer import Tokenizer


def load_tokenizer(vocab_pkl, merges_pkl, special_tokens=None):
    with open(vocab_pkl, 'rb') as f:
        vocab = pickle.load(f)
    with open(merges_pkl, 'rb') as f:
        merges = pickle.load(f)
    return Tokenizer(vocab, merges, special_tokens=special_tokens)


def count_tokens(tokenizer, filepath):
    cnt = 0
    with open(filepath, encoding='utf-8') as f:
        for _ in tokenizer.encode_iterable(f):
            cnt += 1
    return cnt


def write_tokens_memmap(tokenizer, filepath, out_path, dtype=np.uint16):
    print(f"Counting tokens for {filepath}...")
    start = time.time()
    total = count_tokens(tokenizer, filepath)
    dur = time.time() - start
    print(f"Token count={total} (counting time={dur:.2f}s)")

    if total == 0:
        print("No tokens found; writing empty file.")
        np.save(out_path, np.array([], dtype=dtype))
        return total

    # prepare memmap
    print(f"Creating memmap {out_path} with {total} entries of type {dtype}...")
    mm = np.memmap(out_path, dtype=dtype, mode='w+', shape=(total,))

    print(f"Writing tokens to memmap...")
    i = 0
    start = time.time()
    with open(filepath, encoding='utf-8') as f:
        for tid in tokenizer.encode_iterable(f):
            if tid >= 2**16:
                raise ValueError(f"Token id {tid} >= 65536; uint16 overflow. Use uint32.")
            mm[i] = np.uint16(tid)
            i += 1
    mm.flush()
    dur = time.time() - start
    print(f"Wrote {i} tokens to {out_path} (time={dur:.2f}s, {i/dur if dur>0 else float('inf'):.0f} tokens/s)")
    return total


def main():
    os.makedirs('out', exist_ok=True)

    jobs = [
        # (vocab, merges, special_tokens, input, output)
        ('tokenizer_vocab.pkl', 'tokenizer_merges.pkl', ['<|endoftext|>'], 'data/TinyStoriesV2-GPT4-train.txt', 'out/tiny_train_ids.npy'),
        ('tokenizer_vocab.pkl', 'tokenizer_merges.pkl', ['<|endoftext|>'], 'data/TinyStoriesV2-GPT4-valid.txt', 'out/tiny_valid_ids.npy'),
        ('owt_small_tokenizer_vocab.pkl', 'owt_small_tokenizer_merges.pkl', ['<|endoftext|>'], 'data/owt_train_100mb.txt', 'out/owt_train_100mb_ids.npy'),
        ('owt_small_tokenizer_vocab.pkl', 'owt_small_tokenizer_merges.pkl', ['<|endoftext|>'], 'data/owt_valid.txt', 'out/owt_valid_ids.npy'),
    ]

    for vocab_pkl, merges_pkl, special_tokens, input_path, out_path in jobs:
        print('\n=== JOB: ', input_path, '->', out_path, '===')
        if not os.path.exists(vocab_pkl) or not os.path.exists(merges_pkl):
            print(f"Missing tokenizer files {vocab_pkl} or {merges_pkl}; skipping.")
            continue
        if not os.path.exists(input_path):
            print(f"Input file {input_path} not found; skipping.")
            continue

        tokenizer = load_tokenizer(vocab_pkl, merges_pkl, special_tokens)
        try:
            total = write_tokens_memmap(tokenizer, input_path, out_path, dtype=np.uint16)
            print(f"Finished encoding {input_path}: total tokens={total}")
        except Exception as e:
            print(f"Error processing {input_path}: {e}")


if __name__ == '__main__':
    main()
