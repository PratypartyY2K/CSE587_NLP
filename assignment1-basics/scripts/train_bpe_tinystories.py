#!/usr/bin/env python3
"""
Train a byte-level BPE on TinyStories with a special <|endoftext|> token.
This script uses multiprocessing for pretokenization (splitting on the special
token) and then runs the same BPE merge loop as `cs336_basics/tokenizer.train_bpe`.

Outputs:
  - out_dir/vocab.json  (token-string -> id)
  - out_dir/merges.txt (one "tokenA tokenB" per line)

Usage example (macOS zsh):
  /usr/bin/time -l python3 scripts/train_bpe_tinystories.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --vocab_size 10000 \
    --special_token '<|endoftext|>' \
    --out_dir out/bpe_10k --workers 8

"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from typing import List, Dict, Tuple, Optional
import multiprocessing as mp

import regex as re
import cProfile
import pstats
import io

# try to import psutil for in-script RSS reporting; optional
try:
    import psutil
except Exception:
    psutil = None

# import GPT2 split pattern from repo tokenizer (keeps tokenization consistent)
from cs336_basics.tokenizer import GPT2_SPLIT_PATTERN, Tokenizer


def worker_tokenize(args):
    """Worker that tokenizes a text segment and returns a Counter mapping tuple(int)->count.

    The segment will NOT contain the special token; segments are created by splitting
    the original file on the special token so merges never cross document boundaries.
    """
    seg, _split_pat = args
    if not seg:
        return Counter()
    split_regex = _split_pat
    word_counts: Counter = Counter()
    for m in split_regex.finditer(seg):
        tok_bytes = m.group(0).encode("utf-8")
        token_ids = tuple(tok_bytes)  # each byte as int 0-255
        word_counts[token_ids] += 1
    return word_counts


def make_segments_generator(path: str, special_token: str):
    """Yield segments split by special_token. The special token is removed so
    no merges cross it. This reads the file in streaming fashion.
    """
    # We'll read the whole file into memory for simplicity (TinyStories is not huge)
    # If needed, this can be replaced with a streaming splitter.
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # split removes the delimiter; that ensures no token merges cross doc boundaries
    parts = text.split(special_token)
    for part in parts:
        yield part


def train_bpe_from_counters(word_counts: Counter, vocab_size: int, special_tokens: List[str]):
    # Initialize raw-byte vocab
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: List[Tuple[bytes, bytes]] = []

    num_merges = vocab_size - 256 - len(special_tokens)
    if num_merges < 0:
        num_merges = 0

    for i in range(num_merges):
        pair_counts: Counter = Counter()
        for word, freq in word_counts.items():
            for j in range(len(word) - 1):
                pair = (word[j], word[j + 1])
                pair_counts[pair] += freq

        if not pair_counts:
            print(f"Stopping early at merge {i}: no more pairs")
            break

        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]])))

        a_bytes = vocab[best_pair[0]]
        b_bytes = vocab[best_pair[1]]
        merges.append((a_bytes, b_bytes))

        new_id = 256 + i
        vocab[new_id] = a_bytes + b_bytes

        # Apply merge to all words
        new_word_counts: Counter = Counter()
        for word, freq in word_counts.items():
            new_word = []
            j = 0
            L = len(word)
            while j < L:
                if j < L - 1 and word[j] == best_pair[0] and word[j + 1] == best_pair[1]:
                    new_word.append(new_id)
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            new_word_counts[tuple(new_word)] += freq

        word_counts = new_word_counts
    # Append special tokens after highest id
    current_id = max(vocab.keys()) + 1
    for st in special_tokens:
        vocab[current_id] = st.encode("utf-8")
        current_id += 1

    return vocab, merges


def serialize_outputs(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Save vocab.json as token-string -> id (decoded for human inspection)
    vocab_json = {v.decode("utf-8", errors="replace"): k for k, v in vocab.items()}
    vocab_path = os.path.join(out_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    merges_path = os.path.join(out_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.decode('utf-8', errors='replace')} {b.decode('utf-8', errors='replace')}\n")

    return vocab_path, merges_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--special_token", type=str, default="<|endoftext|>")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    parser.add_argument("--sample_text_chars", type=int, default=1024)
    parser.add_argument("--profile", action="store_true", help="Run cProfile and dump stats to out_dir/profile.prof")
    args = parser.parse_args()

    def run_training():
        start_time = time.perf_counter()

        # Prepare regex
        split_regex = re.compile(GPT2_SPLIT_PATTERN)

        # Build segments generator
        segments = list(make_segments_generator(args.input, args.special_token))
        print(f"Split input into {len(segments)} segments (delimited by special token)")

        # Tokenize segments in parallel; worker returns Counter per segment
        pool_args = ((seg, split_regex) for seg in segments)
        aggregated: Counter = Counter()
        pretoken_start = time.perf_counter()
        with mp.Pool(processes=args.workers) as pool:
            for i, cnt in enumerate(pool.imap_unordered(worker_tokenize, pool_args, chunksize=16)):
                aggregated.update(cnt)
                # log less frequently to avoid extremely large console output
                if (i + 1) % 10000 == 0:
                    print(f"Processed {i+1} segments; current unique token-words: {len(aggregated)}")
        pretoken_end = time.perf_counter()

        print(f"Completed token counting: unique words={len(aggregated)}, total types (Counter sum)={sum(aggregated.values())}")

        # Train BPE (merge loop)
        merge_start = time.perf_counter()
        vocab, merges = train_bpe_from_counters(aggregated, args.vocab_size, [args.special_token])
        merge_end = time.perf_counter()

        vocab_path, merges_path = serialize_outputs(vocab, merges, args.out_dir)

        elapsed = time.perf_counter() - start_time

        print("--- Summary ---")
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print(f"Pretokenization time: {pretoken_end - pretoken_start:.2f} seconds")
        print(f"Merge training time: {merge_end - merge_start:.2f} seconds")
        if psutil:
            p = psutil.Process()
            rss_mb = p.memory_info().rss / 1024 ** 2
            print(f"Current RSS: {rss_mb:.2f} MB (psutil)")

        print(f"Vocab size (items): {len(vocab)}")
        # find special token id
        special_bytes = args.special_token.encode("utf-8")
        special_id = None
        for k, v in vocab.items():
            if v == special_bytes:
                special_id = k
                break
        print(f"Special token '{args.special_token}' id: {special_id}")

        # Longest token
        max_len = 0
        max_tok = b""
        for v in vocab.values():
            if len(v) > max_len:
                max_len = len(v)
                max_tok = v
        print(f"Longest token length (bytes): {max_len}")
        print(f"Longest token (decoded with replacement): '{max_tok.decode('utf-8', errors='replace')}'")

        print(f"Saved vocab to: {vocab_path}")
        print(f"Saved merges to: {merges_path}")

        # quick sanity check: instantiate Tokenizer and round-trip a sample
        try:
            tkn = Tokenizer(vocab, merges, [args.special_token])
            with open(args.input, "r", encoding="utf-8") as f:
                sample = f.read(args.sample_text_chars)
            ids = tkn.encode(sample)
            decoded = tkn.decode(ids)
            print(f"Sample encode -> {len(ids)} ids; decoded sample length {len(decoded)} chars")
        except Exception as e:
            print(f"Tokenizer sanity check failed: {e}")

        # return timing breakdown
        return {
            'elapsed': elapsed,
            'pretoken': pretoken_end - pretoken_start,
            'merge': merge_end - merge_start,
            'vocab_size': len(vocab),
            'longest_token': max_tok.decode('utf-8', errors='replace')
        }

    # run / profile
    if args.profile:
        os.makedirs(args.out_dir, exist_ok=True)
        prof_path = os.path.join(args.out_dir, 'profile.prof')
        pr = cProfile.Profile()
        pr.enable()
        results = run_training()
        pr.disable()
        pr.dump_stats(prof_path)
        # write a small human-readable top functions by cumulative time
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        with open(os.path.join(args.out_dir, 'profile_top.txt'), 'w', encoding='utf-8') as f:
            f.write(s.getvalue())
        print(f"Profile dumped to: {prof_path}")
        print(f"Top functions written to: {os.path.join(args.out_dir, 'profile_top.txt')}")
    else:
        run_training()


if __name__ == "__main__":
    main()
