import os
import sys
import argparse
import pickle
import time
from multiprocessing import Pool, cpu_count
from collections.abc import Iterator

import numpy as np
from cs336_basics.impl import Tokenizer


_WORKER_TOKENIZER = None


def _init_worker(vocab_pkl: str, merges_pkl: str, special_tokens=None):
    global _WORKER_TOKENIZER
    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_pkl, "rb") as f:
        merges = pickle.load(f)
    _WORKER_TOKENIZER = Tokenizer(vocab, merges, special_tokens)


def _worker_encode(doc: str) -> bytes:
    global _WORKER_TOKENIZER
    ids = _WORKER_TOKENIZER.encode(doc)
    arr = np.asarray(ids, dtype=np.uint16)
    return arr.tobytes()


def doc_stream(filepath: str, delimiter: str = "<|endoftext|>") -> Iterator[str]:
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


def encode_to_binary(vocab_pkl: str, merges_pkl: str, special_tokens, input_path: str, out_bin_path: str, workers: int):
    os.makedirs(os.path.dirname(out_bin_path) or ".", exist_ok=True)
    tmp_bin = out_bin_path
    if os.path.exists(tmp_bin):
        os.remove(tmp_bin)

    start = time.time()
    processed_docs = 0
    total_bytes = 0
    total_tokens = 0
    with Pool(workers, initializer=_init_worker, initargs=(vocab_pkl, merges_pkl, special_tokens)) as pool:
        for buf in pool.imap(_worker_encode, doc_stream(input_path), chunksize=10):
            if buf:
                with open(tmp_bin, "ab") as out:
                    out.write(buf)
                total_bytes += len(buf)
                total_tokens += len(buf) // 2
            processed_docs += 1
            if processed_docs % 100 == 0:
                elapsed = time.time() - start
                rate = total_bytes / elapsed if elapsed > 0 else 0
                print(
                    f"Processed {processed_docs} docs, tokens={total_tokens}, bytes={total_bytes}, bytes/s={rate:.1f}"
                )
    elapsed = time.time() - start
    print(
        "Finished encoding to binary. "
        f"docs={processed_docs} tokens={total_tokens} bytes={total_bytes} "
        f"time={elapsed:.2f}s"
    )
    return tmp_bin, total_tokens


def bin_to_npy(bin_path: str, npy_path: str, dtype=np.uint16, chunksize=10_000_000):
    size = os.path.getsize(bin_path)
    if size % np.dtype(dtype).itemsize != 0:
        raise ValueError("Binary file size not multiple of dtype size")
    nelems = size // np.dtype(dtype).itemsize
    mmap = np.lib.format.open_memmap(npy_path, mode="w+", dtype=dtype, shape=(nelems,))
    with open(bin_path, "rb") as f:
        offset = 0
        while True:
            chunk = f.read(chunksize * np.dtype(dtype).itemsize)
            if not chunk:
                break
            arr = np.frombuffer(chunk, dtype=dtype)
            mmap[offset : offset + arr.size] = arr
            offset += arr.size
    mmap.flush()
    return nelems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", default="tokenizer_vocab.pkl")
    parser.add_argument("--merges", default="tokenizer_merges.pkl")
    parser.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--input")
    parser.add_argument("--out", required=True, help="output .npy path (will produce .bin temporary)")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = parser.parse_args()

    if not args.input:
        print("Please provide --input path")
        sys.exit(1)
    bin_path = args.out + ".bin"
    npy_path = args.out

    print("Encoding:", args.input)
    print("Workers:", args.workers)
    bin_file, tokens = encode_to_binary(args.vocab, args.merges, args.special, args.input, bin_path, args.workers)
    print("Converting binary to .npy memmap...")
    nelems = bin_to_npy(bin_file, npy_path)
    print(f"Wrote {nelems} tokens to {npy_path}")


if __name__ == "__main__":
    main()
