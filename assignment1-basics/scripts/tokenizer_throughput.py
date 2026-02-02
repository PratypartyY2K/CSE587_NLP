import time
import pickle
from cs336_basics.impl import Tokenizer

with open("owt_small_tokenizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("owt_small_tokenizer_merges.pkl", "rb") as f:
    merges = pickle.load(f)

tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])

path = "data/owt_train_sample_1mb.txt"
start = time.time()
bytes_processed = 0
ids_count = 0
with open(path, encoding="utf-8") as f:
    for tid in tokenizer.encode_iterable(f):
        ids_count += 1
with open(path, "rb") as fb:
    bytes_processed = len(fb.read())
end = time.time()
elapsed = end - start
print("elapsed_seconds=", elapsed)
print("bytes_processed=", bytes_processed)
print("tokens_produced=", ids_count)
print("bytes_per_second=", bytes_processed / elapsed if elapsed > 0 else float("inf"))
print("tokens_per_second=", ids_count / elapsed if elapsed > 0 else float("inf"))
