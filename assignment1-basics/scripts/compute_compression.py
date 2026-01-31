import pickle
import random
from cs336_basics.impl import Tokenizer

random.seed(0)

# load tokenizers
with open("tokenizer_vocab.pkl", "rb") as f:
    tiny_vocab = pickle.load(f)
with open("tokenizer_merges.pkl", "rb") as f:
    tiny_merges = pickle.load(f)

tiny_tokenizer = Tokenizer(tiny_vocab, tiny_merges, special_tokens=["<|endoftext|>"])

with open("owt_small_tokenizer_vocab.pkl", "rb") as f:
    owt_vocab = pickle.load(f)
with open("owt_small_tokenizer_merges.pkl", "rb") as f:
    owt_merges = pickle.load(f)

owt_tokenizer = Tokenizer(owt_vocab, owt_merges, special_tokens=["<|endoftext|>"])


# helper to sample 10 documents from a file using <|endoftext|> as delimiter
def sample_docs(path, n=10):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    docs = text.split("<|endoftext|>")
    docs = [d for d in docs if d.strip()]
    if len(docs) <= n:
        return docs
    return random.sample(docs, n)


# TinyStories sample
tiny_docs = sample_docs("data/TinyStoriesV2-GPT4-train.txt", 10)
tiny_bytes = 0
tiny_tokens = 0
for d in tiny_docs:
    b = (d + "<|endoftext|>").encode("utf-8")
    ids = tiny_tokenizer.encode(d + "<|endoftext|>")
    tiny_bytes += len(b)
    tiny_tokens += len(ids)

tiny_ratio = tiny_bytes / tiny_tokens if tiny_tokens else float("inf")

# OpenWebText sample (use the 1MB sample to save time)
owt_docs = sample_docs("data/owt_train_sample_1mb.txt", 10)
owt_bytes = 0
owt_tokens = 0
for d in owt_docs:
    b = (d + "<|endoftext|>").encode("utf-8")
    ids = owt_tokenizer.encode(d + "<|endoftext|>")
    owt_bytes += len(b)
    owt_tokens += len(ids)

owt_ratio = owt_bytes / owt_tokens if owt_tokens else float("inf")

print("tiny_bytes=", tiny_bytes, "tiny_tokens=", tiny_tokens)
print("tiny_bytes_per_token=", tiny_ratio)
print("owt_bytes=", owt_bytes, "owt_tokens=", owt_tokens)
print("owt_bytes_per_token=", owt_ratio)
