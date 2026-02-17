import time
import pickle
from cs336_basics.impl import train_bpe

input_path = "data/owt_train.txt"
vocab_size = 32000
special_tokens = ["<|endoftext|>"]

start = time.time()
vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
end = time.time()

with open("owt_small_tokenizer_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open("owt_small_tokenizer_merges.pkl", "wb") as f:
    pickle.dump(merges, f)

longest_token_id = max(vocab.keys(), key=lambda k: len(vocab[k]))
longest_token = vocab[longest_token_id]

print("elapsed_seconds=", end - start)
print("vocab_size=", len(vocab))
print("num_merges=", len(merges))
print("longest_token_id=", longest_token_id)
print("longest_token_bytes_len=", len(longest_token))
print("longest_token_utf8=", longest_token.decode("utf-8", errors="replace"))
