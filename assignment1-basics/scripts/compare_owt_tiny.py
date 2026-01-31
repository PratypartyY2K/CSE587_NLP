import pickle, random
from cs336_basics.impl import Tokenizer

random.seed(0)

with open('tokenizer_vocab.pkl','rb') as f:
    tiny_vocab = pickle.load(f)
with open('tokenizer_merges.pkl','rb') as f:
    tiny_merges = pickle.load(f)

tiny = Tokenizer(tiny_vocab, tiny_merges, special_tokens=['<|endoftext|>'])

with open('owt_small_tokenizer_vocab.pkl','rb') as f:
    owt_vocab = pickle.load(f)
with open('owt_small_tokenizer_merges.pkl','rb') as f:
    owt_merges = pickle.load(f)

owt = Tokenizer(owt_vocab, owt_merges, special_tokens=['<|endoftext|>'])

with open('data/owt_train_sample_1mb.txt', encoding='utf-8') as f:
    text = f.read()

docs = [d for d in text.split('<|endoftext|>') if d.strip()]
if len(docs) < 10:
    sample = docs
else:
    sample = random.sample(docs, 10)

bytes_total = 0
tokens_by_owt = 0
tokens_by_tiny = 0

for d in sample:
    s = d + '<|endoftext|>'
    b = s.encode('utf-8')
    bytes_total += len(b)
    ids_owt = owt.encode(s)
    ids_tiny = tiny.encode(s)
    tokens_by_owt += len(ids_owt)
    tokens_by_tiny += len(ids_tiny)

ratio_owt = bytes_total / tokens_by_owt if tokens_by_owt else float('inf')
ratio_tiny = bytes_total / tokens_by_tiny if tokens_by_tiny else float('inf')

print(f'bytes_total={bytes_total}')
print(f'tokens_by_owt={tokens_by_owt} bytes/token={ratio_owt:.4f}')
print(f'tokens_by_tiny={tokens_by_tiny} bytes/token={ratio_tiny:.4f}')

# print a short example of tokenization counts for the first sample
s0 = (sample[0] + '<|endoftext|>')
print('\nExample first doc bytes=', len(s0.encode('utf-8')))
print('OWT tokens=', len(owt.encode(s0)))
print('Tiny tokens=', len(tiny.encode(s0)))
print('\nFirst 20 OWT ids:', owt.encode(s0)[:20])
print('First 40 Tiny ids:', tiny.encode(s0)[:40])
print('\nDecoded check OWT==orig:', owt.decode(owt.encode(s0))==s0)
print('Decoded check Tiny==orig:', tiny.decode(tiny.encode(s0))==s0)
