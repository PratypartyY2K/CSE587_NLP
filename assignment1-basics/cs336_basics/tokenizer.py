import regex as re
from collections import Counter
from typing import List, Dict, Tuple

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    min_allowed = 256 + len(special_tokens)
    if vocab_size < min_allowed:
        raise ValueError(f"`vocab_size` must be at least {min_allowed} (256 + number of special tokens).")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    split_regex = re.compile(GPT2_SPLIT_PATTERN)

    if special_tokens:
        sorted_tokens = sorted(special_tokens, key=len, reverse=True)
        escaped = "|".join(re.escape(t) for t in sorted_tokens)
        split_pattern = f"({escaped})"
        segments = re.split(split_pattern, text)
    else:
        segments = [text]

    word_counts: Counter = Counter()
    for seg in segments:
        if not seg:
            continue
        if special_tokens and seg in special_tokens:
            continue
        for m in split_regex.finditer(seg):
            tok_bytes = m.group(0).encode("utf-8")
            token_ids = tuple(tok_bytes)
            word_counts[token_ids] += 1

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
            break

        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]])))

        a_bytes = vocab[best_pair[0]]
        b_bytes = vocab[best_pair[1]]
        merges.append((a_bytes, b_bytes))

        new_id = 256 + i
        vocab[new_id] = a_bytes + b_bytes

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

    current_id = max(vocab.keys()) + 1
    for st in special_tokens:
        vocab[current_id] = st.encode("utf-8")
        current_id += 1

    return vocab, merges
