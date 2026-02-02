"""Byte-level BPE training and tokenizer implementation.

Exports `train_bpe` (train a byte-level BPE tokenizer) and `Tokenizer`.
"""

from __future__ import annotations

import regex as re
from collections import Counter
from collections.abc import Iterator
import json

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    min_allowed = 256 + len(special_tokens)
    if vocab_size < min_allowed:
        raise ValueError(f"`vocab_size` must be at least {min_allowed} (256 + number of special tokens).")

    with open(input_path, encoding="utf-8") as f:
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

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

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


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # inverse mapping bytes->id
        self.vocab_inv: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        # map special token string -> id if present in vocab
        self.special_token_to_id: dict[str, int] = {}
        for st in self.special_tokens:
            b = st.encode("utf-8")
            if b in self.vocab_inv:
                self.special_token_to_id[st] = self.vocab_inv[b]

        # merge ranks: earlier merges have lower rank
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self.merges)}

        # compile split pattern
        self.split_regex = re.compile(GPT2_SPLIT_PATTERN)

        # prepare special token split pattern (longest-first)
        if self.special_tokens:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self._special_pattern = re.compile("(" + "|".join(re.escape(t) for t in sorted_tokens) + ")")
            self._max_special_len = max(len(t) for t in sorted_tokens)
        else:
            self._special_pattern = None
            self._max_special_len = 0

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_vocab = json.load(f)
        vocab: dict[int, bytes] = {int(idx): token.encode("utf-8") for token, idx in gpt2_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue
                parts = line.split(" ")
                if len(parts) != 2:
                    continue
                merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab, merges, special_tokens)

    def _apply_bpe_to_token_bytes(self, token_bytes: bytes) -> list[int]:
        word = [self.vocab_inv[bytes([b])] for b in token_bytes]

        while True:
            found = {}
            for i in range(len(word) - 1):
                pair = (self.vocab[word[i]], self.vocab[word[i + 1]])
                if pair in self.merge_ranks and pair not in found:
                    found[pair] = i
            if not found:
                break
            best_pair = min(found.keys(), key=lambda p: self.merge_ranks[p])
            new_id = self.vocab_inv.get(best_pair[0] + best_pair[1])
            if new_id is None:
                break
            new_word = []
            i = 0
            L = len(word)
            while i < L:
                if i < L - 1 and self.vocab[word[i]] == best_pair[0] and self.vocab[word[i + 1]] == best_pair[1]:
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        if self._special_pattern:
            parts = self._special_pattern.split(text)
        else:
            parts = [text]
        for part in parts:
            if not part:
                continue
            if part in self.special_token_to_id:
                ids.append(self.special_token_to_id[part])
            else:
                for m in self.split_regex.finditer(part):
                    token_bytes = m.group(0).encode("utf-8")
                    ids.extend(self._apply_bpe_to_token_bytes(token_bytes))
        return ids

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        buffer = ""
        lookahead = max(1024, self._max_special_len + 16)
        for chunk in iterable:
            if not isinstance(chunk, str):
                chunk = chunk.decode("utf-8")
            text = buffer + chunk
            cutoff = len(text) - lookahead
            if cutoff < 0:
                buffer = text
                continue

            last_yield_pos = 0
            if self._special_pattern:
                parts = self._special_pattern.split(text)
            else:
                parts = [text]

            pos = 0
            stop = False
            for part in parts:
                if part == "":
                    pos += 0
                    continue
                part_start = pos
                part_end = pos + len(part)

                if part in self.special_token_to_id:
                    if part_end <= cutoff:
                        yield self.special_token_to_id[part]
                        last_yield_pos = part_end
                        pos = part_end
                        continue
                    else:
                        stop = True
                        break

                for m in self.split_regex.finditer(part):
                    m_end = part_start + m.end()
                    if m_end <= cutoff:
                        token_bytes = m.group(0).encode("utf-8")
                        for tid in self._apply_bpe_to_token_bytes(token_bytes):
                            yield tid
                        last_yield_pos = m_end
                    else:
                        stop = True
                        break
                if stop:
                    break
                pos = part_end

            buffer = text[last_yield_pos:]

        if buffer:
            if self._special_pattern:
                parts = self._special_pattern.split(buffer)
            else:
                parts = [buffer]
            for part in parts:
                if not part:
                    continue
                if part in self.special_token_to_id:
                    yield self.special_token_to_id[part]
                else:
                    for m in self.split_regex.finditer(part):
                        token_bytes = m.group(0).encode("utf-8")
                        for tid in self._apply_bpe_to_token_bytes(token_bytes):
                            yield tid

    def decode(self, ids: list[int]) -> str:
        parts: list[bytes] = []
        for tid in ids:
            if tid in self.vocab:
                parts.append(self.vocab[tid])
        return b"".join(parts).decode("utf-8", errors="replace")
