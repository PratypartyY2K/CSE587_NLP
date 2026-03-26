from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize prompts and outputs separately, then build shifted LM tensors."""
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length")

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define either pad_token_id or eos_token_id")
        pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = tokenizer.pad_token_id

    prompt_tokenized = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    output_tokenized = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
    prompt_and_output_ids = [
        prompt_ids + output_ids
        for prompt_ids, output_ids in zip(prompt_tokenized, output_tokenized, strict=True)
    ]

    max_len = max((len(ids) for ids in prompt_and_output_ids), default=0)
    batch_size = len(prompt_and_output_ids)

    if max_len == 0:
        empty_ids = torch.empty((batch_size, 0), dtype=torch.long)
        empty_mask = torch.empty((batch_size, 0), dtype=torch.bool)
        return {
            "input_ids": empty_ids,
            "labels": empty_ids.clone(),
            "response_mask": empty_mask,
        }

    padded_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, max_len - 1), dtype=torch.bool)

    for row_idx, (prompt_ids, output_ids, all_ids) in enumerate(
        zip(prompt_tokenized, output_tokenized, prompt_and_output_ids, strict=True)
    ):
        seq_len = len(all_ids)
        padded_ids[row_idx, :seq_len] = torch.tensor(all_ids, dtype=torch.long)

        if output_ids:
            start_idx = max(len(prompt_ids) - 1, 0)
            end_idx = start_idx + len(output_ids)
            response_mask[row_idx, start_idx:end_idx] = True

    return {
        "input_ids": padded_ids[:, :-1],
        "labels": padded_ids[:, 1:],
        "response_mask": response_mask,
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropy over the vocabulary dimension."""
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = logits - log_z
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)
