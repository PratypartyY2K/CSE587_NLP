from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase


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

    prompt_tokenized = tokenizer(prompt_strs, add_special_tokens=False, verbose=False)["input_ids"]
    output_tokenized = tokenizer(output_strs, add_special_tokens=False, verbose=False)["input_ids"]
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


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Get per-token conditional log-probabilities and optional token entropy."""
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)

    output = {
        "log_probs": torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1),
    }
    if return_token_entropy:
        output["token_entropy"] = compute_entropy(logits)

    return output


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """Sum over tensor values selected by mask and divide by a constant."""
    masked_tensor = tensor * mask.to(dtype=tensor.dtype)
    if dim is None:
        return masked_tensor.sum() / normalize_constant
    return masked_tensor.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute masked SFT loss for one microbatch and backpropagate it."""
    token_loss = -masked_normalize(
        tensor=policy_log_probs,
        mask=response_mask,
        normalize_constant=normalize_constant,
    )
    ce_loss = token_loss / policy_log_probs.shape[0]
    loss = ce_loss / gradient_accumulation_steps
    loss.backward()

    metadata = {
        "ce_loss": ce_loss.detach(),
        "num_response_tokens": response_mask.sum().detach(),
    }
    return loss.detach(), metadata
