"""Sampling / generation helpers for TransformerLM.

Exports:
- run_generate_impl: autoregressive sampling with temperature and top-p (nucleus) sampling.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def _top_p_filtering(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero-out probabilities outside the smallest nucleus with cumulative prob >= top_p.

    probs: 1D tensor of probabilities (not logits), shape (V,)
    Returns a renormalized probability tensor.
    """
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    # keep tokens with cumulative <= top_p; we must keep at least the top token
    cutoff_mask = cumulative <= top_p
    # ensure at least one True
    if not cutoff_mask.any():
        cutoff_mask[0] = True
    # mask out tokens beyond nucleus
    filtered = torch.zeros_like(probs)
    keep_indices = sorted_indices[cutoff_mask]
    filtered[keep_indices] = probs[keep_indices]
    s = filtered.sum()
    if s <= 0:
        # if numerical issues, fall back to original probs
        return probs
    return filtered / s


def run_generate_impl(
    model: torch.nn.Module,
    input_ids: Iterable[int] | torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
    eos_token_id: int | None = None,
) -> list[int]:
    """Autoregressively generate tokens from `model` starting from `input_ids`.

    Parameters
    - model: a callable that maps input tensor of shape (1, seq_len) to logits (1, seq_len, vocab_size)
    - input_ids: sequence of initial token ids (list/int iterable or torch.Tensor)
    - max_new_tokens: maximum number of tokens to generate
    - temperature: softmax temperature (float > 0). Values <1 concentrate, >1 flatten.
    - top_p: nucleus sampling threshold in (0,1]; if 1.0 no filtering
    - device: device string where tensors and model are located
    - eos_token_id: optional id that terminates generation when sampled

    Returns the full output token id sequence (prompt + generated) as a Python list of ints.
    """
    model = model.to(device)
    model.eval()

    if isinstance(input_ids, torch.Tensor):
        cur = input_ids.detach().cpu().tolist() if input_ids.device != torch.device(device) else input_ids.tolist()
    else:
        cur = list(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([cur], dtype=torch.long, device=device)
            logits = model(input_tensor)  # (1, seq_len, V)
            last_logits = logits[0, -1, :]  # (V,)
            if temperature != 1.0:
                logits_scaled = last_logits / float(temperature)
            else:
                logits_scaled = last_logits
            probs = F.softmax(logits_scaled, dim=-1)
            # apply top-p nucleus filtering
            probs = _top_p_filtering(probs, float(top_p))
            # sample
            next_id = torch.multinomial(probs, num_samples=1).item()
            cur.append(int(next_id))
            if eos_token_id is not None and next_id == eos_token_id:
                break
    return cur
