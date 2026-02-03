from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def _top_p_filtering(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    cutoff_mask = cumulative <= top_p
    if not cutoff_mask.any():
        cutoff_mask[0] = True
    filtered = torch.zeros_like(probs)
    keep_indices = sorted_indices[cutoff_mask]
    filtered[keep_indices] = probs[keep_indices]
    s = filtered.sum()
    if s <= 0:
        return probs
    return filtered / s


def generate(
    model: torch.nn.Module,
    input_ids: Iterable[int] | torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str = "cpu",
    eos_token_id: int | None = None,
) -> list[int]:
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
            probs = _top_p_filtering(probs, float(top_p))
            next_id = torch.multinomial(probs, num_samples=1).item()
            cur.append(int(next_id))
            if eos_token_id is not None and next_id == eos_token_id:
                break
    return cur
