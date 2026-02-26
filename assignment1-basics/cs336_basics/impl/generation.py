from __future__ import annotations

from collections.abc import Iterable

import torch


def _top_p_filtering(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    keep_count = int(torch.searchsorted(cumulative, torch.tensor(top_p, device=probs.device), right=True).item()) + 1
    keep_count = min(keep_count, probs.numel())
    filtered = torch.zeros_like(probs)
    keep_indices = sorted_indices[:keep_count]
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
    model_context_len = getattr(model, "context_length", None)

    if isinstance(input_ids, torch.Tensor):
        cur = input_ids.detach().tolist()
    else:
        cur = list(input_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if model_context_len is not None and model_context_len > 0 and len(cur) > model_context_len:
                model_input = cur[-model_context_len:]
            else:
                model_input = cur
            input_tensor = torch.as_tensor(model_input, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(input_tensor)  # (1, seq_len, V)
            last_logits = logits[0, -1, :]  # (V,)
            if temperature != 1.0:
                logits_scaled = last_logits / float(temperature)
            else:
                logits_scaled = last_logits
            probs = torch.softmax(logits_scaled, dim=-1)
            probs = _top_p_filtering(probs, float(top_p))
            next_id = torch.multinomial(probs, num_samples=1).item()
            cur.append(int(next_id))
            if eos_token_id is not None and next_id == eos_token_id:
                break
    return cur
