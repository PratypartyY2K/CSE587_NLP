from __future__ import annotations

from typing import Any
import torch


def save_checkpoint(
    model: Any,
    optimizer: Any,
    iteration: int,
    out: Any,
):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(payload, out)


def load_checkpoint(src: Any, model: Any, optimizer: Any) -> int:
    payload = torch.load(src, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload["iteration"])
