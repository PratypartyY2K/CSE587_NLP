from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
    "*.bin",
    "*.py",
    "*.tiktoken",
]


def resolve_model_source(model_name_or_path: str | Path) -> str:
    source_path = Path(model_name_or_path).expanduser()
    if source_path.exists():
        return str(source_path.resolve())

    try:
        return snapshot_download(
            repo_id=str(model_name_or_path),
            allow_patterns=MODEL_ALLOW_PATTERNS,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download model repo '{model_name_or_path}'. "
            "If a previous Hugging Face download was interrupted, retry after clearing the "
            "partial cache for that repo or pass a fully-downloaded local path."
        ) from exc
