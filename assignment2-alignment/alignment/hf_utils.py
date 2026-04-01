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


def resolve_model_source(model_name_or_path: str | Path, cache_dir: str | Path | None = None) -> str:
    source_path = Path(model_name_or_path).expanduser()
    if source_path.exists():
        return str(source_path.resolve())

    resolved_cache_dir = None if cache_dir is None else str(Path(cache_dir).expanduser())
    try:
        return snapshot_download(
            repo_id=str(model_name_or_path),
            allow_patterns=MODEL_ALLOW_PATTERNS,
            cache_dir=resolved_cache_dir,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download model repo '{model_name_or_path}'. "
            "If a previous Hugging Face download was interrupted, retry after clearing the "
            "partial cache for that repo, pass a fully-downloaded local path, or set a "
            "different Hugging Face cache directory."
        ) from exc
