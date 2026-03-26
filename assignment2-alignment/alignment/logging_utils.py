from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


class GenerationLog(list[dict[str, Any]]):
    """List of per-example logging rows with aggregate summary statistics."""

    def __init__(self, rows: list[dict[str, Any]], summary: dict[str, float | int | None]) -> None:
        super().__init__(rows)
        self.summary = summary


def _coerce_optional_sequence(
    values: Sequence[Any] | None,
    name: str,
    expected_len: int,
) -> list[Any] | None:
    if values is None:
        return None
    coerced = list(values)
    if len(coerced) != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got {len(coerced)}")
    return coerced


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return mean(values)


def _infer_response_length(generation: str) -> int:
    stripped = generation.strip()
    if not stripped:
        return 0
    return len(stripped.split())


def _normalize_reward_info(reward_info: Mapping[str, Any] | float | int | None) -> dict[str, Any] | None:
    if reward_info is None:
        return None
    if isinstance(reward_info, Mapping):
        reward_dict = dict(reward_info)
        if "total_reward" in reward_dict and "reward" not in reward_dict:
            reward_dict["reward"] = reward_dict["total_reward"]
        return reward_dict
    return {"reward": reward_info}


def _is_correct_response(
    reward_dict: dict[str, Any] | None,
    explicit_is_correct: bool | None,
) -> bool | None:
    if explicit_is_correct is not None:
        return explicit_is_correct
    if reward_dict is None:
        return None
    if "answer_reward" in reward_dict:
        return bool(reward_dict["answer_reward"])
    if "reward" in reward_dict:
        return bool(reward_dict["reward"])
    return None


def log_generations(
    prompts: Sequence[str],
    generations: Sequence[str],
    *,
    ground_truths: Sequence[str | None] | None = None,
    references: Sequence[str | None] | None = None,
    reward_info: Sequence[Mapping[str, Any] | float | int | None] | None = None,
    rewards: Sequence[float | int | None] | None = None,
    avg_token_entropies: Sequence[float | None] | None = None,
    response_lengths: Sequence[int | None] | None = None,
    is_correct: Sequence[bool | None] | None = None,
    metadata: Sequence[Mapping[str, Any] | None] | None = None,
    output_path: str | Path | None = None,
    wandb_run: Any | None = None,
    table_name: str = "generations",
    summary_name: str = "generation_summary",
    step: int | None = None,
    max_rows: int | None = None,
) -> GenerationLog:
    """Log per-example generations and aggregate response statistics.

    The function records prompt, generation, ground truth, reward components,
    average token entropy, and response length for each example. It also computes
    aggregate response-length summaries over all, correct, and incorrect responses.
    """
    prompt_list = list(prompts)
    generation_list = list(generations)
    if len(prompt_list) != len(generation_list):
        raise ValueError(
            "prompts and generations must have the same length, "
            f"got {len(prompt_list)} and {len(generation_list)}"
        )

    if ground_truths is not None and references is not None:
        raise ValueError("pass only one of ground_truths or references")

    ground_truth_list = _coerce_optional_sequence(
        ground_truths if ground_truths is not None else references,
        "ground_truths" if ground_truths is not None else "references",
        len(prompt_list),
    )
    reward_info_list = _coerce_optional_sequence(reward_info, "reward_info", len(prompt_list))
    scalar_rewards_list = _coerce_optional_sequence(rewards, "rewards", len(prompt_list))
    avg_token_entropy_list = _coerce_optional_sequence(
        avg_token_entropies,
        "avg_token_entropies",
        len(prompt_list),
    )
    response_length_list = _coerce_optional_sequence(response_lengths, "response_lengths", len(prompt_list))
    is_correct_list = _coerce_optional_sequence(is_correct, "is_correct", len(prompt_list))
    metadata_list = _coerce_optional_sequence(metadata, "metadata", len(prompt_list))

    limit = len(prompt_list) if max_rows is None else min(max_rows, len(prompt_list))
    rows: list[dict[str, Any]] = []
    all_lengths: list[float] = []
    correct_lengths: list[float] = []
    incorrect_lengths: list[float] = []

    for idx in range(limit):
        row: dict[str, Any] = {
            "index": idx,
            "prompt": prompt_list[idx],
            "generation": generation_list[idx],
        }
        if ground_truth_list is not None:
            row["ground_truth"] = ground_truth_list[idx]

        reward_dict = None
        if reward_info_list is not None:
            reward_dict = _normalize_reward_info(reward_info_list[idx])
        elif scalar_rewards_list is not None:
            reward_dict = _normalize_reward_info(scalar_rewards_list[idx])
        if reward_dict is not None:
            row["reward_info"] = reward_dict

        if avg_token_entropy_list is not None:
            row["avg_token_entropy"] = avg_token_entropy_list[idx]

        response_length = (
            response_length_list[idx]
            if response_length_list is not None and response_length_list[idx] is not None
            else _infer_response_length(generation_list[idx])
        )
        row["response_length"] = response_length
        all_lengths.append(float(response_length))

        correct = _is_correct_response(
            reward_dict=reward_dict,
            explicit_is_correct=is_correct_list[idx] if is_correct_list is not None else None,
        )
        if correct is not None:
            row["is_correct"] = correct
            if correct:
                correct_lengths.append(float(response_length))
            else:
                incorrect_lengths.append(float(response_length))

        if metadata_list is not None:
            row["metadata"] = metadata_list[idx]
        rows.append(row)

    summary: dict[str, float | int | None] = {
        "num_examples": len(rows),
        "avg_response_length": _mean_or_none(all_lengths),
        "avg_response_length_correct": _mean_or_none(correct_lengths),
        "avg_response_length_incorrect": _mean_or_none(incorrect_lengths),
    }

    generation_log = GenerationLog(rows=rows, summary=summary)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in generation_log:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary_path = path.with_name(f"{path.stem}_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if wandb_run is not None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("wandb logging requested but wandb is not installed") from exc

        columns = [
            "index",
            "prompt",
            "generation",
            "ground_truth",
            "reward_info",
            "avg_token_entropy",
            "response_length",
            "is_correct",
            "metadata",
        ]
        table = wandb.Table(columns=columns)
        for row in generation_log:
            table.add_data(
                row["index"],
                row["prompt"],
                row["generation"],
                row.get("ground_truth"),
                row.get("reward_info"),
                row.get("avg_token_entropy"),
                row.get("response_length"),
                row.get("is_correct"),
                row.get("metadata"),
            )

        payload: dict[str, Any] = {
            table_name: table,
            summary_name: summary,
        }
        if step is None:
            wandb_run.log(payload)
        else:
            wandb_run.log(payload, step=step)

    return generation_log
