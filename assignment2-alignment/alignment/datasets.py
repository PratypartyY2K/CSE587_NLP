from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from alignment.drgrpo_grader import extract_answer


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open() as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def infer_dataset_format(example: dict[str, Any]) -> str:
    if "problem" in example and "solution" in example:
        return "math"
    if "question" in example and "answer" in example:
        return "gsm8k"
    if "prompt" in example and "response" in example:
        return "sft"
    if "final_answer" in example and ("problem" in example or "question" in example):
        return "canonical"
    raise ValueError(f"Unable to infer dataset format from keys: {sorted(example.keys())}")


def extract_gsm8k_final_answer(answer: str) -> str | None:
    if "####" in answer:
        return answer.rsplit("####", 1)[-1].strip()

    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", answer)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_short_answer(solution: str, dataset_format: str) -> str | None:
    if dataset_format == "math":
        return extract_answer(solution)
    if dataset_format == "gsm8k":
        return extract_gsm8k_final_answer(solution)
    if dataset_format == "canonical":
        return solution.strip() or None
    return None


def normalize_example(
    example: dict[str, Any],
    dataset_format: str = "auto",
) -> dict[str, Any]:
    if dataset_format == "auto":
        dataset_format = infer_dataset_format(example)

    if dataset_format == "math":
        problem = example["problem"]
        solution = example["solution"]
        final_answer = extract_short_answer(solution, "math")
        return {
            "problem": problem,
            "solution": solution,
            "final_answer": final_answer,
            "dataset_format": "math",
            "source_example": example,
        }

    if dataset_format == "gsm8k":
        problem = example["question"]
        solution = example["answer"]
        final_answer = extract_short_answer(solution, "gsm8k")
        return {
            "problem": problem,
            "solution": solution,
            "final_answer": final_answer,
            "dataset_format": "gsm8k",
            "source_example": example,
        }

    if dataset_format == "canonical":
        problem = example.get("problem", example.get("question"))
        solution = example.get("solution", example.get("answer", ""))
        final_answer = example.get("final_answer")
        return {
            "problem": problem,
            "solution": solution,
            "final_answer": final_answer,
            "dataset_format": example.get("dataset_format", "canonical"),
            "source_example": example,
        }

    raise ValueError(f"Unsupported dataset_format: {dataset_format}")


def normalize_dataset(
    examples: list[dict[str, Any]],
    dataset_format: str = "auto",
) -> list[dict[str, Any]]:
    return [normalize_example(example, dataset_format=dataset_format) for example in examples]


def load_normalized_dataset(
    path: str | Path,
    dataset_format: str = "auto",
) -> list[dict[str, Any]]:
    return normalize_dataset(load_jsonl(path), dataset_format=dataset_format)
