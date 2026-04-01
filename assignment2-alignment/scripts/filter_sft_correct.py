from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignment.datasets import load_normalized_dataset, write_jsonl
from alignment.drgrpo_grader import r1_zero_reward_fn


QUESTION_RE = re.compile(r"\nUser:\s*(.*?)\nAssistant:\s*<think>\s*$", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="data/sft.jsonl")
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--reference-path",
        action="append",
        default=["data/MATH/train.jsonl"],
        help="Canonical-format or raw math/gsm8k dataset path(s) used to recover ground-truth answers.",
    )
    parser.add_argument(
        "--reference-format",
        action="append",
        default=["auto"],
        choices=["auto", "math", "gsm8k", "canonical"],
        help="Dataset format(s) for --reference-path entries, in the same order.",
    )
    parser.add_argument("--summary-path", default=None)
    return parser.parse_args()


def extract_question(prompt: str) -> str:
    match = QUESTION_RE.search(prompt)
    if match is None:
        raise ValueError("Could not extract question from prompt.")
    return match.group(1)


def canonicalize_question(text: str) -> str:
    text = text.replace("\\!", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_reference_records(path: str, dataset_format: str) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records:
        return []

    first = records[0]
    if "question" in first and "deepseek_grade" in first:
        normalized_records = []
        for record in records:
            normalized_records.append(
                {
                    "problem": canonicalize_question(record["question"]),
                    "answers": [record["solution"]] if record.get("solution") else [],
                    "is_correct": str(record.get("deepseek_grade", "")).strip().lower() == "yes",
                }
            )
        return normalized_records

    normalized_records = []
    for example in load_normalized_dataset(path, dataset_format=dataset_format):
        answers = []
        if example["final_answer"] is not None:
            answers.append(example["final_answer"])
        if example["solution"]:
            answers.append(example["solution"])
        normalized_records.append(
            {
                "problem": canonicalize_question(example["problem"]),
                "answers": answers,
                "is_correct": None,
            }
        )
    return normalized_records


def main() -> None:
    args = parse_args()
    if len(args.reference_path) != len(args.reference_format):
        raise ValueError("--reference-path and --reference-format must have the same length.")

    references_by_problem: dict[str, dict[str, Any]] = defaultdict(lambda: {"answers": [], "is_correct": None})
    for path, dataset_format in zip(args.reference_path, args.reference_format, strict=True):
        for record in load_reference_records(path, dataset_format=dataset_format):
            ref = references_by_problem[record["problem"]]
            for answer in record["answers"]:
                if answer not in ref["answers"]:
                    ref["answers"].append(answer)
            if record["is_correct"] is not None:
                ref["is_correct"] = record["is_correct"]

    rows = [json.loads(line) for line in Path(args.input_path).read_text().splitlines() if line.strip()]
    filtered_rows: list[dict[str, str]] = []
    unmatched = 0
    correct = 0

    for row in rows:
        problem = canonicalize_question(extract_question(row["prompt"]))
        reference = references_by_problem.get(problem)
        if not reference:
            unmatched += 1
            continue

        if reference["is_correct"] is not None:
            keep = bool(reference["is_correct"])
        else:
            metrics = r1_zero_reward_fn(row["response"], reference["answers"])
            keep = float(metrics["answer_reward"]) == 1.0

        if keep:
            filtered_rows.append(row)
            correct += 1

    write_jsonl(args.output_path, filtered_rows)

    summary = {
        "input_path": args.input_path,
        "output_path": args.output_path,
        "input_examples": len(rows),
        "matched_examples": len(rows) - unmatched,
        "unmatched_examples": unmatched,
        "filtered_examples": len(filtered_rows),
        "retained_fraction": (len(filtered_rows) / len(rows)) if rows else 0.0,
    }

    summary_path = Path(args.summary_path) if args.summary_path else Path(args.output_path).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
