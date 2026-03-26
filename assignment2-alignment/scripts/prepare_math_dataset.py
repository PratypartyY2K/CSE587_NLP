from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignment.datasets import load_normalized_dataset, write_jsonl

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--dataset-format",
        default="auto",
        choices=["auto", "math", "gsm8k", "canonical"],
    )
    args = parser.parse_args()

    normalized_examples = load_normalized_dataset(
        path=args.input_path,
        dataset_format=args.dataset_format,
    )
    write_jsonl(args.output_path, normalized_examples)

    num_missing_final_answers = sum(
        example["final_answer"] is None for example in normalized_examples
    )
    summary = {
        "num_examples": len(normalized_examples),
        "num_missing_final_answers": num_missing_final_answers,
        "input_path": args.input_path,
        "output_path": args.output_path,
        "dataset_format": args.dataset_format,
    }
    logger.info("%s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
