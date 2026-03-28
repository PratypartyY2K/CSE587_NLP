from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignment.datasets import load_normalized_dataset
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.hf_utils import resolve_model_source

logger = logging.getLogger(__name__)


def evaluate_vllm(
    vllm_model: Any,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: Any,
    output_path: Path,
    examples: list[dict],
    model_name_or_path: str,
) -> dict:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    results = []
    for idx, (example, prompt, output, ground_truth) in enumerate(
        zip(examples, prompts, outputs, ground_truths)
    ):
        generation = output.outputs[0].text
        metrics = reward_fn(generation, ground_truth)
        results.append(
            {
                "index": idx,
                "example": example,
                "prompt": prompt,
                "generation": generation,
                "metrics": metrics,
                "model_name_or_path": model_name_or_path,
            }
        )

    with output_path.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    category_counts = {
        "correct_format1_answer1": 0,
        "formatted_wrong_format1_answer0": 0,
        "unformatted_format0_answer0": 0,
    }
    for result in results:
        metrics = result["metrics"]
        if metrics["format_reward"] == 1.0 and metrics["answer_reward"] == 1.0:
            category_counts["correct_format1_answer1"] += 1
        elif metrics["format_reward"] == 1.0 and metrics["answer_reward"] == 0.0:
            category_counts["formatted_wrong_format1_answer0"] += 1
        else:
            category_counts["unformatted_format0_answer0"] += 1

    return {
        "num_examples": len(results),
        "mean_metrics": {
            key: mean(result["metrics"][key] for result in results)
            for key in ("format_reward", "answer_reward", "reward")
        },
        "category_counts": category_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="data/gsm8k/test.jsonl")
    parser.add_argument(
        "--dataset-format",
        default="gsm8k",
        choices=["auto", "math", "gsm8k", "canonical"],
    )
    parser.add_argument("--prompt-path", default="alignment/prompts/r1_zero.prompt")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()
    vllm = importlib.import_module("vllm")

    input_path = Path(args.input_path)
    prompt_path = Path(args.prompt_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_examples = load_normalized_dataset(
        input_path,
        dataset_format=args.dataset_format,
    )
    prompt_template = prompt_path.read_text()
    prompts = [
        prompt_template.format(question=example["problem"])
        for example in normalized_examples
    ]
    ground_truths = [
        example["final_answer"] if example["final_answer"] is not None else example["solution"]
        for example in normalized_examples
    ]

    resolved_model_path = resolve_model_source(args.model_name_or_path)

    model = vllm.LLM(
        model=resolved_model_path,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )
    sampling_params = vllm.SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    results_path = output_dir / "results.jsonl"
    summary = evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=results_path,
        examples=normalized_examples,
        model_name_or_path=args.model_name_or_path,
    )
    summary["results_path"] = str(results_path)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info("Wrote results to %s", results_path)
    logger.info("Wrote summary to %s", summary_path)
    for key, value in summary["mean_metrics"].items():
        logger.info("%s: %.4f", key, value)
    for key, value in summary["category_counts"].items():
        logger.info("%s: %d", key, value)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
