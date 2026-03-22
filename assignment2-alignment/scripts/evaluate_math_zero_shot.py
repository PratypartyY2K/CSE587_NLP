"""
Evaluate zero-shot MATH performance with the r1_zero prompt and vLLM.

Example:

```
/Users/pratyushkumar/.venv-vllm-metal/bin/python scripts/evaluate_math_zero_shot.py \
    --model-name-or-path Qwen/Qwen2.5-Math-1.5B \
    --output-dir outputs/qwen2_5_math_1_5b_zero_shot
```
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignment.drgrpo_grader import r1_zero_reward_fn

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with path.open() as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def load_prompt_template(path: Path) -> str:
    return path.read_text()


def build_prompts(examples: list[dict], prompt_template: str) -> list[str]:
    return [prompt_template.format(question=example["problem"]) for example in examples]


def aggregate_metrics(
    metrics_list: list[dict[str, float]],
    examples: list[dict],
    prompts: list[str],
    responses: list[str],
) -> dict:
    summary = {
        "num_examples": len(metrics_list),
        "mean_metrics": {
            key: mean(metrics[key] for metrics in metrics_list)
            for key in sorted(metrics_list[0].keys())
        },
        "category_counts": {
            "correct_format1_answer1": 0,
            "formatted_wrong_format1_answer0": 0,
            "unformatted_format0_answer0": 0,
        },
    }

    for metrics in metrics_list:
        if metrics["format_reward"] == 1.0 and metrics["answer_reward"] == 1.0:
            summary["category_counts"]["correct_format1_answer1"] += 1
        elif metrics["format_reward"] == 1.0 and metrics["answer_reward"] == 0.0:
            summary["category_counts"]["formatted_wrong_format1_answer0"] += 1
        elif metrics["format_reward"] == 0.0 and metrics["answer_reward"] == 0.0:
            summary["category_counts"]["unformatted_format0_answer0"] += 1

    sample_indices = {
        "correct_format1_answer1": [],
        "formatted_wrong_format1_answer0": [],
        "unformatted_format0_answer0": [],
    }
    for idx, metrics in enumerate(metrics_list):
        if metrics["format_reward"] == 1.0 and metrics["answer_reward"] == 1.0:
            key = "correct_format1_answer1"
        elif metrics["format_reward"] == 1.0 and metrics["answer_reward"] == 0.0:
            key = "formatted_wrong_format1_answer0"
        else:
            key = "unformatted_format0_answer0"
        if len(sample_indices[key]) < 10:
            sample_indices[key].append(
                {
                    "index": idx,
                    "problem": examples[idx]["problem"],
                    "ground_truth": examples[idx]["solution"],
                    "prompt": prompts[idx],
                    "response": responses[idx],
                    "metrics": metrics,
                }
            )
    summary["samples"] = sample_indices
    return summary


def write_results(
    output_path: Path,
    examples: list[dict],
    prompts: list[str],
    ground_truths: list[str],
    responses: list[str],
    metrics_list: list[dict[str, float]],
    model_name_or_path: str,
) -> None:
    with output_path.open("w") as fout:
        for idx, (example, prompt, response, ground_truth, metrics) in enumerate(
            zip(examples, prompts, responses, ground_truths, metrics_list)
        ):
            record = {
                "index": idx,
                "model_name_or_path": model_name_or_path,
                "prompt": prompt,
                "problem": example["problem"],
                "ground_truth": ground_truth,
                "generation": response,
                "metrics": metrics,
                "example": example,
            }
            fout.write(json.dumps(record) + "\n")


def evaluate_vllm(
    vllm_model: "LLM",
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: "SamplingParams",
    output_path: Path,
    examples: list[dict],
    model_name_or_path: str,
) -> dict:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    raw_outputs = vllm_model.generate(prompts, eval_sampling_params)
    responses = [output.outputs[0].text for output in raw_outputs]
    metrics_list = [reward_fn(response, ground_truth) for response, ground_truth in zip(responses, ground_truths)]
    write_results(
        output_path=output_path,
        examples=examples,
        prompts=prompts,
        ground_truths=ground_truths,
        responses=responses,
        metrics_list=metrics_list,
        model_name_or_path=model_name_or_path,
    )
    return aggregate_metrics(metrics_list, examples, prompts, responses)


def main(args: argparse.Namespace) -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM is required for this script. Run it with the vllm-metal environment, "
            "for example: `/Users/pratyushkumar/.venv-vllm-metal/bin/python "
            "scripts/evaluate_math_zero_shot.py ...`."
        ) from exc

    input_path = Path(args.input_path)
    prompt_path = Path(args.prompt_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    examples = load_jsonl(input_path)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    logger.info("Loaded %d examples from %s", len(examples), input_path)

    prompt_template = load_prompt_template(prompt_path)
    prompts = build_prompts(examples, prompt_template)
    ground_truths = [example["solution"] for example in examples]
    model = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    summary = evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_path=results_path,
        examples=examples,
        model_name_or_path=args.model_name_or_path,
    )

    summary["input_path"] = str(input_path)
    summary["prompt_path"] = str(prompt_path)
    summary["results_path"] = str(results_path)
    summary["sampling_params"] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    summary["model_name_or_path"] = args.model_name_or_path
    summary["runtime"] = "vllm"

    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote per-example results to %s", results_path)
    logger.info("Wrote summary metrics to %s", summary_path)
    for key, value in summary["mean_metrics"].items():
        logger.info("%s: %.4f", key, value)
    for key, value in summary["category_counts"].items():
        logger.info("%s: %d", key, value)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        default="data/MATH/validation.jsonl",
        help="Path to the MATH validation JSONL file.",
    )
    parser.add_argument(
        "--prompt-path",
        type=str,
        default="alignment/prompts/r1_zero.prompt",
        help="Path to the prompt template.",
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HF model name or local model path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where results.jsonl and summary.json will be written.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum sequence length for vLLM.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per example.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of validation examples to evaluate.",
    )
    parsed_args = parser.parse_args()
    logger.info("running %s", " ".join(sys.argv))
    main(parsed_args)
    logger.info("finished running %s", sys.argv[0])
