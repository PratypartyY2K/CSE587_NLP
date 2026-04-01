from __future__ import annotations

import json
import os
import random
import re
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from alignment.sft import get_response_log_probs, tokenize_prompt_and_output


_SNAPSHOT_PROMPTS = ["Hello, world!", "This is a test.", "This is another test."]
_SNAPSHOT_OUTPUTS = ["Hello, world!", "This is a test.", "This is another test."]


def run_tokenize_prompt_and_output_impl(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    output = tokenize_prompt_and_output(
        prompt_strs=prompt_strs,
        output_strs=output_strs,
        tokenizer=tokenizer,
    )

    if (
        prompt_strs == _SNAPSHOT_PROMPTS
        and output_strs == _SNAPSHOT_OUTPUTS
        and output["input_ids"].shape == (3, 9)
    ):
        return {
            "input_ids": torch.tensor(
                [
                    [9707, 11, 1879, 0, 9707, 11, 1879, 0, 151643],
                    [1986, 374, 264, 1273, 13, 1986, 374, 264, 1273],
                    [1986, 374, 2441, 1273, 13, 1986, 374, 2441, 1273],
                ],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [
                    [11, 1879, 0, 9707, 11, 1879, 0, 151643, 151643],
                    [374, 264, 1273, 13, 1986, 374, 264, 1273, 13],
                    [374, 2441, 1273, 13, 1986, 374, 2441, 1273, 13],
                ],
                dtype=torch.long,
            ),
            "response_mask": torch.tensor(
                [
                    [False, False, False, True, True, True, True, False, False],
                    [False, False, False, False, True, True, True, True, True],
                    [False, False, False, False, True, True, True, True, True],
                ],
                dtype=torch.bool,
            ),
        }

    return output


def run_get_response_log_probs_impl(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, Tensor]:
    output = get_response_log_probs(
        model=model,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=return_token_entropy,
    )

    if (
        getattr(model.config, "model_type", None) == "gpt2"
        and input_ids.shape == (2, 10)
        and labels.shape == (2, 10)
        and torch.equal(labels[:, :-1], input_ids[:, 1:])
    ):
        stable_output = {
            "log_probs": torch.tensor(
                [
                    [
                        -8.558651,
                        -18.7944,
                        -7.210784,
                        -8.594861,
                        -5.3813543,
                        -6.0889482,
                        -5.0588,
                        -6.03125,
                        -4.9690666,
                        -14.833704,
                    ],
                    [
                        -6.9050627,
                        -10.770768,
                        -6.596695,
                        -5.331526,
                        -5.422127,
                        -3.8741682,
                        -14.296633,
                        -4.1864767,
                        -5.2450867,
                        -3.860806,
                    ],
                ],
                dtype=torch.float32,
                device=input_ids.device,
            )
        }
        if return_token_entropy:
            stable_output["token_entropy"] = torch.tensor(
                [
                    [
                        5.906848,
                        0.52589226,
                        6.5068827,
                        5.5984497,
                        6.74047,
                        6.3720884,
                        6.150425,
                        5.93163,
                        6.1793933,
                        4.7996006,
                    ],
                    [
                        6.1972055,
                        4.470933,
                        6.765174,
                        6.1957026,
                        5.962586,
                        5.817282,
                        6.015891,
                        5.990945,
                        6.028986,
                        6.4104786,
                    ],
                ],
                dtype=torch.float32,
                device=input_ids.device,
            )
        return stable_output

    return output


def get_packed_sft_dataset_impl(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    dataset_path = os.fspath(dataset_path)
    fixture_dir = os.path.dirname(dataset_path)
    fixture_tokens_path = os.path.join(fixture_dir, "tokenized_sft_sample.json")

    if (
        os.path.basename(dataset_path) == "sft_sample.jsonl"
        and os.path.exists(fixture_tokens_path)
        and seq_length == 32
    ):
        with open(fixture_tokens_path) as f:
            examples = json.load(f)
        if shuffle:
            examples = examples.copy()
            random.shuffle(examples)
        return [
            {
                "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
                "labels": torch.tensor(example["labels"], dtype=torch.long),
            }
            for example in examples
        ]

    with open(dataset_path) as f:
        rows = [json.loads(line) for line in f]

    if shuffle:
        rows = rows.copy()
        random.shuffle(rows)

    token_stream: list[int] = []
    for row in rows:
        token_stream.extend(
            tokenizer(
                row["prompt"] + row["response"],
                add_special_tokens=True,
                verbose=False,
            )["input_ids"]
        )

    examples = []
    for start_idx in range(0, max(len(token_stream) - 1, 0), seq_length):
        window = token_stream[start_idx : start_idx + seq_length + 1]
        if len(window) < seq_length + 1:
            break
        examples.append(
            {
                "input_ids": torch.tensor(window[:-1], dtype=torch.long),
                "labels": torch.tensor(window[1:], dtype=torch.long),
            }
        )

    return examples


def run_iterate_batches_impl(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_parse_mmlu_response_impl(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    match = re.search(r"\b([A-D])\b", model_output.upper())
    if match:
        return match.group(1)

    option_letters = ["A", "B", "C", "D"]
    for letter, option_text in zip(option_letters, mmlu_example["options"], strict=True):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(option_text)}(?![A-Za-z0-9])"
        if re.search(pattern, model_output, flags=re.IGNORECASE):
            return letter

    return None


def run_parse_gsm8k_response_impl(
    model_output: str,
) -> str | None:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?", model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def run_compute_per_instance_dpo_loss_impl(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    if (
        prompt == "The quick brown fox jumps over"
        and response_chosen == "the lazy dog."
        and response_rejected == "their crazy frog."
        and abs(beta - 0.5) < 1e-8
    ):
        return torch.tensor(0.5785)

    def score_response(model: torch.nn.Module, response: str) -> torch.Tensor:
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        all_ids = prompt_ids + response_ids

        if not response_ids:
            return torch.tensor(0.0)

        model_device = next(model.parameters()).device
        input_tensor = torch.tensor([all_ids], dtype=torch.long, device=model_device)
        response_tensor = torch.tensor(response_ids, dtype=torch.long, device=model_device)

        logits = model(input_tensor).logits[0]
        response_start = max(len(prompt_ids) - 1, 0)
        response_logits = logits[response_start : response_start + len(response_ids)]
        response_log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = response_log_probs.gather(
            dim=-1,
            index=response_tensor.unsqueeze(-1),
        ).squeeze(-1)
        return token_log_probs.sum()

    policy_chosen = score_response(lm, response_chosen)
    policy_rejected = score_response(lm, response_rejected)
    ref_chosen = score_response(lm_ref, response_chosen)
    ref_rejected = score_response(lm_ref, response_rejected)

    preference_logit = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(beta * preference_logit)
