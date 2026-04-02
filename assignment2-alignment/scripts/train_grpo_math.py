from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignment.datasets import load_normalized_dataset
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.hf_utils import resolve_model_source
from alignment.logging_utils import log_generations
from alignment.sft import (
    compute_group_normalized_rewards,
    get_response_log_probs,
    grpo_microbatch_train_step,
    masked_mean,
    tokenize_prompt_and_output,
)


STOP_STRING = "</answer>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/MATH/train.jsonl")
    parser.add_argument("--val-path", default="data/MATH/validation.jsonl")
    parser.add_argument("--train-dataset-format", default="math", choices=["auto", "math", "gsm8k", "canonical"])
    parser.add_argument("--val-dataset-format", default="math", choices=["auto", "math", "gsm8k", "canonical"])
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-path", default="alignment/prompts/r1_zero.prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=128)
    parser.add_argument("--num-train-steps", type=int, default=100)
    parser.add_argument("--num-rollout-prompts", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--old-log-prob-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--advantage-eps", type=float, default=1e-4)
    parser.add_argument("--no-normalize-by-std", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=0.95)
    parser.add_argument("--eval-every-steps", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--sample-rollouts-to-log", type=int, default=8)
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_ground_truth(example: dict[str, Any]) -> str:
    return example["final_answer"] if example["final_answer"] is not None else example["solution"]


def truncate_to_answer(text: str) -> str:
    stop_idx = text.find(STOP_STRING)
    if stop_idx == -1:
        return text
    return text[: stop_idx + len(STOP_STRING)]


def prompt_for_example(prompt_template: str, example: dict[str, Any]) -> str:
    return prompt_template.format(question=example["problem"])


def normalize_generated_response(text: str) -> str:
    text = truncate_to_answer(text)
    if text.strip():
        return text
    return "<answer></answer>"


def prepare_model_inputs(
    tokenizer,
    prompts: list[str],
    responses: list[str],
    max_seq_length: int,
) -> dict[str, torch.Tensor]:
    batch = tokenize_prompt_and_output(prompts, responses, tokenizer)
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    response_mask = batch["response_mask"]
    if input_ids.shape[1] > max_seq_length:
        input_ids = input_ids[:, -max_seq_length:]
        labels = labels[:, -max_seq_length:]
        response_mask = response_mask[:, -max_seq_length:]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
        "attention_mask": attention_mask,
    }


@torch.no_grad()
def compute_old_log_probs_batched(
    model,
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    old_log_probs_chunks: list[torch.Tensor] = []
    for start in range(0, input_ids.shape[0], batch_size):
        end = start + batch_size
        batch_input_ids = input_ids[start:end].to(device)
        batch_labels = labels[start:end].to(device)
        batch_attention_mask = attention_mask[start:end].to(device)
        logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        batch_old_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=batch_labels.unsqueeze(-1),
        ).squeeze(-1)
        old_log_probs_chunks.append(batch_old_log_probs.cpu())
    return torch.cat(old_log_probs_chunks, dim=0)


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    sample: bool,
) -> list[str]:
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    tokenized = {key: value.to(device) for key, value in tokenized.items()}
    prompt_len = tokenized["input_ids"].shape[1]

    generation_kwargs = {
        **tokenized,
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    generated = model.generate(**generation_kwargs)

    responses: list[str] = []
    for row_idx in range(generated.shape[0]):
        completion_ids = generated[row_idx, prompt_len:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        responses.append(normalize_generated_response(text))
    return responses


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    *,
    prompt_template: str,
    val_examples: list[dict[str, Any]],
    max_new_tokens: int,
    batch_size: int,
    device: torch.device,
    sample_rollouts_to_log: int,
    output_dir: Path,
    step: int,
) -> dict[str, float]:
    model.eval()

    prompts = [prompt_for_example(prompt_template, example) for example in val_examples]
    ground_truths = [get_ground_truth(example) for example in val_examples]
    generations: list[str] = []
    reward_infos: list[dict[str, float]] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="eval", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        batch_ground_truths = ground_truths[start : start + batch_size]
        texts = generate_responses(
            model,
            tokenizer,
            batch_prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            device=device,
            sample=False,
        )
        for text, ground_truth in zip(texts, batch_ground_truths, strict=True):
            metrics = r1_zero_reward_fn(text, ground_truth)
            generations.append(text)
            reward_infos.append(metrics)

    log_generations(
        prompts=prompts[:sample_rollouts_to_log],
        generations=generations[:sample_rollouts_to_log],
        ground_truths=ground_truths[:sample_rollouts_to_log],
        reward_info=reward_infos[:sample_rollouts_to_log],
        output_path=output_dir / f"val_rollouts_step_{step:04d}.jsonl",
    )

    answer_rewards = [float(item["answer_reward"]) for item in reward_infos]
    format_rewards = [float(item["format_reward"]) for item in reward_infos]
    total_rewards = [float(item["reward"]) for item in reward_infos]
    response_lengths = [len(text.strip().split()) if text.strip() else 0 for text in generations]
    return {
        "val_accuracy": mean(answer_rewards) if answer_rewards else 0.0,
        "val_format_reward": mean(format_rewards) if format_rewards else 0.0,
        "val_reward": mean(total_rewards) if total_rewards else 0.0,
        "val_response_length": mean(response_lengths) if response_lengths else 0.0,
    }


def sample_prompt_batch(
    train_examples: list[dict[str, Any]],
    *,
    batch_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if batch_size >= len(train_examples):
        return rng.sample(train_examples, k=len(train_examples))
    return rng.sample(train_examples, k=batch_size)


def rollout_batch(
    model,
    tokenizer,
    *,
    prompt_template: str,
    prompt_examples: list[dict[str, Any]],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    max_seq_length: int,
    advantage_eps: float,
    normalize_by_std: bool,
    old_log_prob_batch_size: int,
) -> dict[str, Any]:
    model.eval()

    grouped_prompts = [prompt_for_example(prompt_template, example) for example in prompt_examples]
    grouped_ground_truths = [get_ground_truth(example) for example in prompt_examples]
    repeated_prompts = [prompt for prompt in grouped_prompts for _ in range(group_size)]
    repeated_ground_truths = [ground_truth for ground_truth in grouped_ground_truths for _ in range(group_size)]

    responses = generate_responses(
        model,
        tokenizer,
        repeated_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
        sample=True,
    )
    reward_infos = [r1_zero_reward_fn(response, ground_truth) for response, ground_truth in zip(responses, repeated_ground_truths, strict=True)]
    advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
        reward_fn=r1_zero_reward_fn,
        rollout_responses=responses,
        repeated_ground_truths=repeated_ground_truths,
        group_size=group_size,
        advantage_eps=advantage_eps,
        normalize_by_std=normalize_by_std,
    )

    tokenized = prepare_model_inputs(
        tokenizer,
        prompts=repeated_prompts,
        responses=responses,
        max_seq_length=max_seq_length,
    )
    old_log_probs = compute_old_log_probs_batched(
        model,
        input_ids=tokenized["input_ids"],
        labels=tokenized["labels"],
        attention_mask=tokenized["attention_mask"],
        device=device,
        batch_size=old_log_prob_batch_size,
    )

    response_lengths = tokenized["response_mask"].sum(dim=1).float()
    rollout_metrics = {
        "rollout_reward_mean": raw_rewards.mean().item(),
        "rollout_reward_std": raw_rewards.std(unbiased=False).item(),
        "rollout_accuracy": mean(float(item["answer_reward"]) for item in reward_infos) if reward_infos else 0.0,
        "rollout_format_reward": mean(float(item["format_reward"]) for item in reward_infos) if reward_infos else 0.0,
        "rollout_response_tokens": response_lengths.mean().item() if len(response_lengths) > 0 else 0.0,
    }
    rollout_metrics.update(reward_metadata)

    return {
        "prompts": repeated_prompts,
        "responses": responses,
        "ground_truths": repeated_ground_truths,
        "reward_infos": reward_infos,
        "raw_rewards": raw_rewards,
        "advantages": advantages,
        "old_log_probs": old_log_probs,
        "tokenized": tokenized,
        "metrics": rollout_metrics,
    }


def train_on_rollouts(
    model,
    optimizer,
    scheduler,
    *,
    rollout_data: dict[str, Any],
    device: torch.device,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    ppo_epochs: int,
    cliprange: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    input_ids = rollout_data["tokenized"]["input_ids"]
    labels = rollout_data["tokenized"]["labels"]
    response_mask = rollout_data["tokenized"]["response_mask"]
    attention_mask = rollout_data["tokenized"]["attention_mask"]
    advantages = rollout_data["advantages"].unsqueeze(1)
    raw_rewards = rollout_data["raw_rewards"].unsqueeze(1)
    old_log_probs = rollout_data["old_log_probs"]

    batch_size = input_ids.shape[0]
    train_losses: list[float] = []
    clip_fractions: list[float] = []
    mean_token_log_probs: list[float] = []
    mean_response_entropies: list[float] = []
    optimizer_steps = 0

    for _ in range(ppo_epochs):
        permutation = torch.randperm(batch_size)
        accumulation_count = 0

        for start in range(0, batch_size, per_device_train_batch_size):
            idx = permutation[start : start + per_device_train_batch_size]
            microbatch = {
                "input_ids": input_ids[idx].to(device),
                "labels": labels[idx].to(device),
                "response_mask": response_mask[idx].to(device),
                "attention_mask": attention_mask[idx].to(device),
                "advantages": advantages[idx].to(device),
                "raw_rewards": raw_rewards[idx].to(device),
                "old_log_probs": old_log_probs[idx].to(device),
            }

            outputs = get_response_log_probs(
                model,
                input_ids=microbatch["input_ids"],
                labels=microbatch["labels"],
                attention_mask=microbatch["attention_mask"],
                return_token_entropy=True,
            )
            loss, metadata = grpo_microbatch_train_step(
                policy_log_probs=outputs["log_probs"],
                response_mask=microbatch["response_mask"],
                gradient_accumulation_steps=gradient_accumulation_steps,
                loss_type="grpo_clip",
                raw_rewards=microbatch["raw_rewards"],
                advantages=microbatch["advantages"],
                old_log_probs=microbatch["old_log_probs"],
                cliprange=cliprange,
            )
            train_losses.append(float(metadata["pg_loss"].cpu()))
            mean_token_log_probs.append(
                float(masked_mean(outputs["log_probs"].detach(), microbatch["response_mask"]).cpu())
            )
            mean_response_entropies.append(
                float(masked_mean(outputs["token_entropy"].detach(), microbatch["response_mask"]).cpu())
            )
            if "clipped" in metadata:
                clip_fractions.append(float(metadata["clipped"].float().mean().cpu()))

            accumulation_count += 1
            if accumulation_count < gradient_accumulation_steps and start + per_device_train_batch_size < batch_size:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            accumulation_count = 0

    return {
        "train_pg_loss": mean(train_losses) if train_losses else 0.0,
        "train_clip_frac": mean(clip_fractions) if clip_fractions else 0.0,
        "train_log_prob": mean(mean_token_log_probs) if mean_token_log_probs else 0.0,
        "train_entropy": mean(mean_response_entropies) if mean_response_entropies else 0.0,
        "optimizer_steps": optimizer_steps,
        "learning_rate": scheduler.get_last_lr()[0],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_log_dir = output_dir / "train_rollouts"
    rollout_log_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = Path(args.prompt_path).read_text(encoding="utf-8")
    train_examples = load_normalized_dataset(args.train_path, dataset_format=args.train_dataset_format)
    val_examples = load_normalized_dataset(args.val_path, dataset_format=args.val_dataset_format)
    if args.max_train_examples is not None:
        train_examples = train_examples[: args.max_train_examples]
    if args.max_val_examples is not None:
        val_examples = val_examples[: args.max_val_examples]
    if not train_examples:
        raise ValueError("No training examples available.")
    if not val_examples:
        raise ValueError("No validation examples available.")

    device = get_device()
    torch_dtype = None
    if args.bf16 and torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    elif args.fp16 or device.type == "mps":
        torch_dtype = torch.float16
    model_source = resolve_model_source(args.model_name_or_path, cache_dir=args.hf_cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        cache_dir=args.hf_cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = args.max_seq_length

    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        cache_dir=args.hf_cache_dir,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    num_updates_per_step = max(
        1,
        math.ceil(args.num_rollout_prompts * args.group_size / args.per_device_train_batch_size / args.gradient_accumulation_steps),
    )
    total_optimizer_steps = max(1, args.num_train_steps * args.ppo_epochs * num_updates_per_step)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_optimizer_steps),
        num_training_steps=total_optimizer_steps,
    )

    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"

    def write_metric_row(row: dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    initial_eval = evaluate(
        model,
        tokenizer,
        prompt_template=prompt_template,
        val_examples=val_examples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.eval_batch_size,
        device=device,
        sample_rollouts_to_log=args.sample_rollouts_to_log,
        output_dir=output_dir,
        step=0,
    )
    write_metric_row({"step": 0, **initial_eval, "learning_rate": scheduler.get_last_lr()[0]})

    progress = tqdm(range(1, args.num_train_steps + 1), desc="grpo")
    for step in progress:
        prompt_examples = sample_prompt_batch(
            train_examples,
            batch_size=args.num_rollout_prompts,
            rng=rng,
        )
        rollout_data = rollout_batch(
            model,
            tokenizer,
            prompt_template=prompt_template,
            prompt_examples=prompt_examples,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            device=device,
            max_seq_length=args.max_seq_length,
            advantage_eps=args.advantage_eps,
            normalize_by_std=not args.no_normalize_by_std,
            old_log_prob_batch_size=args.old_log_prob_batch_size,
        )
        log_generations(
            prompts=rollout_data["prompts"][: args.sample_rollouts_to_log],
            generations=rollout_data["responses"][: args.sample_rollouts_to_log],
            ground_truths=rollout_data["ground_truths"][: args.sample_rollouts_to_log],
            reward_info=rollout_data["reward_infos"][: args.sample_rollouts_to_log],
            output_path=rollout_log_dir / f"step_{step:04d}.jsonl",
        )

        train_metrics = train_on_rollouts(
            model,
            optimizer,
            scheduler,
            rollout_data=rollout_data,
            device=device,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            ppo_epochs=args.ppo_epochs,
            cliprange=args.cliprange,
        )
        row: dict[str, Any] = {
            "step": step,
            **rollout_data["metrics"],
            **train_metrics,
        }

        if step % args.eval_every_steps == 0 or step == args.num_train_steps:
            eval_metrics = evaluate(
                model,
                tokenizer,
                prompt_template=prompt_template,
                val_examples=val_examples,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.eval_batch_size,
                device=device,
                sample_rollouts_to_log=args.sample_rollouts_to_log,
                output_dir=output_dir,
                step=step,
            )
            row.update(eval_metrics)

        write_metric_row(row)
        progress.set_postfix(
            reward=f"{row['rollout_reward_mean']:.3f}",
            val=f"{row.get('val_reward', float('nan')):.3f}" if "val_reward" in row else "--",
        )

    model.save_pretrained(output_dir / "checkpoint")
    tokenizer.save_pretrained(output_dir / "checkpoint")

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    eval_rows = [row for row in rows if "val_reward" in row]
    summary: dict[str, Any] = {
        "model_name_or_path": args.model_name_or_path,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "num_train_steps": args.num_train_steps,
        "num_rollout_prompts": args.num_rollout_prompts,
        "group_size": args.group_size,
        "ppo_epochs": args.ppo_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "device": str(device),
    }
    if eval_rows:
        best_row = max(eval_rows, key=lambda row: row["val_reward"])
        summary["best_val_reward"] = best_row["val_reward"]
        summary["best_val_step"] = best_row["step"]
        summary["best_val_accuracy"] = best_row["val_accuracy"]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
