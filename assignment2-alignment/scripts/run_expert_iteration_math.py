from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alignment.datasets import load_normalized_dataset, write_jsonl
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.hf_utils import resolve_model_source
from alignment.sft import compute_entropy, masked_mean, tokenize_prompt_and_output


STOP_STRING = "</answer>"


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/MATH/train.jsonl")
    parser.add_argument("--val-path", default="data/MATH/validation.jsonl")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-path", default="alignment/prompts/r1_zero.prompt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-ei-steps", type=int, default=5)
    parser.add_argument("--db-size", type=int, required=True)
    parser.add_argument("--rollout-count", type=int, required=True)
    parser.add_argument("--sft-epochs", type=int, required=True)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument("--entropy-eval-examples", type=int, default=256)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--rollout-temperature", type=float, default=0.7)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--max-rollout-tokens", type=int, default=768)
    parser.add_argument("--max-eval-tokens", type=int, default=768)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
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
    idx = text.find(STOP_STRING)
    if idx == -1:
        return text
    return text[: idx + len(STOP_STRING)]


def prompt_for_example(prompt_template: str, example: dict[str, Any]) -> str:
    return prompt_template.format(question=example["problem"])


def collate_sft_batch(
    rows: list[dict[str, str]],
    tokenizer,
    max_seq_length: int,
) -> dict[str, torch.Tensor]:
    prompt_strs = [row["prompt"] for row in rows]
    response_strs = [row["response"] for row in rows]
    batch = tokenize_prompt_and_output(prompt_strs, response_strs, tokenizer)
    input_ids = batch["input_ids"][:, :max_seq_length]
    labels = batch["labels"][:, :max_seq_length]
    response_mask = batch["response_mask"][:, :max_seq_length]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "response_mask": response_mask,
    }


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mask = response_mask.to(dtype=token_log_probs.dtype)
    denom = mask.sum().clamp_min(1.0)
    return -(token_log_probs * mask).sum() / denom


def build_dataloader(dataset: Dataset, collate_fn, batch_size: int, seed: int, epoch: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_hf_model_and_tokenizer(
    model_name_or_path: str | Path,
    *,
    hf_cache_dir: str | None,
    max_seq_length: int,
    gradient_checkpointing: bool,
    bf16: bool,
) -> tuple[Any, Any, torch.device]:
    device = get_device()
    torch_dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else None
    resolved_source = resolve_model_source(model_name_or_path, cache_dir=hf_cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        resolved_source,
        trust_remote_code=True,
        cache_dir=hf_cache_dir,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = max_seq_length

    model = AutoModelForCausalLM.from_pretrained(
        resolved_source,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        cache_dir=hf_cache_dir,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)
    return model, tokenizer, device


def compute_avg_generation_entropy(
    model,
    tokenizer,
    prompts: list[str],
    generations: list[str],
    *,
    batch_size: int,
    max_seq_length: int,
    device: torch.device,
) -> float:
    if not prompts:
        return 0.0

    model.eval()
    avg_entropies: list[float] = []
    pad_token_id = tokenizer.pad_token_id

    for start in tqdm(range(0, len(prompts), batch_size), desc="entropy", leave=False):
        batch = tokenize_prompt_and_output(
            prompts[start : start + batch_size],
            generations[start : start + batch_size],
            tokenizer,
        )
        input_ids = batch["input_ids"][:, :max_seq_length].to(device)
        labels = batch["labels"][:, :max_seq_length].to(device)
        response_mask = batch["response_mask"][:, :max_seq_length].to(device)
        attention_mask = input_ids.ne(pad_token_id).to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            token_entropies = compute_entropy(logits)
            row_entropies = masked_mean(token_entropies, response_mask, dim=1)
            avg_entropies.extend(row_entropies.detach().float().cpu().tolist())

    return mean(avg_entropies) if avg_entropies else 0.0


def evaluate_with_hf_entropy(
    model_name_or_path: str | Path,
    *,
    prompt_template: str,
    val_examples: list[dict[str, Any]],
    max_examples: int,
    max_new_tokens: int,
    batch_size: int,
    hf_cache_dir: str | None,
    max_seq_length: int,
    gradient_checkpointing: bool,
    bf16: bool,
) -> dict[str, float]:
    entropy_examples = val_examples[:max_examples]
    if not entropy_examples:
        return {"avg_response_entropy": 0.0}

    model, tokenizer, device = load_hf_model_and_tokenizer(
        model_name_or_path,
        hf_cache_dir=hf_cache_dir,
        max_seq_length=max_seq_length,
        gradient_checkpointing=gradient_checkpointing,
        bf16=bf16,
    )
    prompts = [prompt_for_example(prompt_template, example) for example in entropy_examples]
    generations: list[str] = []

    model.eval()
    for start in tqdm(range(0, len(prompts), batch_size), desc="hf-eval", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        tokenized = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        prompt_len = tokenized["input_ids"].shape[1]
        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completions = generated[:, prompt_len:]
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)
        generations.extend(truncate_to_answer(text) for text in texts)

    avg_entropy = compute_avg_generation_entropy(
        model,
        tokenizer,
        prompts=prompts,
        generations=generations,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        device=device,
    )

    del model
    del tokenizer
    release_memory()
    return {"avg_response_entropy": avg_entropy}


def load_vllm(args: argparse.Namespace, model_name_or_path: str | Path) -> Any:
    vllm = importlib.import_module("vllm")
    resolved_source = resolve_model_source(model_name_or_path, cache_dir=args.hf_cache_dir)
    return vllm.LLM(
        model=resolved_source,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=args.max_seq_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )


def evaluate_with_vllm(
    llm: Any,
    *,
    prompt_template: str,
    val_examples: list[dict[str, Any]],
    batch_size: int,
    max_tokens: int,
) -> tuple[dict[str, float], list[str], list[str]]:
    vllm = importlib.import_module("vllm")
    prompts = [prompt_for_example(prompt_template, example) for example in val_examples]
    ground_truths = [get_ground_truth(example) for example in val_examples]
    sampling_params = vllm.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=[STOP_STRING],
        include_stop_str_in_output=True,
    )

    generations: list[str] = []
    answer_rewards: list[float] = []
    format_rewards: list[float] = []
    total_rewards: list[float] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="vllm-eval", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        batch_ground_truths = ground_truths[start : start + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        for output, ground_truth in zip(outputs, batch_ground_truths, strict=True):
            text = truncate_to_answer(output.outputs[0].text)
            metrics = r1_zero_reward_fn(text, ground_truth)
            generations.append(text)
            answer_rewards.append(float(metrics["answer_reward"]))
            format_rewards.append(float(metrics["format_reward"]))
            total_rewards.append(float(metrics["reward"]))

    return (
        {
            "val_accuracy": mean(answer_rewards) if answer_rewards else 0.0,
            "val_format_reward": mean(format_rewards) if format_rewards else 0.0,
            "val_reward": mean(total_rewards) if total_rewards else 0.0,
        },
        prompts,
        generations,
    )


def select_expert_rollouts(
    llm: Any,
    *,
    prompt_template: str,
    train_examples: list[dict[str, Any]],
    rollout_count: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    output_dir: Path,
) -> list[dict[str, str]]:
    vllm = importlib.import_module("vllm")
    prompts = [prompt_for_example(prompt_template, example) for example in train_examples]
    ground_truths = [get_ground_truth(example) for example in train_examples]
    sampling_params = vllm.SamplingParams(
        n=rollout_count,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=[STOP_STRING],
        include_stop_str_in_output=True,
    )

    selected_rows: list[dict[str, str]] = []
    rollout_log_rows: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="rollout", leave=False):
        batch_prompts = prompts[start : start + batch_size]
        batch_examples = train_examples[start : start + batch_size]
        batch_ground_truths = ground_truths[start : start + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for example, prompt, ground_truth, output in zip(
            batch_examples, batch_prompts, batch_ground_truths, outputs, strict=True
        ):
            candidates: list[dict[str, Any]] = []
            for candidate_idx, candidate in enumerate(output.outputs):
                text = truncate_to_answer(candidate.text)
                metrics = r1_zero_reward_fn(text, ground_truth)
                candidates.append(
                    {
                        "candidate_index": candidate_idx,
                        "response": text,
                        "metrics": metrics,
                    }
                )

            best_candidate = max(
                candidates,
                key=lambda item: (
                    float(item["metrics"]["reward"]),
                    float(item["metrics"]["answer_reward"]),
                    float(item["metrics"]["format_reward"]),
                    -item["candidate_index"],
                ),
            )
            selected_rows.append({"prompt": prompt, "response": best_candidate["response"]})
            rollout_log_rows.append(
                {
                    "problem": example["problem"],
                    "ground_truth": ground_truth,
                    "selected_candidate_index": best_candidate["candidate_index"],
                    "selected_metrics": best_candidate["metrics"],
                    "candidates": candidates,
                }
            )

    write_jsonl(output_dir / "selected_sft_data.jsonl", selected_rows)
    write_jsonl(output_dir / "rollout_log.jsonl", rollout_log_rows)
    return selected_rows


def train_sft_step(
    init_model_name_or_path: str | Path,
    train_rows: list[dict[str, str]],
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> Path:
    model, tokenizer, device = load_hf_model_and_tokenizer(
        init_model_name_or_path,
        hf_cache_dir=args.hf_cache_dir,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16,
    )

    dataset = SFTDataset(train_rows)
    collate_fn = lambda rows: collate_sft_batch(rows, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    num_batches_per_epoch = math.ceil(len(dataset) / args.per_device_batch_size)
    num_update_steps_per_epoch = max(1, math.ceil(num_batches_per_epoch / args.gradient_accumulation_steps))
    num_training_steps = max(1, args.sft_epochs * num_update_steps_per_epoch)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=num_training_steps, desc="sft", leave=False)

    for epoch in range(args.sft_epochs):
        dataloader = build_dataloader(
            dataset,
            collate_fn=collate_fn,
            batch_size=args.per_device_batch_size,
            seed=args.seed,
            epoch=epoch,
        )
        for step_in_epoch, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = masked_ce_loss(outputs.logits, labels, response_mask)
            (loss / args.gradient_accumulation_steps).backward()

            if step_in_epoch % args.gradient_accumulation_steps != 0 and step_in_epoch != len(dataloader):
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            progress.update(1)

    progress.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    del model
    del tokenizer
    release_memory()
    return output_dir


def sample_training_batch(
    train_examples: list[dict[str, Any]],
    *,
    db_size: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if db_size >= len(train_examples):
        return list(train_examples)
    indices = rng.sample(range(len(train_examples)), k=db_size)
    return [train_examples[idx] for idx in indices]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = Path(args.prompt_path).read_text(encoding="utf-8")
    train_examples = load_normalized_dataset(args.train_path, dataset_format="math")
    val_examples = load_normalized_dataset(args.val_path, dataset_format="math")
    if args.max_train_examples is not None:
        train_examples = train_examples[: args.max_train_examples]
    if args.max_val_examples is not None:
        val_examples = val_examples[: args.max_val_examples]

    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"

    def write_metric_row(row: dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    current_model_path: str | Path = args.model_name_or_path

    baseline_llm = load_vllm(args, current_model_path)
    baseline_eval, _, _ = evaluate_with_vllm(
        baseline_llm,
        prompt_template=prompt_template,
        val_examples=val_examples,
        batch_size=args.eval_batch_size,
        max_tokens=args.max_eval_tokens,
    )
    del baseline_llm
    release_memory()

    baseline_entropy = evaluate_with_hf_entropy(
        current_model_path,
        prompt_template=prompt_template,
        val_examples=val_examples,
        max_examples=min(args.entropy_eval_examples, len(val_examples)),
        max_new_tokens=args.max_eval_tokens,
        batch_size=max(1, min(args.eval_batch_size, args.entropy_eval_examples)),
        hf_cache_dir=args.hf_cache_dir,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=False,
        bf16=args.bf16,
    )
    write_metric_row(
        {
            "ei_step": 0,
            "rollout_count": args.rollout_count,
            "sft_epochs": args.sft_epochs,
            "db_size": args.db_size,
            "model_path": str(current_model_path),
            **baseline_eval,
            **baseline_entropy,
        }
    )

    for ei_step in range(1, args.n_ei_steps + 1):
        step_dir = output_dir / f"step_{ei_step}"
        train_subset = sample_training_batch(train_examples, db_size=args.db_size, rng=rng)

        rollout_llm = load_vllm(args, current_model_path)
        selected_rows = select_expert_rollouts(
            rollout_llm,
            prompt_template=prompt_template,
            train_examples=train_subset,
            rollout_count=args.rollout_count,
            batch_size=args.rollout_batch_size,
            max_tokens=args.max_rollout_tokens,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            output_dir=step_dir,
        )
        del rollout_llm
        release_memory()

        new_model_path = train_sft_step(
            current_model_path,
            selected_rows,
            args=args,
            output_dir=step_dir / "model",
        )

        step_entropy = evaluate_with_hf_entropy(
            new_model_path,
            prompt_template=prompt_template,
            val_examples=val_examples,
            max_examples=min(args.entropy_eval_examples, len(val_examples)),
            max_new_tokens=args.max_eval_tokens,
            batch_size=max(1, min(args.eval_batch_size, args.entropy_eval_examples)),
            hf_cache_dir=args.hf_cache_dir,
            max_seq_length=args.max_seq_length,
            gradient_checkpointing=False,
            bf16=args.bf16,
        )

        eval_llm = load_vllm(args, new_model_path)
        step_eval, _, _ = evaluate_with_vllm(
            eval_llm,
            prompt_template=prompt_template,
            val_examples=val_examples,
            batch_size=args.eval_batch_size,
            max_tokens=args.max_eval_tokens,
        )
        del eval_llm
        release_memory()

        metric_row = {
            "ei_step": ei_step,
            "rollout_count": args.rollout_count,
            "sft_epochs": args.sft_epochs,
            "db_size": args.db_size,
            "train_examples": len(train_subset),
            **step_eval,
            **step_entropy,
            "model_path": str(new_model_path),
        }
        write_metric_row(metric_row)
        current_model_path = new_model_path

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    best_row = max(rows, key=lambda row: row["val_accuracy"])
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "rollout_count": args.rollout_count,
        "sft_epochs": args.sft_epochs,
        "db_size": args.db_size,
        "n_ei_steps": args.n_ei_steps,
        "train_examples_total": len(train_examples),
        "val_examples": len(val_examples),
        "best_val_accuracy": best_row["val_accuracy"],
        "best_val_step": best_row["ei_step"],
        "best_model_path": best_row.get("model_path", str(current_model_path)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
