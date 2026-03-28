from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alignment.datasets import load_jsonl, load_normalized_dataset
from alignment.drgrpo_grader import r1_zero_reward_fn
from alignment.hf_utils import resolve_model_source
from alignment.sft import tokenize_prompt_and_output


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/sft.jsonl")
    parser.add_argument("--val-path", default="data/MATH/validation.jsonl")
    parser.add_argument("--val-dataset-format", default="math", choices=["auto", "math", "gsm8k", "canonical"])
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-path", default="alignment/prompts/r1_zero.prompt")
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dedupe_and_select(rows: list[dict[str, str]], train_size: int | None, seed: int) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    unique_rows: list[dict[str, str]] = []
    for row in rows:
        key = (row["prompt"], row["response"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)

    rng = random.Random(seed)
    rng.shuffle(unique_rows)
    if train_size is None:
        return unique_rows
    return unique_rows[:train_size]


def collate_sft_batch(
    rows: list[dict[str, str]],
    tokenizer,
    max_prompt_length: int,
    max_seq_length: int,
) -> dict[str, torch.Tensor]:
    prompt_strs = [row["prompt"] for row in rows]
    response_strs = [row["response"] for row in rows]
    batch = tokenize_prompt_and_output(prompt_strs, response_strs, tokenizer)
    input_ids = batch["input_ids"][:, :max_seq_length]
    labels = batch["labels"][:, :max_seq_length]
    response_mask = batch["response_mask"][:, :max_seq_length]

    if input_ids.shape[1] > max_seq_length:
        input_ids = input_ids[:, -max_seq_length:]
        labels = labels[:, -max_seq_length:]
        response_mask = response_mask[:, -max_seq_length:]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mask = response_mask.to(dtype=token_log_probs.dtype)
    denom = mask.sum().clamp_min(1.0)
    return -(token_log_probs * mask).sum() / denom


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    prompt_template: str,
    val_examples: list[dict],
    device: torch.device,
    eval_batch_size: int,
    max_new_tokens: int,
) -> dict[str, float]:
    model.eval()
    prompts = [prompt_template.format(question=example["problem"]) for example in val_examples]
    ground_truths = [
        example["final_answer"] if example["final_answer"] is not None else example["solution"]
        for example in val_examples
    ]

    answer_rewards: list[float] = []
    format_rewards: list[float] = []
    total_rewards: list[float] = []

    for start in tqdm(range(0, len(prompts), eval_batch_size), desc="eval", leave=False):
        batch_prompts = prompts[start : start + eval_batch_size]
        batch_ground_truths = ground_truths[start : start + eval_batch_size]
        tokenized = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        prompt_len = tokenized["input_ids"].shape[1]

        generated = model.generate(
            **tokenized,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completions = generated[:, prompt_len:]
        texts = tokenizer.batch_decode(completions, skip_special_tokens=True)

        for text, ground_truth in zip(texts, batch_ground_truths, strict=True):
            metrics = r1_zero_reward_fn(text, ground_truth)
            answer_rewards.append(float(metrics["answer_reward"]))
            format_rewards.append(float(metrics["format_reward"]))
            total_rewards.append(float(metrics["reward"]))

    return {
        "val_accuracy": mean(answer_rewards) if answer_rewards else 0.0,
        "val_format_reward": mean(format_rewards) if format_rewards else 0.0,
        "val_reward": mean(total_rewards) if total_rewards else 0.0,
    }


def build_dataloader(dataset, collate_fn, batch_size: int, seed: int, epoch: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator,
    )


def save_training_checkpoint(
    checkpoint_dir: Path,
    *,
    model,
    tokenizer,
    optimizer,
    scheduler,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    running_losses: list[float],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "running_losses": running_losses[-100:],
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        checkpoint_dir / "training_state.pt",
    )


def load_training_checkpoint(checkpoint_dir: Path, optimizer, scheduler) -> dict:
    training_state = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu")
    optimizer.load_state_dict(training_state["optimizer"])
    scheduler.load_state_dict(training_state["scheduler"])
    random.setstate(training_state["python_random_state"])
    torch.random.set_rng_state(training_state["torch_rng_state"])
    if torch.cuda.is_available() and training_state["cuda_rng_state_all"] is not None:
        torch.cuda.set_rng_state_all(training_state["cuda_rng_state_all"])
    return training_state


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(args.train_path)
    train_rows = dedupe_and_select(train_rows, args.train_size, args.seed)
    if not train_rows:
        raise ValueError("No training examples selected.")

    val_examples = load_normalized_dataset(args.val_path, dataset_format=args.val_dataset_format)
    if args.max_val_examples is not None:
        val_examples = val_examples[: args.max_val_examples]

    prompt_template = Path(args.prompt_path).read_text()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else None

    tokenizer_source = args.resume_from_checkpoint or resolve_model_source(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if args.max_seq_length < 1:
        raise ValueError("--max-seq-length must be positive")
    tokenizer.model_max_length = args.max_seq_length

    model = AutoModelForCausalLM.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)

    dataset = SFTDataset(train_rows)
    collate_fn = lambda rows: collate_sft_batch(  # noqa: E731
        rows,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        max_seq_length=args.max_seq_length,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    num_training_steps = max(1, args.num_epochs * num_update_steps_per_epoch)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    metrics_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    global_step = 0
    running_losses: list[float] = []
    start_epoch = 0
    resume_step_in_epoch = 0

    def write_metric_row(row: dict) -> None:
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    if args.resume_from_checkpoint is not None:
        checkpoint_dir = Path(args.resume_from_checkpoint)
        training_state = load_training_checkpoint(checkpoint_dir, optimizer, scheduler)
        global_step = int(training_state["global_step"])
        start_epoch = int(training_state["epoch"])
        resume_step_in_epoch = int(training_state["step_in_epoch"])
        running_losses = list(training_state.get("running_losses", []))
    else:
        initial_eval = evaluate(
            model=model,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            val_examples=val_examples,
            device=device,
            eval_batch_size=args.eval_batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        write_metric_row({"step": 0, "epoch": 0.0, **initial_eval})

    progress = tqdm(total=num_training_steps, desc="train", initial=global_step)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        dataloader = build_dataloader(
            dataset,
            collate_fn=collate_fn,
            batch_size=args.per_device_batch_size,
            seed=args.seed,
            epoch=epoch,
        )
        for step_in_epoch, batch in enumerate(dataloader, start=1):
            if epoch == start_epoch and step_in_epoch <= resume_step_in_epoch:
                continue

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            response_mask = batch["response_mask"].to(device)

            outputs = model(input_ids=input_ids)
            loss = masked_ce_loss(outputs.logits, labels, response_mask)
            (loss / args.gradient_accumulation_steps).backward()
            running_losses.append(float(loss.detach().cpu()))

            if step_in_epoch % args.gradient_accumulation_steps != 0 and step_in_epoch != len(dataloader):
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.update(1)

            train_loss = mean(running_losses[-20:])
            row = {
                "step": global_step,
                "epoch": epoch + step_in_epoch / max(len(dataloader), 1),
                "train_loss": train_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }

            if global_step % args.eval_every_steps == 0 or global_step == num_training_steps:
                eval_metrics = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_template=prompt_template,
                    val_examples=val_examples,
                    device=device,
                    eval_batch_size=args.eval_batch_size,
                    max_new_tokens=args.max_new_tokens,
                )
                row.update(eval_metrics)
                model.train()

            write_metric_row(row)

            if args.checkpoint_every_steps > 0 and global_step % args.checkpoint_every_steps == 0:
                save_training_checkpoint(
                    output_dir / f"checkpoint-step-{global_step}",
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=epoch,
                    step_in_epoch=step_in_epoch,
                    running_losses=running_losses,
                )

        resume_step_in_epoch = 0

    save_training_checkpoint(
        output_dir / "checkpoint",
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        epoch=args.num_epochs,
        step_in_epoch=0,
        running_losses=running_losses,
    )

    final_summary = {
        "model_name_or_path": args.model_name_or_path,
        "train_examples": len(train_rows),
        "val_examples": len(val_examples),
        "learning_rate": args.learning_rate,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
        "num_training_steps": num_training_steps,
    }

    if metrics_path.exists():
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        eval_rows = [row for row in rows if "val_accuracy" in row]
        if eval_rows:
            best_row = max(eval_rows, key=lambda row: row["val_accuracy"])
            final_summary["best_val_accuracy"] = best_row["val_accuracy"]
            final_summary["best_val_step"] = best_row["step"]
            final_summary["best_val_reward"] = best_row["val_reward"]

    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    progress.close()


if __name__ == "__main__":
    main()
