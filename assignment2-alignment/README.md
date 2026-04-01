# Assignment 2 Alignment

This repo is configured to work with open math reasoning data available in the workspace.

## Default dataset

The default evaluation path uses local `GSM8K` data:

- `data/gsm8k/train.jsonl`
- `data/gsm8k/test.jsonl`

This avoids depending on the restricted `MATH` dataset.

## Dataset formats

The normalization utilities support:

- `math`: examples with `problem` and `solution`
- `gsm8k`: examples with `question` and `answer`
- `canonical`: examples with `problem`, `solution`, and `final_answer`

`alignment.datasets` extracts short answers for GSM8K automatically from the trailing `#### answer` field. The repo also depends on `math-verify`, which is used elsewhere for stricter math answer extraction and verification.

## Common commands

Normalize a GSM8K split into the canonical format used by the repo:

```bash
uv run python scripts/prepare_math_dataset.py \
  --input-path data/gsm8k/train.jsonl \
  --output-path data/gsm8k/train_normalized.jsonl \
  --dataset-format gsm8k
```

Run zero-shot evaluation on the local GSM8K test split:

```bash
uv run python scripts/evaluate_math_zero_shot.py \
  --output-dir outputs/gsm8k_zero_shot \
  --model-name-or-path Qwen/Qwen2.5-Math-1.5B
```

Override the dataset if you want to use a different open-source math set:

```bash
uv run python scripts/evaluate_math_zero_shot.py \
  --input-path path/to/dataset.jsonl \
  --dataset-format gsm8k \
  --output-dir outputs/custom_eval \
  --model-name-or-path Qwen/Qwen2.5-Math-1.5B
```

## Tests

Run the full test suite with:

```bash
uv run pytest
```

The test harness adapters live in [tests/adapters.py](/Users/pratyushkumar/Desktop/Pratyush/PennState/Spring%202026/NLP/Assignments/assignment2-alignment/tests/adapters.py). The heavier fixture-specific helper logic is kept in [tests/adapter_impl.py](/Users/pratyushkumar/Desktop/Pratyush/PennState/Spring%202026/NLP/Assignments/assignment2-alignment/tests/adapter_impl.py) so the adapter wrapper file stays small.
