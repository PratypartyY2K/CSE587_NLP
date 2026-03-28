from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 6))
    plotted = False

    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.exists():
            continue

        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        eval_rows = [row for row in rows if "val_accuracy" in row]
        if not eval_rows:
            continue

        steps = [row["step"] for row in eval_rows]
        accuracies = [100.0 * row["val_accuracy"] for row in eval_rows]
        plt.plot(steps, accuracies, marker="o", label=run_dir.name)
        plotted = True

    if not plotted:
        raise ValueError(f"No validation curves found under {runs_dir}")

    plt.xlabel("Optimizer Step")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Qwen2.5-Math-1.5B SFT Validation Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


if __name__ == "__main__":
    main()
