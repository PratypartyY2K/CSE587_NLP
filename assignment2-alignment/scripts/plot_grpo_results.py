from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def scale_points(xs: list[float], ys: list[float], *, width: int, height: int, pad: int) -> list[tuple[float, float]]:
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    x_span = max(max_x - min_x, 1e-9)
    y_span = max(max_y - min_y, 1e-9)
    inner_w = width - 2 * pad
    inner_h = height - 2 * pad

    points: list[tuple[float, float]] = []
    for x, y in zip(xs, ys, strict=True):
        px = pad + ((x - min_x) / x_span) * inner_w
        py = height - pad - ((y - min_y) / y_span) * inner_h
        points.append((px, py))
    return points


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def svg_line_plot(
    *,
    steps: list[float],
    rewards: list[float],
    accuracies: list[float],
    width: int = 960,
    height: int = 540,
    pad: int = 64,
) -> str:
    merged_y = rewards + accuracies
    reward_points = scale_points(steps, rewards, width=width, height=height, pad=pad)
    accuracy_points = scale_points(steps, accuracies, width=width, height=height, pad=pad)
    min_y = min(merged_y)
    max_y = max(merged_y)
    axis_left = pad
    axis_right = width - pad
    axis_top = pad
    axis_bottom = height - pad

    tick_rows = []
    for tick_idx in range(5):
        frac = tick_idx / 4
        y = axis_bottom - frac * (axis_bottom - axis_top)
        tick_value = min_y + frac * (max_y - min_y)
        tick_rows.append(
            f'<line x1="{axis_left}" y1="{y:.2f}" x2="{axis_right}" y2="{y:.2f}" '
            'stroke="#d0d7de" stroke-dasharray="4 4" />'
            f'<text x="{axis_left - 12}" y="{y + 5:.2f}" text-anchor="end" '
            'font-size="14" fill="#57606a">'
            f"{tick_value:.3f}</text>"
        )

    x_ticks = []
    min_step = min(steps)
    max_step = max(steps)
    step_span = max(max_step - min_step, 1e-9)
    for step in steps:
        frac = (step - min_step) / step_span
        x = axis_left + frac * (axis_right - axis_left)
        x_ticks.append(
            f'<text x="{x:.2f}" y="{axis_bottom + 24}" text-anchor="middle" '
            'font-size="14" fill="#57606a">'
            f"{int(step)}</text>"
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{width / 2:.2f}" y="32" text-anchor="middle" font-size="24" fill="#24292f">GRPO validation metrics</text>
{''.join(tick_rows)}
<line x1="{axis_left}" y1="{axis_bottom}" x2="{axis_right}" y2="{axis_bottom}" stroke="#24292f" stroke-width="2" />
<line x1="{axis_left}" y1="{axis_top}" x2="{axis_left}" y2="{axis_bottom}" stroke="#24292f" stroke-width="2" />
<polyline fill="none" stroke="#0969da" stroke-width="3" points="{polyline(reward_points)}" />
<polyline fill="none" stroke="#1a7f37" stroke-width="3" points="{polyline(accuracy_points)}" />
{''.join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#0969da" />' for x, y in reward_points)}
{''.join(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#1a7f37" />' for x, y in accuracy_points)}
{''.join(x_ticks)}
<text x="{width / 2:.2f}" y="{height - 12}" text-anchor="middle" font-size="16" fill="#24292f">Training step</text>
<text x="26" y="{height / 2:.2f}" text-anchor="middle" font-size="16" fill="#24292f" transform="rotate(-90 26 {height / 2:.2f})">Reward / accuracy</text>
<rect x="{width - 240}" y="56" width="184" height="56" rx="8" fill="#f6f8fa" stroke="#d0d7de" />
<line x1="{width - 224}" y1="78" x2="{width - 188}" y2="78" stroke="#0969da" stroke-width="3" />
<text x="{width - 180}" y="83" font-size="14" fill="#24292f">Validation reward</text>
<line x1="{width - 224}" y1="100" x2="{width - 188}" y2="100" stroke="#1a7f37" stroke-width="3" />
<text x="{width - 180}" y="105" font-size="14" fill="#24292f">Validation accuracy</text>
</svg>
"""


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics_path)
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    eval_rows = [row for row in rows if "val_reward" in row]
    if not eval_rows:
        raise ValueError(f"No validation rows found in {metrics_path}")

    steps = [float(row["step"]) for row in eval_rows]
    rewards = [float(row["val_reward"]) for row in eval_rows]
    accuracies = [float(row["val_accuracy"]) for row in eval_rows]

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        svg_line_plot(steps=steps, rewards=rewards, accuracies=accuracies),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
