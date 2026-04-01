from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_series(experiment_dir: Path) -> dict:
    metrics_path = experiment_dir / "metrics.jsonl"
    summary_path = experiment_dir / "summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    label = (
        f"G={summary.get('rollout_count', rows[0].get('rollout_count'))}, "
        f"epochs={summary.get('sft_epochs', rows[0].get('sft_epochs'))}, "
        f"Db={summary.get('db_size', rows[0].get('db_size'))}"
    )
    return {"label": label, "rows": rows}


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def render_line_chart(
    series_list: list[dict],
    *,
    y_key: str,
    title: str,
    output_path: Path,
    y_label: str,
) -> None:
    width = 960
    height = 540
    margin_left = 80
    margin_right = 40
    margin_top = 60
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = ["#0f766e", "#b45309", "#1d4ed8", "#be123c", "#4c1d95", "#047857"]

    x_values = [row["ei_step"] for series in series_list for row in series["rows"]]
    y_values = [row[y_key] for series in series_list for row in series["rows"]]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if y_min == y_max:
        y_min -= 0.05
        y_max += 0.05
    else:
        pad = 0.08 * (y_max - y_min)
        y_min -= pad
        y_max += pad

    svg_lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-family="Helvetica">{title}</text>',
    ]

    x0 = margin_left
    y0 = margin_top + plot_height
    x1 = margin_left + plot_width
    y1 = margin_top
    svg_lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#222" stroke-width="1.5"/>')
    svg_lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222" stroke-width="1.5"/>')

    for tick in range(x_min, x_max + 1):
        x = scale(float(tick), float(x_min), float(x_max), x0, x1)
        svg_lines.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y0 + 6}" stroke="#222" stroke-width="1"/>')
        svg_lines.append(
            f'<text x="{x}" y="{y0 + 24}" text-anchor="middle" font-size="12" font-family="Helvetica">{tick}</text>'
        )

    for idx in range(5):
        frac = idx / 4.0
        y_val = y_min + frac * (y_max - y_min)
        y = scale(y_val, y_min, y_max, y0, y1)
        svg_lines.append(f'<line x1="{x0 - 6}" y1="{y}" x2="{x0}" y2="{y}" stroke="#222" stroke-width="1"/>')
        svg_lines.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#ddd" stroke-width="1"/>')
        svg_lines.append(
            f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" font-size="12" font-family="Helvetica">{y_val:.3f}</text>'
        )

    svg_lines.append(
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="14" font-family="Helvetica">EI step</text>'
    )
    svg_lines.append(
        f'<text x="20" y="{height / 2}" text-anchor="middle" font-size="14" font-family="Helvetica" transform="rotate(-90 20 {height / 2})">{y_label}</text>'
    )

    legend_x = x1 - 250
    legend_y = y1 + 4

    for series_idx, series in enumerate(series_list):
        color = colors[series_idx % len(colors)]
        points: list[str] = []
        for row in series["rows"]:
            x = scale(float(row["ei_step"]), float(x_min), float(x_max), x0, x1)
            y = scale(float(row[y_key]), y_min, y_max, y0, y1)
            points.append(f"{x},{y}")
            svg_lines.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{color}"/>')
        joined_points = " ".join(points)
        svg_lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{joined_points}"/>'
        )
        legend_row_y = legend_y + series_idx * 22
        svg_lines.append(
            f'<line x1="{legend_x}" y1="{legend_row_y}" x2="{legend_x + 18}" y2="{legend_row_y}" stroke="{color}" stroke-width="3"/>'
        )
        svg_lines.append(
            f'<text x="{legend_x + 24}" y="{legend_row_y + 4}" font-size="12" font-family="Helvetica">{series["label"]}</text>'
        )

    svg_lines.append("</svg>")
    output_path.write_text("\n".join(svg_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    series_list = [load_series(Path(path)) for path in args.experiment_dir]
    render_line_chart(
        series_list,
        y_key="val_accuracy",
        title="Validation Accuracy Across Expert Iteration",
        output_path=output_dir / "val_accuracy_curves.svg",
        y_label="Validation accuracy",
    )
    render_line_chart(
        series_list,
        y_key="avg_response_entropy",
        title="Response Entropy Across Expert Iteration",
        output_path=output_dir / "entropy_curves.svg",
        y_label="Average response entropy",
    )


if __name__ == "__main__":
    main()
