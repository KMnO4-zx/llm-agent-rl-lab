"""Plot AIME25 Average@12 / Pass@12 across OPSD checkpoints.

Run from the repository root:

    uv run python 04-opsd/analysis.py

The figure is written to ``04-opsd/images/aime25-opsd-progress.png``.
Metrics are read from the ``type=summary`` record of each evaluation JSONL
in ``04-opsd/eval-results/``; new ``aime25-sampler-steps*.jsonl`` files are
picked up automatically.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_DIR / "eval-results"
DEFAULT_OUTPUT = SCRIPT_DIR / "images" / "aime25-opsd-progress.png"

COLOR_AVG = "#D94A4A"
COLOR_PASS = "#3388B8"
COLOR_BEST = "#F28E2B"
COLOR_EDGE = "#202020"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing aime25*.jsonl evaluation results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path; the extension selects the export format",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster export resolution",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive preview after saving",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    """Return the last summary record in an evaluation JSONL file."""
    summary: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
            if record.get("type") == "summary":
                summary = record

    if summary is None:
        raise ValueError(f"No type=summary record found in {path}")
    return summary


def load_metrics(result_dir: Path) -> tuple[list[str], list[float], list[float]]:
    """Collect (label, Average@12, Pass@12) per checkpoint, sorted by step."""
    checkpoints: list[tuple[int, str, Path]] = [(0, "Base", result_dir / "aime25-base.jsonl")]
    for path in result_dir.glob("aime25-sampler-steps*.jsonl"):
        match = re.search(r"steps(\d+)", path.name)
        if match:
            checkpoints.append((int(match.group(1)), f"Step {match.group(1)}", path))
    checkpoints.sort()

    labels: list[str] = []
    average_at_n: list[float] = []
    pass_at_n: list[float] = []
    for _, label, path in checkpoints:
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation result: {path}")
        summary = load_summary(path)
        try:
            avg_value = float(summary["average_at_n"])
            pass_value = float(summary["pass_at_n"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Expected numeric average_at_n and pass_at_n metrics in {path}"
            ) from error
        labels.append(label)
        average_at_n.append(avg_value)
        pass_at_n.append(pass_value)

    return labels, average_at_n, pass_at_n


def configure_style() -> None:
    """Apply a restrained paper-style Matplotlib theme."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": COLOR_EDGE,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def make_figure(
    labels: list[str],
    average_at_n: list[float],
    pass_at_n: list[float],
) -> plt.Figure:
    configure_style()

    x_positions = list(range(len(labels)))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.4),
        gridspec_kw={"wspace": 0.16},
    )
    avg_axis, pass_axis = axes

    def bar_panel(axis: plt.Axes, values: list[float], color: str, title: str) -> Any:
        bars = axis.bar(
            x_positions,
            values,
            0.62,
            color=color,
            edgecolor=COLOR_EDGE,
            linewidth=1.0,
            alpha=0.96,
            zorder=3,
        )
        low = min(values) - 0.04
        high = max(values) + 0.05
        axis.set_ylim(low, high)
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (high - low) * 0.015,
                f"{bar.get_height():.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLOR_EDGE,
            )
        axis.set_title(title, pad=14)
        return bars

    # Panel (a): Average@12 柱状图，橙色 + 斜纹标出最佳 checkpoint（与 03-search-r1 图例一致）。
    avg_bars = bar_panel(avg_axis, average_at_n, COLOR_AVG, "(a) Average@12")
    best_index = max(range(len(average_at_n)), key=average_at_n.__getitem__)
    avg_bars[best_index].set_color(COLOR_BEST)
    avg_bars[best_index].set_hatch("///")
    avg_axis.set_ylabel("Accuracy")

    # Panel (b): Pass@12 折线图，橙色圆点同步标出最佳 Average@12 对应的 checkpoint。
    low = min(pass_at_n) - 0.04
    high = max(pass_at_n) + 0.05
    pass_axis.set_ylim(low, high)
    pass_axis.plot(
        x_positions,
        pass_at_n,
        color=COLOR_PASS,
        marker="s",
        markersize=7,
        markerfacecolor="#57B8D2",
        markeredgecolor=COLOR_EDGE,
        markeredgewidth=0.9,
        linewidth=2.1,
        zorder=4,
    )
    pass_axis.scatter(
        [x_positions[best_index]],
        [pass_at_n[best_index]],
        s=90,
        facecolor=COLOR_BEST,
        edgecolor=COLOR_EDGE,
        linewidth=1.0,
        zorder=5,
    )
    for x, value in zip(x_positions, pass_at_n, strict=True):
        pass_axis.text(
            x,
            value + (high - low) * 0.015,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=COLOR_EDGE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.2},
            zorder=6,
        )
    pass_axis.set_title("(b) Pass@12", pad=14)

    for axis in axes:
        axis.set_xticks(x_positions, labels)
        axis.set_xlabel("Model / Checkpoint", labelpad=10)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.grid(
            axis="y",
            color="#C7C7C7",
            linestyle="-.",
            linewidth=0.8,
            alpha=0.75,
            zorder=0,
        )
        axis.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            length=5,
            width=0.9,
        )
        axis.margins(x=0.06)

    figure.suptitle(
        "OPSD AIME25 Checkpoint Evaluation",
        fontsize=17,
        y=0.98,
    )
    figure.text(
        0.5,
        0.915,
        "30 problems × 12 samples per checkpoint · values read from persisted JSONL summaries",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    figure.text(
        0.5,
        0.015,
        f"Orange indicates the best observed Average@12 checkpoint ({labels[best_index]}). "
        "Y-axes are truncated to highlight trends.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.83)
    return figure


def main() -> None:
    args = parse_args()
    labels, average_at_n, pass_at_n = load_metrics(args.result_dir)
    figure = make_figure(labels, average_at_n, pass_at_n)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure: {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
