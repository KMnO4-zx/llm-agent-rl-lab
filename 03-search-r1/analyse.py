"""Plot Macro EM and format rate across Search-R1 checkpoints.

Run from the repository root:

    uv run python 03-search-r1/analyse.py

The figure is written to ``03-search-r1/eval_result/checkpoint_em_format.png``
by default. Each metric is read from the final ``type=summary`` record of the
corresponding evaluation JSONL file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_DIR / "eval_result"
DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "checkpoint_em_format.png"

CHECKPOINTS = (
    ("Base", "eval_results.jsonl"),
    ("Step 20", "eval_results_rl_step_20.jsonl"),
    ("Step 50", "eval_results_rl_step_50.jsonl"),
    ("Step 100", "eval_results_rl_step_100.jsonl"),
    ("Step 150", "eval_results_rl_step_150.jsonl"),
    ("Step 200", "eval_results_rl_step_200.jsonl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing eval_results*.jsonl",
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
    labels: list[str] = []
    macro_em: list[float] = []
    format_rate: list[float] = []

    for label, filename in CHECKPOINTS:
        path = result_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation result: {path}")

        summary = load_summary(path)
        metrics = summary.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Summary metrics are missing in {path}")

        try:
            em_value = float(metrics["em/macro"])
            format_value = float(metrics["format/rate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Expected numeric em/macro and format/rate metrics in {path}"
            ) from error

        labels.append(label)
        macro_em.append(em_value)
        format_rate.append(format_value)

    return labels, macro_em, format_rate


def configure_style() -> None:
    """Apply a restrained paper-style Matplotlib theme."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": "#202020",
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_value_labels(
    axis: plt.Axes,
    x_positions: list[int],
    values: list[float],
    *,
    offset: float,
) -> None:
    for x_position, value in zip(x_positions, values, strict=True):
        axis.text(
            x_position,
            value + offset,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#202020",
        )


def make_figure(
    labels: list[str],
    macro_em: list[float],
    format_rate: list[float],
) -> plt.Figure:
    configure_style()

    x_positions = list(range(len(labels)))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.4),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    em_axis, format_axis = axes

    # Panel (a): absolute checkpoint comparison.
    em_colors = ["#D94A4A"] * len(labels)
    em_colors[2] = "#F28E2B"
    bars = em_axis.bar(
        x_positions,
        macro_em,
        width=0.66,
        color=em_colors,
        edgecolor="#202020",
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
    )
    bars[2].set_hatch("///")
    add_value_labels(em_axis, x_positions, macro_em, offset=0.025)
    em_axis.set_title("(a) Macro Exact Match", pad=14)
    em_axis.set_ylabel("Score")

    # Panel (b): ordered checkpoint trend.
    format_axis.plot(
        x_positions,
        format_rate,
        color="#3388B8",
        marker="s",
        markersize=7,
        markerfacecolor="#57B8D2",
        markeredgecolor="#202020",
        markeredgewidth=0.9,
        linewidth=2.1,
        zorder=4,
    )
    format_axis.scatter(
        [x_positions[2]],
        [format_rate[2]],
        s=90,
        facecolor="#F28E2B",
        edgecolor="#202020",
        linewidth=1.0,
        zorder=5,
    )
    add_value_labels(format_axis, x_positions, format_rate, offset=0.025)
    format_axis.set_title("(b) Valid Answer Format Rate", pad=14)

    for axis in axes:
        axis.set_xticks(x_positions, labels)
        axis.set_xlabel("Model / Checkpoint", labelpad=10)
        axis.set_ylim(0.0, 1.08)
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
        "Search-R1 Checkpoint Evaluation",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.935,
        "Fixed 70-question evaluation set · values read from persisted JSONL summaries",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    figure.text(
        0.5,
        0.015,
        "Orange indicates the best observed Macro EM checkpoint (Step 50).",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.84)
    return figure


def main() -> None:
    args = parse_args()
    labels, macro_em, format_rate = load_metrics(args.result_dir)
    figure = make_figure(labels, macro_em, format_rate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure: {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
