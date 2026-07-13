"""Visualize the final SAR-OPD and IDT-OPD benchmark results.

The script creates two publication-style figures. Each figure contains two
subplots and is exported as both a 300-DPI PNG and a vector PDF.

Usage:
    uv run python 02-opd/analysis.py
    uv run python 02-opd/analysis.py --output-dir 02-opd/images --dpi 300
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "images"

MEDICAL_COLOR = "#DB4B4B"
GENERAL_COLOR = "#4B93C3"
INK_COLOR = "#22252A"
GRID_COLOR = "#C8CDD2"
NEUTRAL_COLOR = "#8A9199"
LIGHT_NEUTRAL = "#E6E9EC"

BASE_MEDQA = 72.17
BASE_CEVAL = 81.67


def parse_args() -> argparse.Namespace:
    """Parse output settings without coupling plotting to the training code."""

    parser = argparse.ArgumentParser(description="Plot SAR-OPD and IDT-OPD results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG and PDF figures",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    args = parser.parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be > 0")
    return args


def configure_style() -> None:
    """Apply one restrained, reference-inspired visual system."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.edgecolor": INK_COLOR,
            "axes.linewidth": 1.15,
            "axes.titleweight": "bold",
            "text.color": INK_COLOR,
            "axes.labelcolor": INK_COLOR,
            "xtick.color": INK_COLOR,
            "ytick.color": INK_COLOR,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    """Match the quiet grid, dark frame, and inward ticks of the reference."""

    ax.set_axisbelow(True)
    ax.grid(
        axis=grid_axis,
        color=GRID_COLOR,
        linestyle="-.",
        linewidth=0.8,
        alpha=0.8,
    )
    ax.tick_params(direction="in", top=True, right=True, length=5, width=1.0)
    for spine in ax.spines.values():
        spine.set_color(INK_COLOR)
        spine.set_linewidth(1.15)


def add_bar_labels(ax: plt.Axes, bars: list, values: list[float]) -> None:
    """Place exact percentage labels above grouped bars."""

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.25,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=INK_COLOR,
        )


def draw_grouped_bars(
    ax: plt.Axes,
    labels: list[str],
    medqa: list[float],
    ceval: list[float],
    *,
    highlighted_indices: set[int],
) -> None:
    """Draw a zero-based grouped comparison with color and hatch redundancy."""

    x = np.arange(len(labels), dtype=float)
    width = 0.34
    med_bars = ax.bar(
        x - width / 2,
        medqa,
        width,
        color=MEDICAL_COLOR,
        edgecolor=INK_COLOR,
        linewidth=1.15,
        alpha=0.95,
        label="MedQA-zh 1k",
        zorder=3,
    )
    ceval_bars = ax.bar(
        x + width / 2,
        ceval,
        width,
        color=GENERAL_COLOR,
        edgecolor=INK_COLOR,
        linewidth=1.15,
        hatch="///",
        alpha=0.92,
        label="C-Eval non-med 8k",
        zorder=3,
    )
    for index in highlighted_indices:
        med_bars[index].set_linewidth(2.3)
        ceval_bars[index].set_linewidth(2.3)

    add_bar_labels(ax, list(med_bars), medqa)
    add_bar_labels(ax, list(ceval_bars), ceval)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.margins(x=0.06)
    style_axes(ax)


def metric_legend_handles() -> list[Patch]:
    """Return consistent metric handles for both figures."""

    return [
        Patch(
            facecolor=MEDICAL_COLOR,
            edgecolor=INK_COLOR,
            linewidth=1.1,
            label="MedQA-zh 1k",
        ),
        Patch(
            facecolor=GENERAL_COLOR,
            edgecolor=INK_COLOR,
            linewidth=1.1,
            hatch="///",
            label="C-Eval non-med 8k",
        ),
    ]


def export_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    """Export a raster preview and a vector publication artifact."""

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def plot_sar_opd(output_dir: Path, dpi: int) -> None:
    """Plot SAR-OPD stage comparison and base-anchor recovery trajectory."""

    stage_labels = [
        "4B Base",
        "Medical SFT\n(epoch 3)",
        "Medical OPD\n(step 300)",
        "SAR-OPD\n(step 300)",
    ]
    stage_medqa = [72.17, 79.00, 80.00, 84.33]
    stage_ceval = [81.67, 69.33, 73.00, 84.67]

    recovery_steps = [0, 50, 100, 150, 200, 250, 300]
    recovery_medqa = [80.00, 84.17, 82.50, 83.00, 82.17, 83.33, 84.33]
    recovery_ceval = [73.00, 76.67, 79.33, 80.33, 83.67, 82.67, 84.67]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.35))
    fig.suptitle(
        "SAR-OPD: Staged Anchor-Restoration OPD",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.920,
        "Accuracy on MedQA-zh (600 questions) and C-Eval non-med (300 questions)",
        ha="center",
        fontsize=10.5,
        color="#555B63",
    )

    left, right = axes
    draw_grouped_bars(
        left,
        stage_labels,
        stage_medqa,
        stage_ceval,
        highlighted_indices={3},
    )
    left.set_title("(a) Stage-wise accuracy")

    right.plot(
        recovery_steps,
        recovery_medqa,
        color=MEDICAL_COLOR,
        marker="o",
        markersize=6.5,
        markeredgecolor=INK_COLOR,
        linewidth=2.2,
        label="MedQA-zh 1k",
        zorder=4,
    )
    right.plot(
        recovery_steps,
        recovery_ceval,
        color=GENERAL_COLOR,
        marker="s",
        markersize=6.2,
        markerfacecolor="white",
        markeredgecolor=GENERAL_COLOR,
        markeredgewidth=1.7,
        linewidth=2.2,
        linestyle="--",
        label="C-Eval non-med 8k",
        zorder=4,
    )
    right.axhline(
        BASE_MEDQA,
        color=MEDICAL_COLOR,
        linestyle=":",
        linewidth=1.5,
        alpha=0.75,
        label="4B Base: MedQA",
        zorder=2,
    )
    right.axhline(
        BASE_CEVAL,
        color=GENERAL_COLOR,
        linestyle=":",
        linewidth=1.5,
        alpha=0.75,
        label="4B Base: C-Eval",
        zorder=2,
    )
    medqa_label_offsets = {200: (0, -16)}
    ceval_label_offsets = {200: (0, 10)}
    for step, med_value, ceval_value in zip(
        recovery_steps,
        recovery_medqa,
        recovery_ceval,
        strict=True,
    ):
        right.annotate(
            f"{med_value:.2f}",
            (step, med_value),
            xytext=medqa_label_offsets.get(step, (0, 9)),
            textcoords="offset points",
            ha="center",
            fontsize=8.2,
            color=MEDICAL_COLOR,
        )
        right.annotate(
            f"{ceval_value:.2f}",
            (step, ceval_value),
            xytext=ceval_label_offsets.get(step, (0, -14)),
            textcoords="offset points",
            ha="center",
            fontsize=8.2,
            color=GENERAL_COLOR,
        )
    right.scatter(
        [300, 300],
        [recovery_medqa[-1], recovery_ceval[-1]],
        s=115,
        facecolors="none",
        edgecolors=INK_COLOR,
        linewidths=1.8,
        zorder=5,
    )
    right.set_title("(b) Base-anchor recovery trajectory (focused scale)")
    right.set_xlabel("Base-anchor C-Eval OPD step")
    right.set_ylabel("Accuracy (%)")
    right.set_xticks(recovery_steps)
    right.set_ylim(68, 87)
    right.set_yticks(np.arange(70, 88, 2.5))
    right.margins(x=0.04)
    style_axes(right)

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=MEDICAL_COLOR,
                marker="o",
                markeredgecolor=INK_COLOR,
                linewidth=2.2,
                label="MedQA-zh 1k",
            ),
            Line2D(
                [0],
                [0],
                color=GENERAL_COLOR,
                marker="s",
                markerfacecolor="white",
                linestyle="--",
                linewidth=2.2,
                label="C-Eval non-med 8k",
            ),
            Line2D(
                [0],
                [0],
                color=INK_COLOR,
                marker="o",
                markerfacecolor="none",
                linewidth=0,
                markersize=9,
                label="Selected checkpoint",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.870),
        ncol=3,
        frameon=True,
        fancybox=True,
        edgecolor=INK_COLOR,
        framealpha=0.96,
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.72, wspace=0.25)
    export_figure(fig, output_dir, "sar_opd_analysis", dpi)
    plt.close(fig)


def plot_idt_opd(output_dir: Path, dpi: int) -> None:
    """Plot IDT-OPD key checkpoints and the medical/general Pareto frontier."""

    comparison_labels = [
        "4B Base",
        "Medical SFT\nTeacher",
        "IDT-OPD\nstep 100",
        "IDT-OPD\nstep 200",
        "IDT-OPD\nstep 600",
    ]
    comparison_medqa = [72.17, 79.00, 79.17, 76.50, 75.00]
    comparison_ceval = [81.67, 69.33, 81.67, 86.00, 82.00]

    checkpoint_steps = [100, 200, 300, 400, 500, 600]
    checkpoint_medqa = [79.17, 76.50, 76.33, 75.17, 75.33, 75.00]
    checkpoint_ceval = [81.67, 86.00, 84.33, 81.33, 83.67, 82.00]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.35))
    fig.suptitle(
        "IDT-OPD: Interleaved Dual-Teacher OPD",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.920,
        "1:1 Medical/C-Eval scheduling; accuracy on fixed 600- and 300-question test sets",
        ha="center",
        fontsize=10.5,
        color="#555B63",
    )

    left, right = axes
    draw_grouped_bars(
        left,
        comparison_labels,
        comparison_medqa,
        comparison_ceval,
        highlighted_indices={2, 3},
    )
    left.set_title("(a) Key checkpoint comparison")

    right.plot(
        checkpoint_ceval,
        checkpoint_medqa,
        color=NEUTRAL_COLOR,
        linewidth=1.4,
        linestyle="--",
        alpha=0.85,
        zorder=2,
    )
    right.scatter(
        checkpoint_ceval,
        checkpoint_medqa,
        s=72,
        color=GENERAL_COLOR,
        edgecolors=INK_COLOR,
        linewidths=1.0,
        zorder=4,
    )
    right.scatter(
        [BASE_CEVAL, 69.33],
        [BASE_MEDQA, 79.00],
        s=82,
        facecolors="white",
        edgecolors=NEUTRAL_COLOR,
        linewidths=1.8,
        marker="D",
        zorder=4,
    )
    pareto_x = [checkpoint_ceval[0], checkpoint_ceval[1]]
    pareto_y = [checkpoint_medqa[0], checkpoint_medqa[1]]
    right.plot(
        pareto_x,
        pareto_y,
        color=MEDICAL_COLOR,
        linewidth=2.2,
        linestyle=":",
        zorder=3,
    )
    right.scatter(
        pareto_x,
        pareto_y,
        s=150,
        facecolors="none",
        edgecolors=MEDICAL_COLOR,
        linewidths=2.4,
        zorder=5,
    )

    reference_points = [
        (BASE_CEVAL, BASE_MEDQA, "4B Base", (-8, -16)),
        (69.33, 79.00, "Medical SFT", (7, 8)),
    ]
    for x_value, y_value, label, offset in reference_points:
        right.annotate(
            label,
            (x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color="#5F666E",
        )

    offsets = {
        100: (-18, 10),
        200: (-18, 9),
        300: (-8, -16),
        400: (-19, -16),
        500: (7, -2),
        600: (-2, 9),
    }
    for step, x_value, y_value in zip(
        checkpoint_steps,
        checkpoint_ceval,
        checkpoint_medqa,
        strict=True,
    ):
        right.annotate(
            f"step {step}",
            (x_value, y_value),
            xytext=offsets[step],
            textcoords="offset points",
            fontsize=8.3,
            fontweight="bold" if step in {100, 200} else "normal",
            color=INK_COLOR,
        )

    right.set_title("(b) Medical-general Pareto frontier (focused scale)")
    right.set_xlabel("C-Eval non-med 8k accuracy (%)  →")
    right.set_ylabel("MedQA-zh 1k accuracy (%)  →")
    right.set_xlim(68, 87.2)
    right.set_ylim(71, 80.4)
    right.set_xticks(np.arange(70, 88, 2.5))
    right.set_yticks(np.arange(72, 81, 1.5))
    style_axes(right, grid_axis="both")
    right.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=GENERAL_COLOR,
                markeredgecolor=INK_COLOR,
                markersize=7,
                label="IDT checkpoints",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="white",
                markeredgecolor=NEUTRAL_COLOR,
                markersize=7,
                label="References",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=MEDICAL_COLOR,
                markerfacecolor="none",
                markeredgecolor=MEDICAL_COLOR,
                linestyle=":",
                markersize=9,
                label="Best trade-offs (Pareto)",
            ),
        ],
        loc="lower left",
        frameon=True,
        fancybox=True,
        edgecolor=INK_COLOR,
        framealpha=0.95,
        fontsize=8.5,
    )

    fig.legend(
        handles=metric_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.870),
        ncol=2,
        frameon=True,
        fancybox=True,
        edgecolor=INK_COLOR,
        framealpha=0.96,
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.72, wspace=0.25)
    export_figure(fig, output_dir, "idt_opd_analysis", dpi)
    plt.close(fig)


def main() -> None:
    """Generate exactly two figures, one for each training scheme."""

    args = parse_args()
    configure_style()
    plot_sar_opd(args.output_dir, args.dpi)
    plot_idt_opd(args.output_dir, args.dpi)


if __name__ == "__main__":
    main()
