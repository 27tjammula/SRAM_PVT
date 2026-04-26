#!/usr/bin/env python3
"""Plot baseline vs high-Vt read SNM by corner.

Visually identical to `scripts/plot_high_vt_leakage_energy.py`. Read-SNM
values come from the bitcell summary CSV (limiting-eye largest-square fit on
the read butterfly curves), so this script does not re-derive the SNM itself.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = REPO_ROOT / "sram_analysis_plots" / "report_figures" / "bitcell" / "bitcell_summary.csv"
OUTPUT_DIR = REPO_ROOT / "sram_analysis_plots" / "report_figures" / "bitcell" / "high_vt"
FIGURE_PATH = OUTPUT_DIR / "high_vt_read_snm_by_corner.png"
OUT_CSV = OUTPUT_DIR / "high_vt_read_snm_by_corner.csv"

# SS 0.8 V is intentionally absent: the baseline cell does not have a valid
# read butterfly at that supply without a wordline-underdrive assist, so the
# limiting-eye SNM is undefined. The energy-bar twin chart keeps SS 0.8 V
# because the transient supply current is still measurable there.
CASE_ORDER = (
    {"key": "ff",    "label": "FF\n1.2 V"},
    {"key": "tt",    "label": "TT\n1.0 V"},
    {"key": "ss_1V", "label": "SS\n1.0 V"},
)

# Matches the canonical bitcell-figure palette in
# `scripts/plot_bitcell_report_figures.py` so this chart pairs visually with
# `bitcell/high_vt/high_vt_hold_leakage_energy_by_corner.png`.
BASELINE_COLOR = "#4c72b0"
HIGH_VT_COLOR = "#dd8452"
DELTA_BETTER_COLOR = "#2f7f4f"
DELTA_WORSE_COLOR = "#b23a48"

FIGSIZE = (6.8, 4.2)
BAR_WIDTH = 0.34
HEADROOM_FACTOR = 1.30
DPI = 300


@dataclass(frozen=True)
class ReadSnmPoint:
    corner: str
    label: str
    baseline_mv: float
    high_vt_mv: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--output", type=Path, default=FIGURE_PATH)
    parser.add_argument("--summary-out", type=Path, default=OUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args.summary_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_summary_csv(args.summary_out, rows)
    configure_matplotlib()
    plot_read_snm_bars(args.output, rows)
    print(args.summary_out.relative_to(REPO_ROOT))
    print(args.output.relative_to(REPO_ROOT))
    print(args.output.with_suffix(".pdf").relative_to(REPO_ROOT))


def configure_matplotlib() -> None:
    """Mirrors the rcParams in `plot_bitcell_report_figures.py`."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
        }
    )


def build_rows(summary_path: Path) -> list[ReadSnmPoint]:
    baseline: dict[str, float] = {}
    high_vt: dict[str, float] = {}
    with summary_path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["metric_family"] != "Read SNM":
                continue
            if row["case"] == "Baseline":
                baseline[row["corner"]] = float(row["value"])
            elif row["case"] == "High Vt":
                high_vt[row["corner"]] = float(row["value"])
    points: list[ReadSnmPoint] = []
    for case in CASE_ORDER:
        key = case["key"]
        if key not in baseline or key not in high_vt:
            raise KeyError(f"Missing read-SNM data for corner {key} in {summary_path}")
        points.append(
            ReadSnmPoint(
                corner=key,
                label=case["label"],
                baseline_mv=baseline[key],
                high_vt_mv=high_vt[key],
            )
        )
    return points


def write_summary_csv(path: Path, rows: list[ReadSnmPoint]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "corner",
                "label",
                "baseline_read_snm_mV",
                "high_vt_read_snm_mV",
                "delta_mV",
                "delta_percent",
            ]
        )
        for row in rows:
            delta = row.high_vt_mv - row.baseline_mv
            delta_pct = 100.0 * delta / row.baseline_mv
            writer.writerow(
                [
                    row.corner,
                    row.label.replace("\n", " "),
                    f"{row.baseline_mv:.6f}",
                    f"{row.high_vt_mv:.6f}",
                    f"{delta:.6f}",
                    f"{delta_pct:.6f}",
                ]
            )


def plot_read_snm_bars(output_path: Path, rows: list[ReadSnmPoint]) -> None:
    plot_high_vt_grouped_bars(
        output_path=output_path,
        labels=[row.label for row in rows],
        baseline_values=[row.baseline_mv for row in rows],
        high_vt_values=[row.high_vt_mv for row in rows],
        y_label="Read SNM (mV)",
        higher_is_better=True,
    )


def plot_high_vt_grouped_bars(
    output_path: Path,
    labels: list[str],
    baseline_values: list[float],
    high_vt_values: list[float],
    *,
    y_label: str,
    higher_is_better: bool,
    figsize: tuple[float, float] = FIGSIZE,
    bar_width: float = BAR_WIDTH,
    headroom_factor: float = HEADROOM_FACTOR,
    dpi: int = DPI,
) -> None:
    """Render the canonical baseline vs High-Vt grouped-bar figure.

    Kept identical to the implementation in `plot_high_vt_leakage_energy.py`
    so the report's High-Vt section reads as a matched pair.
    """
    x = np.arange(len(labels), dtype=float)
    y_max = max(max(baseline_values), max(high_vt_values))

    fig, ax = plt.subplots(figsize=figsize)
    baseline_pos = x - bar_width / 2
    high_vt_pos = x + bar_width / 2

    ax.bar(baseline_pos, baseline_values, bar_width, label="Baseline", color=BASELINE_COLOR)
    ax.bar(high_vt_pos, high_vt_values, bar_width, label="High-Vt", color=HIGH_VT_COLOR, alpha=0.90)

    ax.set_ylabel(y_label)
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    ax.set_ylim(0.0, headroom_factor * y_max)
    apply_paper_style(ax)

    annotate_bar_values(ax, baseline_pos, baseline_values, y_max)
    annotate_bar_values(ax, high_vt_pos, high_vt_values, y_max)

    for index, (baseline_value, high_vt_value) in enumerate(zip(baseline_values, high_vt_values)):
        delta_pct = 100.0 * (high_vt_value - baseline_value) / baseline_value
        is_better = (delta_pct > 0.0) if higher_is_better else (delta_pct < 0.0)
        ax.text(
            x[index],
            max(baseline_value, high_vt_value) + 0.11 * y_max,
            f"{delta_pct:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=DELTA_BETTER_COLOR if is_better else DELTA_WORSE_COLOR,
            fontweight="bold",
        )

    fig.tight_layout()
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)


def apply_paper_style(ax: plt.Axes) -> None:
    """Project paper-style: no grid, no minor ticks, inward ticks on all axes, no title.

    Mirrors `format_axis_common` in `plot_bitcell_report_figures.py` so this
    chart's tick weight and direction match the canonical bitcell figures.
    """
    ax.tick_params(direction="in", top=True, right=True, length=4.0, width=0.9)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def annotate_bar_values(ax: plt.Axes, x: np.ndarray, values: list[float], y_max: float) -> None:
    for xpos, value in zip(x, values):
        ax.text(
            float(xpos),
            value + 0.03 * y_max,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="0.2",
        )


if __name__ == "__main__":
    main()
