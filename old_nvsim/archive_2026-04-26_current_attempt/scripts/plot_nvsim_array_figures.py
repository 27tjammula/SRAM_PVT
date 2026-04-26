#!/usr/bin/env python3
"""Generate report-ready NVSim array figures from the array analysis CSV outputs."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


REPO_ROOT = Path(__file__).resolve().parents[1]
ARRAY_ROOT = REPO_ROOT / "sram_analysis_plots" / "array"
DATA_DIR = ARRAY_ROOT / "data"
FIGURES_DIR = ARRAY_ROOT / "figures"
DOCS_DIR = ARRAY_ROOT / "docs"

PNG_DPI = 400
BAR_FIGSIZE = (6.8, 4.2)

CASE_ORDER = ["Baseline", "High Vt", "Negative BL", "WL Underdrive"]
CASE_COLORS = {
    "Baseline": "#4c72b0",
    "High Vt": "#dd8452",
    "Negative BL": "#55a868",
    "WL Underdrive": "#8172b2",
}
READ_COLOR = "#4c72b0"
WRITE_COLOR = "#dd8452"


@dataclass(frozen=True)
class GuideEntry:
    relative_png: Path
    description: str
    source_files: list[Path]
    processing_steps: list[str]
    takeaway: str
    caveats: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-ready NVSim array figures from CSV outputs.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing array CSV outputs.")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="Directory to save figures into.")
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR, help="Directory to save the figure guide into.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    docs_dir = args.docs_dir.resolve()

    figures_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    case_rows = read_csv_rows(data_dir / "nvsim_case_comparison.csv")
    bitcell_rows = read_csv_rows(data_dir / "bitcell_metrics_summary.csv")
    combined_rows = read_csv_rows(data_dir / "array_analysis_combined_summary.csv")

    ordered_case_rows = sort_case_rows(case_rows)
    guide_entries: list[GuideEntry] = []

    guide_entries.append(plot_macro_latency_comparison(figures_dir, ordered_case_rows))
    guide_entries.append(plot_macro_energy_comparison(figures_dir, ordered_case_rows))
    guide_entries.append(plot_macro_leakage_comparison(figures_dir, ordered_case_rows))
    guide_entries.append(plot_macro_area_summary(figures_dir, ordered_case_rows))
    guide_entries.append(plot_bitcell_stability_summary(figures_dir, ordered_case_rows))
    guide_entries.append(plot_optimization_trend_summary(figures_dir, bitcell_rows, combined_rows))

    guide_path = docs_dir / "figure_guide.md"
    write_figure_guide(guide_path, guide_entries)
    verify_outputs(guide_entries, guide_path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def sort_case_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    order = {case: index for index, case in enumerate(CASE_ORDER)}
    return sorted(rows, key=lambda row: order[row["case"]])


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
        }
    )


def plot_macro_latency_comparison(figures_dir: Path, rows: list[dict[str, str]]) -> GuideEntry:
    configure_matplotlib()
    labels = [row["case"] for row in rows]
    read_ns = np.asarray([float(row["read_latency_ns"]) for row in rows], dtype=float)
    write_ns = np.asarray([float(row["write_latency_ns"]) for row in rows], dtype=float)
    x = np.arange(len(labels), dtype=float)
    width = 0.34

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    read_bars = ax.bar(x - width / 2, read_ns, width, color=READ_COLOR, label="Read")
    write_bars = ax.bar(x + width / 2, write_ns, width, color=WRITE_COLOR, label="Write")
    ax.set_ylabel("Latency (ns)")
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    format_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    annotate_bars(ax, read_bars, "{:.2f}")
    annotate_bars(ax, write_bars, "{:.2f}")

    base_path = figures_dir / "macro_latency_comparison"
    save_figure(fig, base_path)
    plt.close(fig)
    return GuideEntry(
        relative_png=base_path.with_suffix(".png").relative_to(ARRAY_ROOT),
        description="Read and write latency comparison across the Baseline, High Vt, Negative BL, and WL Underdrive macro cases.",
        source_files=[DATA_DIR / "nvsim_case_comparison.csv"],
        processing_steps=[
            "Loaded the macro case comparison table from the array data directory.",
            "Used the NVSim-reported read and write latency columns without additional smoothing or normalization.",
            "Plotted grouped bars on a shared nanosecond axis for the four report cases.",
        ],
        takeaway="Shows which cases change the macro-level latency estimate and which cases reuse the same baseline array timing.",
        caveats=[
            "Negative BL and WL Underdrive keep the baseline macro latency because this NVSim build does not directly model those assist waveforms.",
        ],
    )


def plot_macro_energy_comparison(figures_dir: Path, rows: list[dict[str, str]]) -> GuideEntry:
    configure_matplotlib()
    labels = [row["case"] for row in rows]
    read_pj = np.asarray([float(row["read_energy_pj"]) for row in rows], dtype=float)
    write_pj = np.asarray([float(row["write_energy_pj"]) for row in rows], dtype=float)
    x = np.arange(len(labels), dtype=float)
    width = 0.34

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    read_bars = ax.bar(x - width / 2, read_pj, width, color=READ_COLOR, label="Read")
    write_bars = ax.bar(x + width / 2, write_pj, width, color=WRITE_COLOR, label="Write")
    ax.set_ylabel("Energy (pJ)")
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    format_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    annotate_bars(ax, read_bars, "{:.2f}")
    annotate_bars(ax, write_bars, "{:.2f}")

    base_path = figures_dir / "macro_energy_comparison"
    save_figure(fig, base_path)
    plt.close(fig)
    return GuideEntry(
        relative_png=base_path.with_suffix(".png").relative_to(ARRAY_ROOT),
        description="Read and write energy comparison across the Baseline, High Vt, Negative BL, and WL Underdrive macro cases.",
        source_files=[DATA_DIR / "nvsim_case_comparison.csv"],
        processing_steps=[
            "Loaded the array case comparison table from the generated CSV outputs.",
            "Read the NVSim dynamic-energy columns directly in picojoules.",
            "Plotted grouped read and write bars on one shared energy axis.",
        ],
        takeaway="Makes it clear whether any optimization changes the macro-level dynamic energy estimate in the current NVSim-supported mapping.",
        caveats=[
            "Only the High Vt case applies a supported leakage scaling. The other assist cases reuse the baseline NVSim energy estimate.",
        ],
    )


def plot_macro_leakage_comparison(figures_dir: Path, rows: list[dict[str, str]]) -> GuideEntry:
    configure_matplotlib()
    labels = [row["case"] for row in rows]
    leakage_uw = np.asarray([float(row["leakage_power_uw"]) for row in rows], dtype=float)
    x = np.arange(len(labels), dtype=float)
    colors = [CASE_COLORS[row["case"]] for row in rows]

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    bars = ax.bar(x, leakage_uw, color=colors, width=0.60)
    ax.set_ylabel("Leakage power (uW)")
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    format_axis(ax)
    annotate_bars(ax, bars, "{:.3f}")

    base_path = figures_dir / "macro_leakage_comparison"
    save_figure(fig, base_path)
    plt.close(fig)
    return GuideEntry(
        relative_png=base_path.with_suffix(".png").relative_to(ARRAY_ROOT),
        description="Leakage power comparison across the Baseline, High Vt, Negative BL, and WL Underdrive macro cases.",
        source_files=[DATA_DIR / "nvsim_case_comparison.csv"],
        processing_steps=[
            "Loaded the NVSim case comparison table.",
            "Read the leakage-power column directly in microwatts.",
            "Used one bar per case with fixed case colors across the full figure set.",
        ],
        takeaway="Highlights that High Vt is the only case that changes the macro leakage estimate in the current supported NVSim mapping.",
        caveats=[
            "The High Vt leakage change comes from an `IoffScale` derived from TT hold-window supply energy, not from a full transistor-level macro extraction.",
        ],
    )


def plot_macro_area_summary(figures_dir: Path, rows: list[dict[str, str]]) -> GuideEntry:
    configure_matplotlib()
    labels = [row["case"] for row in rows]
    area_mm2 = np.asarray([float(row["area_mm2"]) for row in rows], dtype=float)
    x = np.arange(len(labels), dtype=float)
    colors = [CASE_COLORS[row["case"]] for row in rows]

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    bars = ax.bar(x, area_mm2, color=colors, width=0.60)
    ax.set_ylabel("Area (mm^2)")
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    format_axis(ax)
    annotate_bars(ax, bars, "{:.5f}")

    base_path = figures_dir / "macro_area_summary"
    save_figure(fig, base_path)
    plt.close(fig)
    return GuideEntry(
        relative_png=base_path.with_suffix(".png").relative_to(ARRAY_ROOT),
        description="Total macro area summary for the four reported SRAM cases.",
        source_files=[DATA_DIR / "nvsim_case_comparison.csv"],
        processing_steps=[
            "Loaded the macro case comparison table from the array data directory.",
            "Read the total area column in square millimeters.",
            "Plotted one bar per case to show that all cases share the same topology area in the current flow.",
        ],
        takeaway="Shows that the current case comparison changes leakage or bitcell trends more than physical macro area.",
        caveats=[
            "All four cases use the same macro topology, so identical area bars are expected unless a future flow explicitly changes the physical organization.",
        ],
    )


def plot_bitcell_stability_summary(figures_dir: Path, rows: list[dict[str, str]]) -> GuideEntry:
    configure_matplotlib()
    metric_labels = ["Hold SNM", "Read SNM", "Write NM"]
    metric_keys = [
        "bitcell_hold_snm_mv_tt",
        "bitcell_read_snm_mv_tt",
        "bitcell_write_nm_mv_tt",
    ]
    x = np.arange(len(metric_labels), dtype=float)
    width = 0.18

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    for index, case in enumerate(CASE_ORDER):
        case_row = next(row for row in rows if row["case"] == case)
        positions = x + (index - 1.5) * width
        heights = []
        valid_positions = []
        for position, key in zip(positions, metric_keys):
            value = parse_optional_float(case_row[key])
            if value is None:
                continue
            valid_positions.append(position)
            heights.append(value)
        if heights:
            bars = ax.bar(valid_positions, heights, width, color=CASE_COLORS[case], label=case)
            annotate_bars(ax, bars, "{:.1f}")

    ax.set_ylabel("Margin (mV)")
    ax.set_xticks(x, metric_labels)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    format_axis(ax)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    base_path = figures_dir / "bitcell_stability_summary"
    save_figure(fig, base_path)
    plt.close(fig)
    return GuideEntry(
        relative_png=base_path.with_suffix(".png").relative_to(ARRAY_ROOT),
        description="TT bitcell stability summary comparing hold SNM, read SNM, and write noise margin where each case has supporting data.",
        source_files=[DATA_DIR / "nvsim_case_comparison.csv"],
        processing_steps=[
            "Loaded the case comparison CSV that already carries the TT bitcell support metrics alongside the macro results.",
            "Plotted only the metrics available for each case instead of fabricating missing stability numbers.",
            "Kept the same case color mapping used throughout the array figure set.",
        ],
        takeaway="Provides one compact view of which cases improve or degrade the TT stability-related Cadence evidence.",
        caveats=[
            "Missing bars mean that no robust TT metric was available for that case and metric family in the current dataset.",
        ],
    )


def plot_optimization_trend_summary(
    figures_dir: Path,
    bitcell_rows: list[dict[str, str]],
    combined_rows: list[dict[str, str]],
) -> GuideEntry:
    configure_matplotlib()
    labels = ["High Vt", "Negative BL", "WL Underdrive"]
    values = [
        improvement_percent(bitcell_rows, "High Vt", "Hold-Window Supply Energy", "tt"),
        improvement_percent(bitcell_rows, "Negative BL", "Write Delay", "tt"),
        improvement_percent(bitcell_rows, "WL Underdrive", "Read Disturb", "tt"),
    ]
    colors = [CASE_COLORS[label] for label in labels]
    x = np.arange(len(labels), dtype=float)

    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    bars = ax.bar(x, values, color=colors, width=0.60)
    ax.set_ylabel("Improvement vs baseline (%)")
    ax.set_xticks(x, labels)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    format_axis(ax)
    annotate_bars(ax, bars, "{:.1f}")

    base_path = figures_dir / "optimization_trend_summary"
    save_figure(fig, base_path)
    plt.close(fig)
    return GuideEntry(
        relative_png=base_path.with_suffix(".png").relative_to(ARRAY_ROOT),
        description="Cadence-backed optimization trend summary using one representative TT improvement metric per optimization case.",
        source_files=[
            DATA_DIR / "bitcell_metrics_summary.csv",
            DATA_DIR / "array_analysis_combined_summary.csv",
        ],
        processing_steps=[
            "Loaded the long-form bitcell metric summary and the combined array summary tables.",
            "Converted TT percentage deltas into positive improvement magnitudes for High Vt, Negative BL, and WL Underdrive.",
            "Plotted one bar per optimization with shared percentage units to keep the summary compact.",
        ],
        takeaway="Summarizes the most important Cadence-backed benefit of each optimization without implying unsupported macro-level assist modeling.",
        caveats=[
            "The three bars represent different physical metrics: hold-window supply energy for High Vt, write delay for Negative BL, and read disturb for WL Underdrive.",
        ],
    )


def improvement_percent(
    bitcell_rows: list[dict[str, str]],
    case: str,
    metric_group: str,
    corner: str,
) -> float:
    for row in bitcell_rows:
        if row["case"] == case and row["metric_group"] == metric_group and row["corner"] == corner:
            delta_percent = float(row["delta_percent"])
            return -delta_percent
    raise ValueError(f"Could not find {case} / {metric_group} / {corner} in bitcell metrics summary")


def parse_optional_float(text: str) -> float | None:
    return None if text == "NA" else float(text)


def annotate_bars(ax: plt.Axes, bars, fmt: str) -> None:
    heights = [bar.get_height() for bar in bars if not math.isnan(bar.get_height())]
    if not heights:
        return
    y_max = max(heights)
    for bar in bars:
        height = bar.get_height()
        if math.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02 * max(y_max, 1e-9),
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8.0,
        )
    ax.set_ylim(top=max(ax.get_ylim()[1], y_max * 1.16))


def format_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", top=True, right=True)
    ax.minorticks_off()


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path.relative_to(REPO_ROOT))
    print(pdf_path.relative_to(REPO_ROOT))


def write_figure_guide(path: Path, entries: list[GuideEntry]) -> None:
    lines = [
        "# Array Figure Guide",
        "",
        "Generated by `scripts/plot_nvsim_array_figures.py` from the CSV outputs under `sram_analysis_plots/array/data/`.",
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                f"## `{entry.relative_png.as_posix()}`",
                "",
                f"1. Figure filename: `{entry.relative_png.name}` and matching `.pdf` saved alongside it",
                f"2. Brief description of what the figure shows: {entry.description}",
                "3. Source data file or files used:",
            ]
        )
        for source in entry.source_files:
            lines.append(f"   - `{source.relative_to(REPO_ROOT).as_posix()}`")
        lines.append("4. Key processing steps applied:")
        for step in entry.processing_steps:
            lines.append(f"   - {step}")
        lines.append(f"5. What the figure helps the reader understand: {entry.takeaway}")
        lines.append("6. Any important caveats or assumptions:")
        for caveat in entry.caveats:
            lines.append(f"   - {caveat}")
        lines.append("")
    path.write_text("\n".join(lines))
    print(path.relative_to(REPO_ROOT))


def verify_outputs(entries: list[GuideEntry], guide_path: Path) -> None:
    guide_text = guide_path.read_text()
    for entry in entries:
        png_path = ARRAY_ROOT / entry.relative_png
        pdf_path = png_path.with_suffix(".pdf")
        if not png_path.exists():
            raise FileNotFoundError(f"Missing generated figure: {png_path}")
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing generated figure: {pdf_path}")
        if entry.relative_png.as_posix() not in guide_text:
            raise ValueError(f"Figure guide is missing entry for {entry.relative_png}")


if __name__ == "__main__":
    main()
