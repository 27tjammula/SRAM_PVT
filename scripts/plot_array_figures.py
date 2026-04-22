#!/usr/bin/env python3
"""Generate report figures from the SRAM bitcell and NVSim summaries."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = REPO_ROOT / "sram_analysis_plots"
NVSIM_DIR = PLOTS_DIR / "nvsim_128x128_45nm"
FIGURE_DIR = PLOTS_DIR / "report_figures"

NUM_BITS = 128 * 128
ROW_BITS = 128
CORNER_ORDER = ["ff", "tt", "ss_1", "ss_08"]
CORNER_LABELS = {
    "ff": "FF\n1.2 V",
    "tt": "TT\n1.0 V",
    "ss_1": "SS\n1.0 V",
    "ss_08": "SS\n0.8 V",
}

COLORS = {
    "read": "#2f6fbb",
    "write": "#d95f02",
    "leakage": "#7a3e9d",
    "bitcell": "#4c9f70",
    "array": "#c44e52",
    "rowdec": "#8a6f3d",
    "bitline": "#008b8b",
    "other": "#8d8d8d",
    "roadmap": "#566573",
    "calibrated": "#b35c1e",
}


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    calibrated = sorted_by_corner(read_rows(NVSIM_DIR / "calibrated_summary.csv"))
    roadmap = read_rows(NVSIM_DIR / "summary.csv")
    bitcell = sorted_by_corner(read_rows(PLOTS_DIR / "rw_metrics.csv"))

    output_paths = [
        plot_calibrated_latency(calibrated),
        plot_calibrated_energy(calibrated),
        plot_calibrated_leakage(calibrated),
        plot_latency_contributors(calibrated),
        plot_array_vs_bitcell(bitcell, calibrated),
        plot_roadmap_vs_calibrated(roadmap, calibrated),
    ]
    write_index(output_paths)

    print("Generated figures:")
    for path in output_paths:
        print(path.relative_to(REPO_ROOT))
    print((FIGURE_DIR / "figure_index.md").relative_to(REPO_ROOT))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def sorted_by_corner(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_corner = {row["corner"]: row for row in rows}
    return [by_corner[corner] for corner in CORNER_ORDER if corner in by_corner]


def plot_calibrated_latency(rows: list[dict[str, str]]) -> Path:
    labels = corner_labels(rows)
    read_lat = values(rows, "read_lat_ps")
    write_lat = values(rows, "write_lat_ps")
    x = list(range(len(rows)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar([i - width / 2 for i in x], read_lat, width, label="Read", color=COLORS["read"])
    ax.bar([i + width / 2 for i in x], write_lat, width, label="Write", color=COLORS["write"], alpha=0.86)
    ax.set_title("Cadence-Calibrated NVSim Latency by Corner")
    ax.set_ylabel("Latency (ps)")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, ncols=2)
    style_axis(ax)
    annotate_bars(ax, [i - width / 2 for i in x], read_lat, fmt="{:.0f}")
    annotate_bars(ax, [i + width / 2 for i in x], write_lat, fmt="{:.0f}")
    return save(fig, "calibrated_latency_by_corner.png")


def plot_calibrated_energy(rows: list[dict[str, str]]) -> Path:
    labels = corner_labels(rows)
    read_energy = values(rows, "read_energy_pJ")
    write_energy = values(rows, "write_energy_pJ")
    x = list(range(len(rows)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar([i - width / 2 for i in x], read_energy, width, label="Read", color=COLORS["read"])
    ax.bar([i + width / 2 for i in x], write_energy, width, label="Write", color=COLORS["write"], alpha=0.86)
    ax.set_title("Cadence-Calibrated NVSim Energy by Corner")
    ax.set_ylabel("Dynamic energy per access (pJ)")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, ncols=2)
    style_axis(ax)
    annotate_bars(ax, [i - width / 2 for i in x], read_energy, fmt="{:.2f}")
    annotate_bars(ax, [i + width / 2 for i in x], write_energy, fmt="{:.2f}")
    return save(fig, "calibrated_energy_by_corner.png")


def plot_calibrated_leakage(rows: list[dict[str, str]]) -> Path:
    labels = corner_labels(rows)
    leakage = values(rows, "leakage_uW")
    x = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(x, leakage, color=COLORS["leakage"], width=0.58)
    ax.set_yscale("log")
    ax.set_title("Cadence-Calibrated NVSim Leakage by Corner")
    ax.set_ylabel("Leakage power (uW, log scale)")
    ax.set_xticks(x, labels)
    style_axis(ax, log=True)
    annotate_bars(ax, x, leakage, fmt="{:.3g}", log=True)
    return save(fig, "calibrated_leakage_by_corner_log.png")


def plot_latency_contributors(rows: list[dict[str, str]]) -> Path:
    labels = corner_labels(rows)
    total = values(rows, "read_lat_ps")
    rowdec = values(rows, "rowdec_lat_ps")
    bitline = values(rows, "bitline_lat_ps")
    other = [max(t - r - b, 0) for t, r, b in zip(total, rowdec, bitline)]
    x = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.bar(x, rowdec, label="Row decoder", color=COLORS["rowdec"], width=0.58)
    ax.bar(x, bitline, bottom=rowdec, label="Bitline", color=COLORS["bitline"], width=0.58)
    ax.bar(
        x,
        other,
        bottom=[r + b for r, b in zip(rowdec, bitline)],
        label="Other NVSim path",
        color=COLORS["other"],
        width=0.58,
    )
    ax.set_title("Calibrated Read Latency Contributors")
    ax.set_ylabel("Latency (ps)")
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, ncols=3, loc="upper left")
    style_axis(ax)
    annotate_bars(ax, x, total, fmt="{:.0f}", totals=True)
    return save(fig, "latency_contributors_by_corner.png")


def plot_array_vs_bitcell(bitcell_rows: list[dict[str, str]], calibrated_rows: list[dict[str, str]]) -> Path:
    labels = corner_labels(calibrated_rows)
    x = list(range(len(calibrated_rows)))
    width = 0.34

    bitcell_by_corner = {row["corner"]: row for row in bitcell_rows}
    cell_read = []
    cell_write = []
    cell_leakage_power = []
    array_read = []
    array_write = []
    array_leakage = []

    for row in calibrated_rows:
        corner = row["corner"]
        source = bitcell_by_corner[corner]
        vdd = float(source["vdd_V"])
        cell_read.append(float(source["read_energy_fJ"]) * ROW_BITS / 1000)
        cell_write.append(float(source["write_energy_fJ"]) * ROW_BITS / 1000)
        cell_leakage_current_ua = float(source["standby_leakage_nA"]) * NUM_BITS / 1000
        cell_leakage_power.append(cell_leakage_current_ua * vdd)
        array_read.append(float(row["read_energy_pJ"]))
        array_write.append(float(row["write_energy_pJ"]))
        array_leakage.append(float(row["leakage_uW"]))

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), constrained_layout=True)
    comparison_panel(
        axes[0],
        x,
        labels,
        cell_read,
        array_read,
        "Read Energy",
        "Energy (pJ, log scale)",
        width,
    )
    comparison_panel(
        axes[1],
        x,
        labels,
        cell_write,
        array_write,
        "Write Energy",
        "Energy (pJ, log scale)",
        width,
    )
    comparison_panel(
        axes[2],
        x,
        labels,
        cell_leakage_power,
        array_leakage,
        "Standby Leakage",
        "Power (uW, log scale)",
        width,
    )
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Bitcell Lower Bounds vs Calibrated 128x128 Array Estimates", y=1.04, fontsize=13)
    return save(fig, "array_vs_bitcell_energy_leakage.png")


def plot_roadmap_vs_calibrated(roadmap_rows: list[dict[str, str]], calibrated_rows: list[dict[str, str]]) -> Path:
    labels = [row["roadmap"] for row in roadmap_rows] + [row["corner"] for row in calibrated_rows]
    latency = values(roadmap_rows, "read_lat_ps") + values(calibrated_rows, "read_lat_ps")
    energy = values(roadmap_rows, "read_energy_pJ") + values(calibrated_rows, "read_energy_pJ")
    leakage = values(roadmap_rows, "leakage_uW") + values(calibrated_rows, "leakage_uW")
    colors = [COLORS["roadmap"]] * len(roadmap_rows) + [COLORS["calibrated"]] * len(calibrated_rows)
    x = list(range(len(labels)))

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), constrained_layout=True)
    roadmap_calibrated_panel(axes[0], x, labels, latency, colors, "Read Latency", "ps")
    roadmap_calibrated_panel(axes[1], x, labels, energy, colors, "Read Energy", "pJ")
    roadmap_calibrated_panel(axes[2], x, labels, leakage, colors, "Leakage Power", "uW", log=True)
    fig.suptitle("Built-In NVSim Roadmaps vs Cadence-Calibrated Corners", y=1.04, fontsize=13)
    return save(fig, "roadmap_vs_calibrated_summary.png")


def comparison_panel(
    ax: plt.Axes,
    x: list[int],
    labels: list[str],
    lower_bound: list[float],
    array_estimate: list[float],
    title: str,
    ylabel: str,
    width: float,
) -> None:
    ax.bar([i - width / 2 for i in x], lower_bound, width, label="Cadence bitcell row/cell-array", color=COLORS["bitcell"])
    ax.bar([i + width / 2 for i in x], array_estimate, width, label="Calibrated NVSim array", color=COLORS["array"], alpha=0.9)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, labels)
    style_axis(ax, log=True)


def roadmap_calibrated_panel(
    ax: plt.Axes,
    x: list[int],
    labels: list[str],
    values_to_plot: list[float],
    colors: list[str],
    title: str,
    ylabel: str,
    log: bool = False,
) -> None:
    ax.bar(x, values_to_plot, color=colors, width=0.62)
    if log:
        ax.set_yscale("log")
        ylabel = f"{ylabel} (log scale)"
    ax.axvline(2.5, color="#444444", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.text(1, ax.get_ylim()[1] * (0.75 if log else 0.9), "roadmap", ha="center", fontsize=8, color="#444444")
    ax.text(5, ax.get_ylim()[1] * (0.75 if log else 0.9), "calibrated", ha="center", fontsize=8, color="#444444")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    style_axis(ax, log=log)


def values(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def corner_labels(rows: list[dict[str, str]]) -> list[str]:
    return [CORNER_LABELS.get(row["corner"], row["corner"]) for row in rows]


def annotate_bars(
    ax: plt.Axes,
    x: list[float],
    heights: list[float],
    fmt: str,
    log: bool = False,
    totals: bool = False,
) -> None:
    max_height = max(heights)
    for xpos, height in zip(x, heights):
        if log:
            y = height * 1.16
        else:
            y = height + max_height * 0.025
        label = fmt.format(height)
        if totals:
            label = f"{label} ps"
        ax.text(xpos, y, label, ha="center", va="bottom", fontsize=8)
    if not log:
        ax.set_ylim(top=max_height * 1.16)


def style_axis(ax: plt.Axes, log: bool = False) -> None:
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if log:
        ax.grid(axis="y", which="minor", color="#eeeeee", linewidth=0.45, alpha=0.9)


def save(fig: plt.Figure, filename: str) -> Path:
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def write_index(paths: list[Path]) -> None:
    lines = [
        "# SRAM Array Report Figures",
        "",
        "Generated by `scripts/plot_array_figures.py` from `rw_metrics.csv` and the NVSim summaries.",
        "",
    ]
    for path in paths:
        lines.extend([f"## {path.stem.replace('_', ' ').title()}", "", f"![{path.stem}]({path.name})", ""])
    (FIGURE_DIR / "figure_index.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
