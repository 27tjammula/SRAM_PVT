#!/usr/bin/env python3
"""Generate SRAM plots in a folder tree that mirrors sim_data."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DATA_ROOT = REPO_ROOT / "sim_data"
PLOT_ROOT = REPO_ROOT / "sram_analysis_plots"

SNM_CURVE_COLORS = ("#1f77b4", "#d62728")
SIGNAL_COLORS = {
    "/Q": "#1f77b4",
    "/QB": "#d62728",
    "/WL": "#2ca02c",
    "/BL": "#9467bd",
    "/BLB": "#ff7f0e",
}
SIGNAL_ORDER = ("/Q", "/QB", "/WL", "/BL", "/BLB")
CURRENT_SIGNAL = "/V0/PLUS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SRAM plots under sram_analysis_plots."
    )
    parser.add_argument(
        "--sim-data-root",
        type=Path,
        default=SIM_DATA_ROOT,
        help="Root directory containing SRAM CSV exports.",
    )
    parser.add_argument(
        "--plot-root",
        type=Path,
        default=PLOT_ROOT,
        help="Output directory for generated plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_data_root = args.sim_data_root.resolve()
    plot_root = args.plot_root.resolve()

    csv_paths = sorted(path for path in sim_data_root.rglob("*.csv") if path.is_file())
    if not csv_paths:
        raise SystemExit(f"No CSV files found under {sim_data_root}")

    generated: list[Path] = []
    for csv_path in csv_paths:
        rel = csv_path.relative_to(sim_data_root)
        with csv_path.open(newline="") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)

        if len(header) == 4:
            output_path = plot_root / rel.with_suffix(".png")
            plot_snm_like_csv(csv_path, output_path, sim_data_root)
            generated.append(output_path)
        else:
            voltage_output = plot_root / rel.parent / f"{rel.stem}_voltages.png"
            current_output = plot_root / rel.parent / f"{rel.stem}_supply_current.png"
            plot_voltage_csv(csv_path, voltage_output, sim_data_root)
            plot_current_csv(csv_path, current_output, sim_data_root)
            generated.extend((voltage_output, current_output))

    print(f"Generated {len(generated)} plot files under {plot_root}")
    for path in generated:
        print(path.relative_to(REPO_ROOT))


def plot_snm_like_csv(csv_path: Path, output_path: Path, sim_data_root: Path) -> None:
    header, data = load_csv(csv_path)
    if data.shape[1] < 4:
        raise ValueError(f"{csv_path} does not have four columns for an SNM/WNM plot")

    x1, y1, x2, y2 = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    vdd = max(float(np.nanmax(data[:, :4])), 1e-3)
    margin = 0.05 * vdd

    fig, ax = plt.subplots(figsize=(6.0, 5.6), constrained_layout=True)
    ax.plot(x1, y1, lw=2.1, color=SNM_CURVE_COLORS[0], label=curve_label(header[0], "Curve A"))
    ax.plot(x2, y2, lw=2.1, color=SNM_CURVE_COLORS[1], label=curve_label(header[2], "Curve B"))
    ax.set_xlim(float(np.nanmin(data[:, :4])) - margin, float(np.nanmax(data[:, :4])) + margin)
    ax.set_ylim(float(np.nanmin(data[:, :4])) - margin, float(np.nanmax(data[:, :4])) + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(title_from_path(csv_path.relative_to(sim_data_root)))
    ax.legend(loc="best", fontsize=8, framealpha=0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_voltage_csv(csv_path: Path, output_path: Path, sim_data_root: Path) -> None:
    signals = load_waveform_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)

    plotted = False
    for signal_name in SIGNAL_ORDER:
        waveform = signals.get(signal_name)
        if waveform is None:
            continue
        ax.plot(
            waveform["time_ns"],
            waveform["value"],
            lw=1.8,
            label=signal_name[1:],
            color=SIGNAL_COLORS[signal_name],
        )
        plotted = True

    if not plotted:
        raise ValueError(f"{csv_path} has no recognized voltage signals")

    ax.grid(True, color="0.9", linewidth=0.8)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"{title_from_path(csv_path.relative_to(sim_data_root))} - Voltages")
    ax.legend(ncol=min(5, len(ax.lines)), fontsize=8, loc="best", framealpha=0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_current_csv(csv_path: Path, output_path: Path, sim_data_root: Path) -> None:
    signals = load_waveform_csv(csv_path)
    waveform = signals.get(CURRENT_SIGNAL)
    if waveform is None:
        raise ValueError(f"{csv_path} is missing {CURRENT_SIGNAL}")

    values = waveform["value"]
    scale, unit = current_scale(values)

    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    ax.plot(waveform["time_ns"], values * scale, lw=1.8, color="#444444")
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(f"I_VDD ({unit})")
    ax.set_title(f"{title_from_path(csv_path.relative_to(sim_data_root))} - Supply Current")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def load_csv(csv_path: Path) -> tuple[list[str], np.ndarray]:
    with csv_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]
    return header, np.asarray(rows, dtype=float)


def load_waveform_csv(csv_path: Path) -> dict[str, dict[str, np.ndarray]]:
    header, data = load_csv(csv_path)
    signals: dict[str, dict[str, np.ndarray]] = {}
    for index in range(0, len(header), 2):
        signal_name = signal_from_header(header[index])
        signals[signal_name] = {
            "time_ns": data[:, index] * 1e9,
            "value": data[:, index + 1],
        }
    return signals


def signal_from_header(header_cell: str) -> str:
    match = re.search(r"(/[^\s(,)\"]+)", header_cell)
    if not match:
        raise ValueError(f"Could not parse signal name from header cell {header_cell!r}")
    return match.group(1)


def curve_label(header_cell: str, fallback: str) -> str:
    try:
        return signal_from_header(header_cell)[1:]
    except ValueError:
        return fallback


def title_from_path(rel_path: Path) -> str:
    parts = [pretty_token(part) for part in rel_path.parts[:-1]]
    stem = pretty_token(rel_path.stem)
    return " / ".join([*parts, stem])


def pretty_token(token: str) -> str:
    token = token.replace(".csv", "")
    token = token.replace("_", " ")
    return token


def current_scale(values: np.ndarray) -> tuple[float, str]:
    peak = float(np.nanmax(np.abs(values)))
    if peak >= 1e-3:
        return 1e3, "mA"
    if peak >= 1e-6:
        return 1e6, "uA"
    if peak >= 1e-9:
        return 1e9, "nA"
    return 1.0, "A"


if __name__ == "__main__":
    main()
