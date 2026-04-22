#!/usr/bin/env python3
"""Plot SRAM SNM CSVs and extract read/write transient metrics."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

NS = 1e-9
PS = 1e-12

DEFAULT_STANDBY_WINDOW_NS = (0.2, 1.8)
DEFAULT_WRITE_WINDOW_NS = (2.995, 6.005)
DEFAULT_READ_BASELINE_WINDOW_NS = (12.5, 12.9)
DEFAULT_READ_WINDOW_NS = (12.995, 16.005)
READ_DELAY_THRESHOLD_V = 0.050

VOLTAGE_SIGNALS = ("/Q", "/QB", "/WL", "/BL", "/BLB")
REQUIRED_RW_SIGNALS = (*VOLTAGE_SIGNALS, "/VDD", "/V0/PLUS")


@dataclass(frozen=True)
class Curve:
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class SnmData:
    kind: str
    corner: str
    path: Path
    curve_a: Curve
    curve_b: Curve
    vdd: float


@dataclass(frozen=True)
class Waveform:
    time: np.ndarray
    value: np.ndarray


@dataclass(frozen=True)
class RwData:
    corner: str
    model: str
    path: Path
    signals: dict[str, Waveform]
    vdd: float


@dataclass(frozen=True)
class RwMetrics:
    corner: str
    model: str
    vdd_v: float
    write_delay_ps: float
    read_disturb_m_v: float
    standby_leakage_n_a: float
    write_energy_f_j: float
    read_energy_f_j: float
    read_delay_proxy_ps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SRAM hold/read SNM and read/write transient CSV exports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "sim_data",
        help="Directory containing holdSNM_*.csv, ReadSNM_*.csv, and rw*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "sram_analysis_plots",
        help="Directory for plots and metric tables.",
    )
    parser.add_argument(
        "--standby-window-ns",
        nargs=2,
        type=float,
        default=DEFAULT_STANDBY_WINDOW_NS,
        metavar=("START", "END"),
        help="Fixed quiet hold/leakage window in ns.",
    )
    parser.add_argument(
        "--write-window-ns",
        nargs=2,
        type=float,
        default=DEFAULT_WRITE_WINDOW_NS,
        metavar=("START", "END"),
        help="Fixed first WL pulse integration window in ns.",
    )
    parser.add_argument(
        "--read-baseline-window-ns",
        nargs=2,
        type=float,
        default=DEFAULT_READ_BASELINE_WINDOW_NS,
        metavar=("START", "END"),
        help="Fixed QB low-baseline window immediately before the read pulse in ns.",
    )
    parser.add_argument(
        "--read-window-ns",
        nargs=2,
        type=float,
        default=DEFAULT_READ_WINDOW_NS,
        metavar=("START", "END"),
        help="Fixed second WL pulse integration/disturb window in ns.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG output resolution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    hold_snm = [load_snm_csv(path, "hold") for path in sorted(input_dir.glob("holdSNM_*.csv"))]
    read_snm = [load_snm_csv(path, "read") for path in sorted(input_dir.glob("ReadSNM_*.csv"))]
    rw_data = [load_rw_csv(path) for path in sorted(input_dir.glob("rw*.csv"))]

    if hold_snm:
        plot_snm_family(hold_snm, args.output_dir, args.dpi)
    if read_snm:
        plot_snm_family(read_snm, args.output_dir, args.dpi)
    if not rw_data:
        raise SystemExit("No rw*.csv transient exports found.")

    windows = {
        "standby": ns_pair(args.standby_window_ns),
        "write": ns_pair(args.write_window_ns),
        "read_baseline": ns_pair(args.read_baseline_window_ns),
        "read": ns_pair(args.read_window_ns),
    }
    metrics = [extract_metrics(data, windows) for data in rw_data]

    write_metrics_csv(metrics, args.output_dir / "rw_metrics.csv")
    write_metrics_markdown(metrics, args.output_dir / "rw_metrics.md", windows)
    plot_rw_by_signal(rw_data, windows, args.output_dir, args.dpi)
    for data in rw_data:
        plot_rw_corner(data, windows, args.output_dir, args.dpi)

    print(format_metrics_table(metrics))
    print()
    print(f"Read inputs from {input_dir}")
    print(f"Wrote outputs to {args.output_dir}")


def ns_pair(values: tuple[float, float] | list[float]) -> tuple[float, float]:
    start, end = values
    if not start < end:
        raise ValueError(f"Invalid window: {start} ns to {end} ns")
    return start * NS, end * NS


def load_snm_csv(path: Path, kind: str) -> SnmData:
    with path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        rows = [[float(value) for value in row[:4]] for row in reader if row]

    if len(header) < 4 or not rows:
        raise ValueError(f"{path} does not look like a four-column SNM CSV export")

    data = np.asarray(rows, dtype=float)
    corner = snm_corner(path, ",".join(header))
    curve_a = Curve(data[:, 0], data[:, 1])
    curve_b = Curve(data[:, 2], data[:, 3])
    vdd = max(float(np.nanmax(data[:, 0])), float(np.nanmax(data[:, 1])), float(np.nanmax(data[:, 2])), float(np.nanmax(data[:, 3])))
    return SnmData(kind=kind, corner=corner, path=path, curve_a=curve_a, curve_b=curve_b, vdd=vdd)


def snm_corner(path: Path, header_text: str) -> str:
    model_match = re.search(r"gpdk045\.scs:([A-Za-z0-9_+-]+)", header_text)
    if model_match:
        return model_match.group(1)

    stem = path.stem
    prefix_match = re.match(r"(?:holdSNM|ReadSNM)[_-](.+)", stem, re.IGNORECASE)
    if prefix_match:
        return prefix_match.group(1)
    return stem


def load_rw_csv(path: Path) -> RwData:
    with path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]

    data = np.asarray(rows, dtype=float)
    signals: dict[str, Waveform] = {}
    for index in range(0, len(header), 2):
        signal_name = signal_from_header(header[index])
        signals[signal_name] = Waveform(data[:, index], data[:, index + 1])

    missing = [signal for signal in REQUIRED_RW_SIGNALS if signal not in signals]
    if missing:
        raise ValueError(f"{path} is missing required signal columns: {', '.join(missing)}")

    model = rw_model(",".join(header))
    corner = rw_corner(path)
    vdd = float(np.median(signals["/VDD"].value))
    return RwData(corner=corner, model=model, path=path, signals=signals, vdd=vdd)


def signal_from_header(header_cell: str) -> str:
    signal_match = re.search(r"(/[^\s(,]+)", header_cell)
    if not signal_match:
        raise ValueError(f"Could not extract signal name from header cell {header_cell!r}")
    return signal_match.group(1)


def rw_model(header_text: str) -> str:
    model_match = re.search(r"gpdk045\.scs:([A-Za-z0-9_+-]+)", header_text)
    return model_match.group(1) if model_match else "unknown"


def rw_corner(path: Path) -> str:
    stem = path.stem
    return stem[3:] if stem.startswith("rw_") else stem


def extract_metrics(data: RwData, windows: dict[str, tuple[float, float]]) -> RwMetrics:
    wl = data.signals["/WL"]
    q = data.signals["/Q"]
    qb = data.signals["/QB"]
    bl = data.signals["/BL"]
    blb = data.signals["/BLB"]
    vdd = data.signals["/VDD"]
    current = data.signals["/V0/PLUS"]

    threshold = 0.5 * data.vdd
    wl_rise_times = crossing_times(wl.time, wl.value, threshold, "rise")
    if len(wl_rise_times) < 2:
        raise ValueError(f"{data.path} does not contain two rising WL crossings")

    q_rise_time = first_crossing_after(q.time, q.value, threshold, "rise", wl_rise_times[0])
    if math.isnan(q_rise_time):
        raise ValueError(f"{data.path} has no rising Q crossing after the first WL edge")

    read_delay_proxy = read_delay(bl, blb, wl_rise_times[1], READ_DELAY_THRESHOLD_V, windows["read"][1])

    qb_baseline = window_mean(qb.time, qb.value, *windows["read_baseline"])
    _, qb_read = window_values(qb.time, qb.value, *windows["read"])
    read_disturb = max(0.0, float(np.max(qb_read)) - qb_baseline)

    standby_leakage = window_mean_abs(current.time, current.value, *windows["standby"])
    write_energy = integrate_power(vdd, current, *windows["write"])
    read_energy = integrate_power(vdd, current, *windows["read"])

    return RwMetrics(
        corner=data.corner,
        model=data.model,
        vdd_v=data.vdd,
        write_delay_ps=(q_rise_time - wl_rise_times[0]) / PS,
        read_disturb_m_v=read_disturb * 1e3,
        standby_leakage_n_a=standby_leakage * 1e9,
        write_energy_f_j=write_energy * 1e15,
        read_energy_f_j=read_energy * 1e15,
        read_delay_proxy_ps=(read_delay_proxy - wl_rise_times[1]) / PS if not math.isnan(read_delay_proxy) else math.nan,
    )


def crossing_times(time: np.ndarray, value: np.ndarray, threshold: float, edge: str) -> list[float]:
    crossings: list[float] = []
    for index in range(len(time) - 1):
        t0, t1 = time[index], time[index + 1]
        v0, v1 = value[index], value[index + 1]
        if v1 == v0:
            continue
        if edge == "rise" and v0 < threshold <= v1:
            fraction = (threshold - v0) / (v1 - v0)
            crossings.append(float(t0 + fraction * (t1 - t0)))
        elif edge == "fall" and v0 > threshold >= v1:
            fraction = (threshold - v0) / (v1 - v0)
            crossings.append(float(t0 + fraction * (t1 - t0)))
    return crossings


def first_crossing_after(
    time: np.ndarray,
    value: np.ndarray,
    threshold: float,
    edge: str,
    after_time: float,
) -> float:
    for crossing in crossing_times(time, value, threshold, edge):
        if crossing >= after_time:
            return crossing
    return math.nan


def read_delay(bl: Waveform, blb: Waveform, start: float, threshold: float, stop: float) -> float:
    time = merged_time_grid((bl, blb), start, stop)
    diff = np.abs(np.interp(time, bl.time, bl.value) - np.interp(time, blb.time, blb.value))

    if diff[0] >= threshold:
        return float(time[0])

    for index in range(len(time) - 1):
        d0, d1 = diff[index], diff[index + 1]
        if d0 < threshold <= d1 and d1 != d0:
            fraction = (threshold - d0) / (d1 - d0)
            return float(time[index] + fraction * (time[index + 1] - time[index]))
    return math.nan


def merged_time_grid(waveforms: tuple[Waveform, ...], start: float, end: float) -> np.ndarray:
    pieces = [np.asarray([start, end], dtype=float)]
    for waveform in waveforms:
        mask = (waveform.time > start) & (waveform.time < end)
        pieces.append(waveform.time[mask])
    return np.unique(np.concatenate(pieces))


def window_values(time: np.ndarray, value: np.ndarray, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    if start < time[0] or end > time[-1]:
        raise ValueError(f"Window {start / NS:.3f}-{end / NS:.3f} ns is outside waveform range")
    mask = (time > start) & (time < end)
    grid = np.unique(np.concatenate(([start], time[mask], [end])))
    return grid, np.interp(grid, time, value)


def window_mean(time: np.ndarray, value: np.ndarray, start: float, end: float) -> float:
    grid, values = window_values(time, value, start, end)
    return float(np.trapezoid(values, grid) / (end - start))


def window_mean_abs(time: np.ndarray, value: np.ndarray, start: float, end: float) -> float:
    grid, values = window_values(time, value, start, end)
    return float(np.trapezoid(np.abs(values), grid) / (end - start))


def integrate_power(vdd: Waveform, current: Waveform, start: float, end: float) -> float:
    time = merged_time_grid((vdd, current), start, end)
    voltage = np.interp(time, vdd.time, vdd.value)
    current_abs = np.abs(np.interp(time, current.time, current.value))
    return float(np.trapezoid(voltage * current_abs, time))


def plot_snm_family(snm_data: list[SnmData], output_dir: Path, dpi: int) -> None:
    kind = snm_data[0].kind
    for data in snm_data:
        fig, ax = plt.subplots(figsize=(6.2, 5.7), constrained_layout=True)
        draw_snm(ax, data)
        ax.set_title(f"{kind.capitalize()} SNM Butterfly - {data.corner.upper()}")
        fig.savefig(output_dir / f"{kind}_snm_butterfly_{data.corner}.png", dpi=dpi)
        plt.close(fig)

    fig, axes = plt.subplots(1, len(snm_data), figsize=(5.1 * len(snm_data), 4.8), constrained_layout=True)
    if len(snm_data) == 1:
        axes = [axes]
    for ax, data in zip(axes, snm_data):
        draw_snm(ax, data)
        ax.set_title(data.corner.upper())
    fig.savefig(output_dir / f"{kind}_snm_butterfly_all_corners.png", dpi=dpi)
    plt.close(fig)


def draw_snm(ax: plt.Axes, data: SnmData) -> None:
    ax.plot(data.curve_a.x, data.curve_a.y, lw=2.1, color="#1f77b4", label="QB vs Q")
    ax.plot(data.curve_b.x, data.curve_b.y, lw=2.1, color="#d62728", label="Q vs QB")
    margin = 0.05 * data.vdd
    ax.set_xlim(-margin, data.vdd + margin)
    ax.set_ylim(-margin, data.vdd + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Voltage (V)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)


def plot_rw_by_signal(
    rw_data: list[RwData],
    windows: dict[str, tuple[float, float]],
    output_dir: Path,
    dpi: int,
) -> None:
    signal_rows = ["/Q", "/QB", "/WL", "/BL", "/BLB", "/V0/PLUS"]
    fig, axes = plt.subplots(len(signal_rows), 1, figsize=(11.5, 11.0), sharex=True, constrained_layout=True)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for row, signal in enumerate(signal_rows):
        ax = axes[row]
        shade_windows(ax, windows)
        for index, data in enumerate(rw_data):
            waveform = data.signals[signal]
            scale = 1e6 if signal == "/V0/PLUS" else 1.0
            label = f"{data.corner} ({data.vdd:.1f} V)"
            ax.plot(waveform.time / NS, waveform.value * scale, lw=1.6, color=color_cycle[index % len(color_cycle)], label=label)
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.set_ylabel("I_VDD (uA)" if signal == "/V0/PLUS" else f"{signal[1:]} (V)")
        if row == 0:
            ax.legend(ncol=min(len(rw_data), 4), fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time (ns)")
    fig.suptitle("Read/Write Transient Overlay by Signal")
    fig.savefig(output_dir / "rw_transient_overlay_all_corners.png", dpi=dpi)
    plt.close(fig)


def plot_rw_corner(
    data: RwData,
    windows: dict[str, tuple[float, float]],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.8), sharex=True, constrained_layout=True, height_ratios=[3, 1])
    for ax in axes:
        shade_windows(ax, windows)

    for signal in VOLTAGE_SIGNALS:
        waveform = data.signals[signal]
        axes[0].plot(waveform.time / NS, waveform.value, lw=1.7, label=signal[1:])
    axes[0].set_ylabel("Voltage (V)")
    axes[0].grid(True, color="0.9", linewidth=0.8)
    axes[0].legend(ncol=5, fontsize=8, loc="upper right")

    current = data.signals["/V0/PLUS"]
    axes[1].plot(current.time / NS, current.value * 1e6, lw=1.5, color="#444444")
    axes[1].set_ylabel("I_VDD (uA)")
    axes[1].set_xlabel("Time (ns)")
    axes[1].grid(True, color="0.9", linewidth=0.8)

    fig.suptitle(f"Read/Write Transient - {data.corner} ({data.vdd:.1f} V)")
    fig.savefig(output_dir / f"rw_transient_{data.corner}.png", dpi=dpi)
    plt.close(fig)


def shade_windows(ax: plt.Axes, windows: dict[str, tuple[float, float]]) -> None:
    spans = [
        ("standby", "#7f7f7f", 0.08),
        ("write", "#2ca02c", 0.08),
        ("read", "#ff7f0e", 0.08),
    ]
    for key, color, alpha in spans:
        start, end = windows[key]
        ax.axvspan(start / NS, end / NS, color=color, alpha=alpha, lw=0)


def write_metrics_csv(metrics: list[RwMetrics], path: Path) -> None:
    rows = [metric_row(metric) for metric in metrics]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_markdown(metrics: list[RwMetrics], path: Path, windows: dict[str, tuple[float, float]]) -> None:
    lines = [
        "# SRAM Read/Write Metrics",
        "",
        "Fixed windows:",
        f"- standby: {windows['standby'][0] / NS:.3f} ns to {windows['standby'][1] / NS:.3f} ns",
        f"- write energy: {windows['write'][0] / NS:.3f} ns to {windows['write'][1] / NS:.3f} ns",
        f"- read baseline: {windows['read_baseline'][0] / NS:.3f} ns to {windows['read_baseline'][1] / NS:.3f} ns",
        f"- read energy/disturb: {windows['read'][0] / NS:.3f} ns to {windows['read'][1] / NS:.3f} ns",
        "",
        format_metrics_table(metrics),
        "",
    ]
    path.write_text("\n".join(lines))


def metric_row(metric: RwMetrics) -> dict[str, str]:
    return {
        "corner": metric.corner,
        "model": metric.model,
        "vdd_V": f"{metric.vdd_v:.6g}",
        "write_delay_ps": f"{metric.write_delay_ps:.3f}",
        "read_disturb_mV": f"{metric.read_disturb_m_v:.3f}",
        "standby_leakage_nA": f"{metric.standby_leakage_n_a:.6f}",
        "write_energy_fJ": f"{metric.write_energy_f_j:.3f}",
        "read_energy_fJ": f"{metric.read_energy_f_j:.3f}",
        "read_delay_proxy_ps": format_nan(metric.read_delay_proxy_ps, ".3f"),
    }


def format_metrics_table(metrics: list[RwMetrics]) -> str:
    rows = [metric_row(metric) for metric in metrics]
    headers = list(rows[0].keys())
    widths = {header: max(len(header), *(len(row[header]) for row in rows)) for header in headers}
    header_line = "| " + " | ".join(header.ljust(widths[header]) for header in headers) + " |"
    separator = "| " + " | ".join("-" * widths[header] for header in headers) + " |"
    body = [
        "| " + " | ".join(row[header].ljust(widths[header]) for header in headers) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def format_nan(value: float, spec: str) -> str:
    return "N/A" if math.isnan(value) else format(value, spec)


if __name__ == "__main__":
    main()
