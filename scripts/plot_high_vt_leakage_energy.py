#!/usr/bin/env python3
"""Plot baseline vs high-Vt hold-window supply energy from transient CSVs."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DATA_ROOT = REPO_ROOT / "sim_data"
FIGURE_DIR = REPO_ROOT / "sram_analysis_plots" / "report_figures"

CURRENT_SIGNAL = "/V0/PLUS"
VDD_SIGNAL = "/VDD"

CASE_ORDER = (
    {
        "key": "ff",
        "label": "FF\n1.2 V",
        "baseline": Path("baseline/rwtrans/rw_ff.csv"),
        "high_vt": Path("optimized/high_vt/trans_vt_opt_ff.csv"),
    },
    {
        "key": "tt",
        "label": "TT\n1.0 V",
        "baseline": Path("baseline/rwtrans/rw_tt.csv"),
        "high_vt": Path("optimized/high_vt/trans_vt_opt_tt.csv"),
    },
    {
        "key": "ss_1V",
        "label": "SS\n1.0 V",
        "baseline": Path("baseline/rwtrans/rw_ss_1.csv"),
        "high_vt": Path("optimized/high_vt/trans_vt_opt_ss_1V.csv"),
    },
    {
        "key": "ss_0.8V",
        "label": "SS\n0.8 V",
        "baseline": Path("baseline/rwtrans/rw_ss_08.csv"),
        "high_vt": Path("optimized/high_vt/trans_vt_opt_ss_0.8V.csv"),
    },
)

BASELINE_COLOR = "#2f6fbb"
HIGH_VT_COLOR = "#d95f02"
DELTA_BETTER_COLOR = "#2f7f4f"
DELTA_WORSE_COLOR = "#b23a48"

FIGSIZE = (8.2, 4.8)
BAR_WIDTH = 0.34
HEADROOM_FACTOR = 1.30
DPI = 300


@dataclass(frozen=True)
class EnergyPoint:
    energy_j: float
    avg_current_a: float
    avg_vdd_v: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline vs high-Vt hold-window energy comparison."
    )
    parser.add_argument(
        "--sim-data-root",
        type=Path,
        default=SIM_DATA_ROOT,
        help="Root directory containing the transient CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Directory for the generated figure and CSV summary.",
    )
    parser.add_argument(
        "--window-ns",
        nargs=2,
        type=float,
        default=(6.0, 9.0),
        metavar=("START", "END"),
        help="Hold-window integration range in ns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start_ns, t_end_ns = args.window_ns
    if not t_start_ns < t_end_ns:
        raise SystemExit("Window start must be strictly less than window end.")

    rows = []
    for case in CASE_ORDER:
        baseline = integrate_case(args.sim_data_root / case["baseline"], t_start_ns, t_end_ns)
        high_vt = integrate_case(args.sim_data_root / case["high_vt"], t_start_ns, t_end_ns)
        rows.append(
            {
                "corner": case["key"],
                "label": case["label"],
                "baseline": baseline,
                "high_vt": high_vt,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "high_vt_hold_leakage_energy_by_corner.csv"
    fig_path = args.output_dir / "high_vt_hold_leakage_energy_by_corner.png"

    write_summary_csv(csv_path, rows, t_start_ns, t_end_ns)
    plot_energy_bars(fig_path, rows, t_start_ns, t_end_ns)

    print(csv_path.relative_to(REPO_ROOT))
    print(fig_path.relative_to(REPO_ROOT))
    print(fig_path.with_suffix(".pdf").relative_to(REPO_ROOT))


def integrate_case(csv_path: Path, t_start_ns: float, t_end_ns: float) -> EnergyPoint:
    signals = load_waveform_csv(csv_path)
    current = signals[CURRENT_SIGNAL]
    vdd = signals[VDD_SIGNAL]
    window_s = (t_end_ns - t_start_ns) * 1e-9

    charge_c = integrate_window_ns(current["time_ns"], current["value"], t_start_ns, t_end_ns)
    vdd_integral = integrate_window_ns(vdd["time_ns"], vdd["value"], t_start_ns, t_end_ns)
    avg_vdd_v = vdd_integral / window_s
    energy_j = -avg_vdd_v * charge_c
    avg_current_a = -charge_c / window_s

    return EnergyPoint(
        energy_j=energy_j,
        avg_current_a=avg_current_a,
        avg_vdd_v=avg_vdd_v,
    )


def write_summary_csv(
    csv_path: Path,
    rows: list[dict[str, object]],
    t_start_ns: float,
    t_end_ns: float,
) -> None:
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "corner",
                "label",
                "window_start_ns",
                "window_end_ns",
                "baseline_energy_aJ",
                "high_vt_energy_aJ",
                "delta_aJ",
                "delta_percent",
                "baseline_avg_current_nA",
                "high_vt_avg_current_nA",
                "baseline_avg_vdd_V",
                "high_vt_avg_vdd_V",
            ]
        )
        for row in rows:
            baseline: EnergyPoint = row["baseline"]  # type: ignore[assignment]
            high_vt: EnergyPoint = row["high_vt"]  # type: ignore[assignment]
            delta_j = high_vt.energy_j - baseline.energy_j
            delta_pct = 100.0 * delta_j / baseline.energy_j
            writer.writerow(
                [
                    row["corner"],
                    row["label"],
                    f"{t_start_ns:.3f}",
                    f"{t_end_ns:.3f}",
                    f"{baseline.energy_j * 1e18:.6f}",
                    f"{high_vt.energy_j * 1e18:.6f}",
                    f"{delta_j * 1e18:.6f}",
                    f"{delta_pct:.6f}",
                    f"{baseline.avg_current_a * 1e9:.6f}",
                    f"{high_vt.avg_current_a * 1e9:.6f}",
                    f"{baseline.avg_vdd_v:.6f}",
                    f"{high_vt.avg_vdd_v:.6f}",
                ]
            )


def plot_energy_bars(
    output_path: Path,
    rows: list[dict[str, object]],
    t_start_ns: float,
    t_end_ns: float,
) -> None:
    labels = [str(row["label"]) for row in rows]
    baseline_aj = [row["baseline"].energy_j * 1e18 for row in rows]  # type: ignore[union-attr]
    high_vt_aj = [row["high_vt"].energy_j * 1e18 for row in rows]  # type: ignore[union-attr]
    plot_high_vt_grouped_bars(
        output_path=output_path,
        labels=labels,
        baseline_values=baseline_aj,
        high_vt_values=high_vt_aj,
        y_label="Consumed supply energy (aJ)",
        higher_is_better=False,
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

    Shared by the hold-window energy and read-SNM per-corner charts so the
    two figures in the report's High-Vt section are visually identical.
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
    """Match the project paper-style rules: no grid, no minor ticks, inward ticks all sides, no title."""
    ax.tick_params(direction="in", top=True, right=True, length=4.0, width=0.8)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


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


def integrate_window_ns(
    time_ns: np.ndarray,
    values: np.ndarray,
    t_start_ns: float,
    t_end_ns: float,
) -> float:
    time_s = np.asarray(time_ns, dtype=float) * 1e-9
    values = np.asarray(values, dtype=float)
    t_start_s = t_start_ns * 1e-9
    t_end_s = t_end_ns * 1e-9

    mask = (time_s >= t_start_s) & (time_s <= t_end_s)
    time_sel = time_s[mask]
    value_sel = values[mask]
    if time_sel.size == 0:
        return 0.0

    if time_sel[0] > t_start_s:
        start_value = np.interp(t_start_s, time_s, values)
        time_sel = np.insert(time_sel, 0, t_start_s)
        value_sel = np.insert(value_sel, 0, start_value)
    else:
        time_sel[0] = t_start_s
        value_sel[0] = np.interp(t_start_s, time_s, values)

    if time_sel[-1] < t_end_s:
        end_value = np.interp(t_end_s, time_s, values)
        time_sel = np.append(time_sel, t_end_s)
        value_sel = np.append(value_sel, end_value)
    else:
        time_sel[-1] = t_end_s
        value_sel[-1] = np.interp(t_end_s, time_s, values)

    try:
        return float(np.trapezoid(value_sel, time_sel))
    except AttributeError:
        return float(np.trapz(value_sel, time_sel))


def load_waveform_csv(csv_path: Path) -> dict[str, dict[str, np.ndarray]]:
    with csv_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]

    data = np.asarray(rows, dtype=float)
    signals: dict[str, dict[str, np.ndarray]] = {}
    for index in range(0, len(header), 2):
        signal_name = signal_from_header(header[index])
        signals[signal_name] = {
            "time_ns": data[:, index] * 1e9,
            "value": data[:, index + 1],
        }

    for required in (CURRENT_SIGNAL, VDD_SIGNAL):
        if required not in signals:
            raise ValueError(f"{csv_path} is missing {required}")
    return signals


def signal_from_header(header_cell: str) -> str:
    match = re.search(r"(/[^\s(,\"]+)", header_cell)
    if not match:
        raise ValueError(f"Could not parse signal name from header cell {header_cell!r}")
    return match.group(1)


if __name__ == "__main__":
    main()
