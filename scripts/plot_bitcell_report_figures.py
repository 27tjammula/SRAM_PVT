#!/usr/bin/env python3
"""Generate report-ready bitcell figures from the raw simulation CSVs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

from generate_structured_sram_plots import (
    Curve2D,
    diagonal_square_between_curves,
    diagonal_square_from_crossings,
    largest_square_between_curves,
    load_csv,
    load_waveform_csv,
    wnm_curves_from_csv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DATA_ROOT = REPO_ROOT / "sim_data"
OUTPUT_ROOT = REPO_ROOT / "sram_analysis_plots" / "report_figures" / "bitcell"

PNG_DPI = 400
BUTTERFLY_FIGSIZE = (6.6, 5.9)
BAR_FIGSIZE = (6.8, 4.2)
GRID_FIGSIZE = (8.0, 5.6)
AXIS_LIMIT = 1.25
AXIS_TICKS = np.arange(0.0, 1.21, 0.2)

BASELINE_COLOR = "#4c72b0"
HIGH_VT_COLOR = "#dd8452"
NEGATIVE_BITLINE_COLOR = "#55a868"
WORDLINE_UNDERDRIVE_COLOR = "#8172b2"
CORNER_COLORS = {
    "ff": "#d55e00",
    "tt": "#0072b2",
    "ss": "#009e73",
}
TRACE_COLORS = {
    "wl": "#2ca02c",
    "bl": "#9467bd",
    "q": "#1f77b4",
}

THREE_CORNER_CASES = (
    {"key": "ff", "label": "FF 1.2 V"},
    {"key": "tt", "label": "TT 1.0 V"},
    {"key": "ss", "label": "SS 1.0 V"},
)
FOUR_CORNER_CASES = (
    {"key": "ff", "label": "FF\n1.2 V", "vdd": 1.2},
    {"key": "tt", "label": "TT\n1.0 V", "vdd": 1.0},
    {"key": "ss_1V", "label": "SS\n1.0 V", "vdd": 1.0},
    {"key": "ss_0.8V", "label": "SS\n0.8 V", "vdd": 0.8},
)


@dataclass(frozen=True)
class ButterflyOverlayCase:
    key: str
    label: str
    curve_a: Curve2D
    curve_b: Curve2D
    margin_mv: float
    square_xy: np.ndarray


@dataclass(frozen=True)
class DelayMeasurement:
    label: str
    vdd: float
    baseline_delay_ps: float
    optimized_delay_ps: float
    time_ns: np.ndarray
    wl_v: np.ndarray
    bl_v: np.ndarray
    q_v: np.ndarray
    wl_cross_ns: float
    q_cross_ns: float


@dataclass(frozen=True)
class GuideEntry:
    relative_png: Path
    description: str
    source_files: list[Path]
    processing_steps: list[str]
    takeaway: str
    caveats: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paper-style bitcell report figures from raw SRAM CSV exports."
    )
    parser.add_argument(
        "--sim-data-root",
        type=Path,
        default=SIM_DATA_ROOT,
        help="Root directory containing raw SRAM CSV exports.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory for generated bitcell report figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_data_root = args.sim_data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    guide_entries: list[GuideEntry] = []

    baseline_dir = output_root / "baseline"
    high_vt_dir = output_root / "high_vt"
    negative_bitline_dir = output_root / "negative_bitline"
    wordline_dir = output_root / "wordline_underdrive"
    for path in (baseline_dir, high_vt_dir, negative_bitline_dir, wordline_dir):
        path.mkdir(parents=True, exist_ok=True)

    guide_entries.append(
        plot_butterfly_overlay(
            baseline_dir / "baseline_hold_snm_all_corners",
            [
                load_snm_overlay_case(sim_data_root / "baseline" / "hold" / "holdSNM_ff.csv", "ff", "FF 1.2 V"),
                load_snm_overlay_case(sim_data_root / "baseline" / "hold" / "holdSNM_tt.csv", "tt", "TT 1.0 V"),
                load_snm_overlay_case(sim_data_root / "baseline" / "hold" / "holdSNM_ss.csv", "ss", "SS 1.0 V"),
            ],
            "Hold-state butterfly curves with the three baseline process corners overlaid on one axis.",
            [
                sim_data_root / "baseline" / "hold" / "holdSNM_ff.csv",
                sim_data_root / "baseline" / "hold" / "holdSNM_tt.csv",
                sim_data_root / "baseline" / "hold" / "holdSNM_ss.csv",
            ],
            [
                "Loaded the raw four-column butterfly CSVs.",
                "Computed the limiting SNM square for each corner with the same largest-square routine used by the per-corner plot generator.",
                "Overlaid all three curve pairs and square outlines on shared 0 to 1.25 V axes.",
            ],
            "How hold stability changes across FF, TT, and SS baseline corners.",
            [
                "The three corners use different VDD values, so the larger FF envelope partly reflects its 1.2 V supply.",
                "Squares are shown as outlines only to keep the overlay readable.",
            ],
        )
    )
    guide_entries.append(
        plot_butterfly_overlay(
            baseline_dir / "baseline_read_snm_all_corners",
            [
                load_snm_overlay_case(sim_data_root / "baseline" / "read" / "ReadSNM_ff.csv", "ff", "FF 1.2 V"),
                load_snm_overlay_case(sim_data_root / "baseline" / "read" / "ReadSNM_tt.csv", "tt", "TT 1.0 V"),
                load_snm_overlay_case(sim_data_root / "baseline" / "read" / "ReadSNM_ss.csv", "ss", "SS 1.0 V"),
            ],
            "Read-bias butterfly curves with the three baseline process corners overlaid on one axis.",
            [
                sim_data_root / "baseline" / "read" / "ReadSNM_ff.csv",
                sim_data_root / "baseline" / "read" / "ReadSNM_tt.csv",
                sim_data_root / "baseline" / "read" / "ReadSNM_ss.csv",
            ],
            [
                "Loaded the raw four-column read-SNM CSVs.",
                "Computed the limiting eye for each corner using the shared SNM square-fit helper.",
                "Used the same axis limits and corner colors as the hold-SNM overlay for direct comparison.",
            ],
            "How baseline read stability shifts across process corners under an asserted wordline read condition.",
            [
                "Only the limiting eye is reported for each corner, so the legend margin values summarize the smallest valid square.",
            ],
        )
    )
    guide_entries.append(
        plot_butterfly_overlay(
            baseline_dir / "baseline_write_nm_all_corners",
            [
                load_wnm_overlay_case(sim_data_root / "baseline" / "write" / "writenm_ff.csv", "ff", "FF 1.2 V"),
                load_wnm_overlay_case(sim_data_root / "baseline" / "write" / "writenm_tt.csv", "tt", "TT 1.0 V"),
                load_wnm_overlay_case(sim_data_root / "baseline" / "write" / "writenm_ss.csv", "ss", "SS 1.0 V"),
            ],
            "Write-noise-margin butterflies with all three baseline corners overlaid on one axis.",
            [
                sim_data_root / "baseline" / "write" / "writenm_ff.csv",
                sim_data_root / "baseline" / "write" / "writenm_tt.csv",
                sim_data_root / "baseline" / "write" / "writenm_ss.csv",
            ],
            [
                "Mapped the raw write CSVs into the common butterfly plane using the existing Q/QB sweep conversion helper.",
                "Placed each WNM square with its diagonal corners on the actual V1 equals V2 curve crossings.",
                "Used shared axis limits so the corner-to-corner writability trend is visually comparable.",
            ],
            "Which baseline corners have the strongest or weakest static writability margin.",
            [
                "The WNM construction follows the current project convention where the square diagonal is constrained to the V1 equals V2 axis.",
            ],
        )
    )

    guide_entries.append(
        plot_butterfly_overlay(
            high_vt_dir / "high_vt_hold_snm_all_corners",
            [
                load_snm_overlay_case(sim_data_root / "optimized" / "high_vt" / "hold_snm_opt_vt_ff.csv", "ff", "FF 1.2 V"),
                load_snm_overlay_case(sim_data_root / "optimized" / "high_vt" / "hold_snm_opt_vt_tt.csv", "tt", "TT 1.0 V"),
                load_snm_overlay_case(sim_data_root / "optimized" / "high_vt" / "hold_snm_opt_vt_ss.csv", "ss", "SS 1.0 V"),
            ],
            "Hold-state butterfly curves for the high-Vt bitcell across the three process corners.",
            [
                sim_data_root / "optimized" / "high_vt" / "hold_snm_opt_vt_ff.csv",
                sim_data_root / "optimized" / "high_vt" / "hold_snm_opt_vt_tt.csv",
                sim_data_root / "optimized" / "high_vt" / "hold_snm_opt_vt_ss.csv",
            ],
            [
                "Loaded the raw optimized hold-SNM CSVs.",
                "Computed the limiting SNM square for each corner using the same fit routine as the baseline hold plot.",
                "Kept colors and axis limits identical to the baseline hold overlay.",
            ],
            "Whether the high-Vt device choice changes hold stability differently by process corner.",
            [
                "This figure isolates the high-Vt cell only; it does not overlay baseline traces directly.",
            ],
        )
    )
    guide_entries.append(
        plot_butterfly_overlay(
            high_vt_dir / "high_vt_read_snm_all_corners",
            [
                load_snm_overlay_case(sim_data_root / "optimized" / "high_vt" / "opt_Vt_readSNM_ff.csv", "ff", "FF 1.2 V"),
                load_snm_overlay_case(sim_data_root / "optimized" / "high_vt" / "opt_Vt_readSNM_tt.csv", "tt", "TT 1.0 V"),
                load_snm_overlay_case(sim_data_root / "optimized" / "high_vt" / "opt_Vt_readSNM_ss.csv", "ss", "SS 1.0 V"),
            ],
            "Read-bias butterfly curves for the high-Vt bitcell across the three process corners.",
            [
                sim_data_root / "optimized" / "high_vt" / "opt_Vt_readSNM_ff.csv",
                sim_data_root / "optimized" / "high_vt" / "opt_Vt_readSNM_tt.csv",
                sim_data_root / "optimized" / "high_vt" / "opt_Vt_readSNM_ss.csv",
            ],
            [
                "Loaded the raw optimized read-SNM CSVs.",
                "Computed the limiting read SNM square for each corner using the shared largest-square routine.",
                "Used the same corner colors and axis limits as the baseline read-SNM overlay.",
            ],
            "How the high-Vt choice changes read stability across the same process sweep.",
            [
                "The figure reports only the optimized bitcell; use the separate baseline overlay for direct shape comparison.",
            ],
        )
    )
    guide_entries.append(
        plot_two_condition_bars(
            high_vt_dir / "high_vt_hold_leakage_energy_by_corner",
            [case["label"] for case in FOUR_CORNER_CASES],
            [
                integrate_supply_energy(sim_data_root / "baseline" / "rwtrans" / "rw_ff.csv", (6.0, 9.0)),
                integrate_supply_energy(sim_data_root / "baseline" / "rwtrans" / "rw_tt.csv", (6.0, 9.0)),
                integrate_supply_energy(sim_data_root / "baseline" / "rwtrans" / "rw_ss_1.csv", (6.0, 9.0)),
                integrate_supply_energy(sim_data_root / "baseline" / "rwtrans" / "rw_ss_08.csv", (6.0, 9.0)),
            ],
            [
                integrate_supply_energy(sim_data_root / "optimized" / "high_vt" / "trans_vt_opt_ff.csv", (6.0, 9.0)),
                integrate_supply_energy(sim_data_root / "optimized" / "high_vt" / "trans_vt_opt_tt.csv", (6.0, 9.0)),
                integrate_supply_energy(sim_data_root / "optimized" / "high_vt" / "trans_vt_opt_ss_1V.csv", (6.0, 9.0)),
                integrate_supply_energy(sim_data_root / "optimized" / "high_vt" / "trans_vt_opt_ss_0.8V.csv", (6.0, 9.0)),
            ],
            "Consumed supply energy (aJ)",
            "Baseline",
            "High-Vt",
            BASELINE_COLOR,
            HIGH_VT_COLOR,
            value_formatter="{:.1f}",
            y_formatter="%.0f",
            delta_mode="percent",
            source_files=[
                sim_data_root / "baseline" / "rwtrans" / "rw_ff.csv",
                sim_data_root / "baseline" / "rwtrans" / "rw_tt.csv",
                sim_data_root / "baseline" / "rwtrans" / "rw_ss_1.csv",
                sim_data_root / "baseline" / "rwtrans" / "rw_ss_08.csv",
                sim_data_root / "optimized" / "high_vt" / "trans_vt_opt_ff.csv",
                sim_data_root / "optimized" / "high_vt" / "trans_vt_opt_tt.csv",
                sim_data_root / "optimized" / "high_vt" / "trans_vt_opt_ss_1V.csv",
                sim_data_root / "optimized" / "high_vt" / "trans_vt_opt_ss_0.8V.csv",
            ],
            description="Baseline versus high-Vt hold-window supply energy extracted from the 6 to 9 ns transient segment.",
            processing_steps=[
                "Loaded the raw /V0/PLUS and /VDD traces from each transient CSV.",
                "Integrated current over the fixed 6 to 9 ns hold window with trapezoidal integration and multiplied by the average VDD over that same window.",
                "Reported consumed energy as negative average VDD times the signed supply-current integral.",
            ],
            takeaway="Which corners show a hold-window energy reduction or penalty after switching to the high-Vt bitcell.",
            caveats=[
                "The 6 to 9 ns window includes brief post-write settling, so this is hold-window supply energy rather than perfectly quiescent leakage alone.",
            ],
        )
    )

    negbl_measurements = [
        measure_negative_bitline_delay(
            "FF\n1.2 V",
            1.2,
            sim_data_root / "baseline" / "rwtrans" / "rw_ff.csv",
            sim_data_root / "optimized" / "negative_bitline" / "trans_negBLopt_ff.csv",
        ),
        measure_negative_bitline_delay(
            "TT\n1.0 V",
            1.0,
            sim_data_root / "baseline" / "rwtrans" / "rw_tt.csv",
            sim_data_root / "optimized" / "negative_bitline" / "trans_negBLopt_tt.csv",
        ),
        measure_negative_bitline_delay(
            "SS\n1.0 V",
            1.0,
            sim_data_root / "baseline" / "rwtrans" / "rw_ss_1.csv",
            sim_data_root / "optimized" / "negative_bitline" / "trans_negBLopt_ss_1V.csv",
        ),
        measure_negative_bitline_delay(
            "SS\n0.8 V",
            0.8,
            sim_data_root / "baseline" / "rwtrans" / "rw_ss_08.csv",
            sim_data_root / "optimized" / "negative_bitline" / "trans_negBLopt_ss_0.8V.csv",
        ),
    ]
    guide_entries.append(
        plot_negative_bitline_delay_grid(
            negative_bitline_dir / "negative_bitline_write_delay_grid",
            negbl_measurements,
            [
                sim_data_root / "optimized" / "negative_bitline" / "trans_negBLopt_ff.csv",
                sim_data_root / "optimized" / "negative_bitline" / "trans_negBLopt_tt.csv",
                sim_data_root / "optimized" / "negative_bitline" / "trans_negBLopt_ss_1V.csv",
                sim_data_root / "optimized" / "negative_bitline" / "trans_negBLopt_ss_0.8V.csv",
            ],
        )
    )
    guide_entries.append(
        plot_two_condition_bars(
            negative_bitline_dir / "negative_bitline_write_delay_by_corner",
            [measurement.label for measurement in negbl_measurements],
            [measurement.baseline_delay_ps for measurement in negbl_measurements],
            [measurement.optimized_delay_ps for measurement in negbl_measurements],
            "Write delay (ps)",
            "Baseline",
            "Neg BL",
            BASELINE_COLOR,
            NEGATIVE_BITLINE_COLOR,
            value_formatter="{:.0f}",
            y_formatter="%.0f",
            delta_mode=None,
            source_files=[
                sim_data_root / "baseline" / "rwtrans" / "rw_ff.csv",
                sim_data_root / "baseline" / "rwtrans" / "rw_tt.csv",
                sim_data_root / "baseline" / "rwtrans" / "rw_ss_1.csv",
                sim_data_root / "baseline" / "rwtrans" / "rw_ss_08.csv",
                sim_data_root / "optimized" / "negative_bitline" / "trans_negBLopt_ff.csv",
                sim_data_root / "optimized" / "negative_bitline" / "trans_negBLopt_tt.csv",
                sim_data_root / "optimized" / "negative_bitline" / "trans_negBLopt_ss_1V.csv",
                sim_data_root / "optimized" / "negative_bitline" / "trans_negBLopt_ss_0.8V.csv",
            ],
            description="Corner-by-corner write delay comparison between the baseline cell and the negative-bitline assisted case.",
            processing_steps=[
                "Measured WL 50 percent to Q 50 percent delay for the baseline first write pulse and the optimized negative-bitline reverse-write pulse.",
                "Used the same delay extraction convention as the earlier negative-bitline zoom plots.",
                "Plotted the resulting delay values in a grouped bar chart without changing the measurement windows.",
            ],
            takeaway="How much the negative-bitline assist changes the measured write delay across the four transient corners.",
            caveats=[
                "The baseline bars correspond to the first write pulse, while the optimized bars correspond to the assisted reverse-write pulse in the third window of the negative-bitline bench.",
            ],
        )
    )

    guide_entries.append(
        plot_butterfly_overlay(
            wordline_dir / "wordline_underdrive_read_snm_all_corners",
            [
                load_snm_overlay_case(sim_data_root / "optimized" / "wordline_underdrive" / "read_snm_opt_ff.csv", "ff", "FF 1.2 V"),
                load_snm_overlay_case(sim_data_root / "optimized" / "wordline_underdrive" / "read_snm_opt_tt.csv", "tt", "TT 1.0 V"),
                load_snm_overlay_case(sim_data_root / "optimized" / "wordline_underdrive" / "read_snm_opt_ss.csv", "ss", "SS 1.0 V"),
            ],
            "Read-bias butterfly curves for the wordline-underdrive cell across the three available corners.",
            [
                sim_data_root / "optimized" / "wordline_underdrive" / "read_snm_opt_ff.csv",
                sim_data_root / "optimized" / "wordline_underdrive" / "read_snm_opt_tt.csv",
                sim_data_root / "optimized" / "wordline_underdrive" / "read_snm_opt_ss.csv",
            ],
            [
                "Loaded the raw optimized read-SNM CSVs for the underdriven wordline case.",
                "Computed the limiting read SNM square for each corner using the same shared fit helper as the baseline read plots.",
                "Used the same axis limits and corner colors as the baseline and high-Vt read overlays.",
            ],
            "The absolute read-SNM shapes for the wordline-underdrive case before comparing them against baseline.",
            [
                "Only FF, TT, and SS 1.0 V read-SNM CSVs are available for the wordline-underdrive dataset.",
            ],
        )
    )

    baseline_read_cases = [
        load_snm_overlay_case(sim_data_root / "baseline" / "read" / "ReadSNM_ff.csv", "ff", "FF 1.2 V"),
        load_snm_overlay_case(sim_data_root / "baseline" / "read" / "ReadSNM_tt.csv", "tt", "TT 1.0 V"),
        load_snm_overlay_case(sim_data_root / "baseline" / "read" / "ReadSNM_ss.csv", "ss", "SS 1.0 V"),
    ]
    wlu_read_cases = [
        load_snm_overlay_case(sim_data_root / "optimized" / "wordline_underdrive" / "read_snm_opt_ff.csv", "ff", "FF 1.2 V"),
        load_snm_overlay_case(sim_data_root / "optimized" / "wordline_underdrive" / "read_snm_opt_tt.csv", "tt", "TT 1.0 V"),
        load_snm_overlay_case(sim_data_root / "optimized" / "wordline_underdrive" / "read_snm_opt_ss.csv", "ss", "SS 1.0 V"),
    ]
    guide_entries.append(
        plot_two_condition_bars(
            wordline_dir / "wordline_underdrive_read_snm_comparison",
            [case.label.replace(" ", "\n", 1) for case in baseline_read_cases],
            [case.margin_mv for case in baseline_read_cases],
            [case.margin_mv for case in wlu_read_cases],
            "Read SNM (mV)",
            "Baseline",
            "WL underdrive",
            BASELINE_COLOR,
            WORDLINE_UNDERDRIVE_COLOR,
            value_formatter="{:.1f}",
            y_formatter="%.0f",
            delta_mode=None,
            source_files=[
                sim_data_root / "baseline" / "read" / "ReadSNM_ff.csv",
                sim_data_root / "baseline" / "read" / "ReadSNM_tt.csv",
                sim_data_root / "baseline" / "read" / "ReadSNM_ss.csv",
                sim_data_root / "optimized" / "wordline_underdrive" / "read_snm_opt_ff.csv",
                sim_data_root / "optimized" / "wordline_underdrive" / "read_snm_opt_tt.csv",
                sim_data_root / "optimized" / "wordline_underdrive" / "read_snm_opt_ss.csv",
            ],
            description="Grouped comparison of baseline and wordline-underdrive read SNM values by corner.",
            processing_steps=[
                "Computed baseline and underdriven read-SNM squares from the raw butterfly CSVs with the same limiting-eye routine.",
                "Converted the square side lengths to millivolts and plotted them on matched axes.",
            ],
            takeaway="Which corners gain or lose read SNM after the read wordline is underdriven.",
            caveats=[
                "The comparison uses only the three corners for which wordline-underdrive read-SNM CSVs are available.",
            ],
        )
    )

    guide_entries.append(
        plot_two_condition_bars(
            wordline_dir / "wordline_underdrive_read_disturb_comparison",
            [case["label"] for case in FOUR_CORNER_CASES],
            [
                measure_read_disturb(
                    sim_data_root / "baseline" / "rwtrans" / "rw_ff.csv",
                    pulse_window_ns=(13.0, 16.0),
                ),
                measure_read_disturb(
                    sim_data_root / "baseline" / "rwtrans" / "rw_tt.csv",
                    pulse_window_ns=(13.0, 16.0),
                ),
                measure_read_disturb(
                    sim_data_root / "baseline" / "rwtrans" / "rw_ss_1.csv",
                    pulse_window_ns=(13.0, 16.0),
                ),
                measure_read_disturb(
                    sim_data_root / "baseline" / "rwtrans" / "rw_ss_08.csv",
                    pulse_window_ns=(13.0, 16.0),
                ),
            ],
            [
                measure_read_disturb(
                    sim_data_root / "optimized" / "wordline_underdrive" / "trans_WLunderopt_ff.csv",
                    pulse_window_ns=(15.0, 18.0),
                ),
                measure_read_disturb(
                    sim_data_root / "optimized" / "wordline_underdrive" / "trans_WLunderopt_tt.csv",
                    pulse_window_ns=(15.0, 18.0),
                ),
                measure_read_disturb(
                    sim_data_root / "optimized" / "wordline_underdrive" / "trans_WLunderopt_ss_1V.csv",
                    pulse_window_ns=(15.0, 18.0),
                ),
                measure_read_disturb(
                    sim_data_root / "optimized" / "wordline_underdrive" / "trans_WLunderopt_ss_0.8V.csv",
                    pulse_window_ns=(15.0, 18.0),
                ),
            ],
            "QB read disturb (mV)",
            "Baseline",
            "WL underdrive",
            BASELINE_COLOR,
            WORDLINE_UNDERDRIVE_COLOR,
            value_formatter="{:.1f}",
            y_formatter="%.0f",
            delta_mode=None,
            source_files=[
                sim_data_root / "baseline" / "rwtrans" / "rw_ff.csv",
                sim_data_root / "baseline" / "rwtrans" / "rw_tt.csv",
                sim_data_root / "baseline" / "rwtrans" / "rw_ss_1.csv",
                sim_data_root / "baseline" / "rwtrans" / "rw_ss_08.csv",
                sim_data_root / "optimized" / "wordline_underdrive" / "trans_WLunderopt_ff.csv",
                sim_data_root / "optimized" / "wordline_underdrive" / "trans_WLunderopt_tt.csv",
                sim_data_root / "optimized" / "wordline_underdrive" / "trans_WLunderopt_ss_1V.csv",
                sim_data_root / "optimized" / "wordline_underdrive" / "trans_WLunderopt_ss_0.8V.csv",
            ],
            description="Baseline versus wordline-underdrive comparison of the QB read-disturb magnitude.",
            processing_steps=[
                "Loaded the raw /QB and /WL transient traces from each CSV.",
                "Detected the read pulse within the requested broad time ranges and used the mean QB level from a short pre-pulse window as the reference.",
                "Reported disturb magnitude as the peak QB excursion above that pre-read reference during the read pulse.",
            ],
            takeaway="Whether underdriving the read wordline reduces the size of the low-node disturb across the transient corners.",
            caveats=[
                "The baseline and underdrive benches use different absolute read time windows, so the bar chart compares disturb magnitude only and does not overlay time traces.",
            ],
        )
    )

    guide_path = output_root / "figure_guide.md"
    write_figure_guide(guide_path, guide_entries)
    verify_guide_entries(guide_path, guide_entries)
    print_saved_path(guide_path)


def configure_matplotlib() -> None:
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


def load_snm_overlay_case(csv_path: Path, key: str, label: str) -> ButterflyOverlayCase:
    header, data = load_csv(csv_path)
    curve_a = Curve2D(data[:, 0], data[:, 1], "curve a")
    curve_b = Curve2D(data[:, 2], data[:, 3], "curve b")
    fit = largest_square_between_curves(curve_a, curve_b, choose_smallest=True)
    if fit is None:
        raise ValueError(f"No SNM eye found in {csv_path}")
    return ButterflyOverlayCase(
        key=key,
        label=label,
        curve_a=curve_a,
        curve_b=curve_b,
        margin_mv=fit.side * 1000.0,
        square_xy=fit.square_xy,
    )


def load_wnm_overlay_case(csv_path: Path, key: str, label: str) -> ButterflyOverlayCase:
    header, data = load_csv(csv_path)
    curve_a, curve_b = wnm_curves_from_csv(csv_path, header, data)
    fit = diagonal_square_from_crossings(curve_a, curve_b)
    if fit is None:
        fit = diagonal_square_between_curves(curve_a, curve_b)
    if fit is None:
        raise ValueError(f"No WNM eye found in {csv_path}")
    return ButterflyOverlayCase(
        key=key,
        label=label,
        curve_a=curve_a,
        curve_b=curve_b,
        margin_mv=fit.side * 1000.0,
        square_xy=fit.square_xy,
    )


def plot_butterfly_overlay(
    base_path: Path,
    cases: list[ButterflyOverlayCase],
    description: str,
    source_files: list[Path],
    processing_steps: list[str],
    takeaway: str,
    caveats: list[str],
) -> GuideEntry:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=BUTTERFLY_FIGSIZE)

    diagonal_x = np.asarray([0.0, AXIS_LIMIT], dtype=float)
    ax.plot(diagonal_x, diagonal_x, color="0.35", lw=0.9, ls=(0, (4, 3)))

    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    for case in cases:
        color = CORNER_COLORS[case.key]
        ax.plot(case.curve_a.x, case.curve_a.y, color=color, lw=1.4, ls="-")
        ax.plot(case.curve_b.x, case.curve_b.y, color=color, lw=1.4, ls="--")
        closed_square = np.vstack((case.square_xy, case.square_xy[:1]))
        ax.plot(closed_square[:, 0], closed_square[:, 1], color=color, lw=2.1)

        legend_handles.extend(
            [
                Line2D([0], [0], color=color, lw=1.4, ls="-"),
                Line2D([0], [0], color=color, lw=2.1, ls="-"),
            ]
        )
        legend_labels.extend(
            [
                f"{case.label} curves",
                f"{case.label} square {case.margin_mv:.1f} mV",
            ]
        )

    format_butterfly_axis(ax)
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.96),
        columnspacing=1.2,
        handlelength=2.2,
        fontsize=8.0,
    )
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.85)
    save_figure(fig, base_path)
    plt.close(fig)

    return GuideEntry(
        relative_png=base_path.with_suffix(".png").relative_to(OUTPUT_ROOT),
        description=description,
        source_files=source_files,
        processing_steps=processing_steps,
        takeaway=takeaway,
        caveats=caveats,
    )


def plot_two_condition_bars(
    base_path: Path,
    x_labels: list[str],
    baseline_values: list[float],
    optimized_values: list[float],
    y_label: str,
    baseline_label: str,
    optimized_label: str,
    baseline_color: str,
    optimized_color: str,
    *,
    value_formatter: str,
    y_formatter: str,
    delta_mode: str | None,
    source_files: list[Path],
    description: str,
    processing_steps: list[str],
    takeaway: str,
    caveats: list[str],
) -> GuideEntry:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=BAR_FIGSIZE)

    x = np.arange(len(x_labels), dtype=float)
    width = 0.34
    baseline_pos = x - width / 2
    optimized_pos = x + width / 2

    baseline_bars = ax.bar(baseline_pos, baseline_values, width, color=baseline_color, label=baseline_label)
    optimized_bars = ax.bar(optimized_pos, optimized_values, width, color=optimized_color, label=optimized_label)

    y_max = max(max(baseline_values), max(optimized_values))
    ax.set_ylim(0.0, y_max * (1.24 if delta_mode == "percent" else 1.16))
    ax.set_ylabel(y_label)
    ax.set_xticks(x, x_labels)
    ax.yaxis.set_major_formatter(FormatStrFormatter(y_formatter))
    format_bar_axis(ax)
    ax.legend(frameon=False, loc="upper right", ncol=2)

    annotate_bars(ax, baseline_bars, value_formatter, y_max)
    annotate_bars(ax, optimized_bars, value_formatter, y_max)

    if delta_mode == "percent":
        for index, (baseline_value, optimized_value) in enumerate(zip(baseline_values, optimized_values)):
            delta_pct = 100.0 * (optimized_value - baseline_value) / baseline_value
            delta_color = "#2f7f4f" if delta_pct < 0.0 else "#b23a48"
            ax.text(
                x[index],
                max(baseline_value, optimized_value) + 0.11 * y_max,
                f"{delta_pct:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=9.0,
                color=delta_color,
                fontweight="bold",
            )

    fig.tight_layout()
    save_figure(fig, base_path)
    plt.close(fig)

    return GuideEntry(
        relative_png=base_path.with_suffix(".png").relative_to(OUTPUT_ROOT),
        description=description,
        source_files=source_files,
        processing_steps=processing_steps,
        takeaway=takeaway,
        caveats=caveats,
    )


def plot_negative_bitline_delay_grid(
    base_path: Path,
    measurements: list[DelayMeasurement],
    source_files: list[Path],
) -> GuideEntry:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=GRID_FIGSIZE, sharex=True, sharey=True)
    axes_flat = list(axes.ravel())
    x_window = (20.95, 21.15)
    y_limits = (-0.10, 1.28)

    for axis, measurement in zip(axes_flat, measurements):
        mask = (measurement.time_ns >= x_window[0]) & (measurement.time_ns <= x_window[1])
        axis.plot(measurement.time_ns[mask], measurement.wl_v[mask], color=TRACE_COLORS["wl"], lw=1.3)
        axis.plot(measurement.time_ns[mask], measurement.bl_v[mask], color=TRACE_COLORS["bl"], lw=1.3)
        axis.plot(measurement.time_ns[mask], measurement.q_v[mask], color=TRACE_COLORS["q"], lw=1.5)
        axis.axhline(0.5 * measurement.vdd, color="0.35", lw=0.8, ls=(0, (3, 3)))
        axis.axvline(measurement.wl_cross_ns, color=TRACE_COLORS["wl"], lw=0.8, ls=(0, (3, 3)))
        axis.axvline(measurement.q_cross_ns, color=TRACE_COLORS["q"], lw=0.8, ls=(0, (3, 3)))
        arrow_y = 1.08 * measurement.vdd
        axis.annotate(
            "",
            xy=(measurement.q_cross_ns, arrow_y),
            xytext=(measurement.wl_cross_ns, arrow_y),
            arrowprops={"arrowstyle": "<->", "lw": 1.2, "color": NEGATIVE_BITLINE_COLOR},
        )
        axis.text(
            0.03,
            0.92,
            measurement.label.replace("\n", ", "),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.3,
        )
        axis.text(
            0.97,
            0.90,
            f"{measurement.optimized_delay_ps:.1f} ps",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9.0,
            color=NEGATIVE_BITLINE_COLOR,
            fontweight="bold",
        )
        axis.set_xlim(*x_window)
        axis.set_ylim(*y_limits)
        format_waveform_axis(axis)

    axes[1, 0].set_xlabel("Time (ns)")
    axes[1, 1].set_xlabel("Time (ns)")
    axes[0, 0].set_ylabel("Voltage (V)")
    axes[1, 0].set_ylabel("Voltage (V)")

    legend_handles = [
        Line2D([0], [0], color=TRACE_COLORS["wl"], lw=1.3),
        Line2D([0], [0], color=TRACE_COLORS["bl"], lw=1.3),
        Line2D([0], [0], color=TRACE_COLORS["q"], lw=1.5),
    ]
    fig.legend(
        legend_handles,
        ["WL", "BL", "Q"],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        columnspacing=1.6,
        handlelength=2.2,
        fontsize=8.0,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.90, wspace=0.16, hspace=0.18)
    save_figure(fig, base_path)
    plt.close(fig)

    return GuideEntry(
        relative_png=base_path.with_suffix(".png").relative_to(OUTPUT_ROOT),
        description="Compact four-corner view of the negative-bitline write-delay zoom traces around the assisted write transition.",
        source_files=source_files,
        processing_steps=[
            "Loaded the raw optimized negative-bitline transient CSVs.",
            "Measured WL 50 percent to Q 50 percent delay on the assisted write pulse and reused those crossing times to place the arrows and labels.",
            "Plotted the same zoom window for all four corners to keep the waveform comparison visually consistent.",
        ],
        takeaway="How the negative-bitline assisted write transient looks across corners while emphasizing the extracted delay value in each panel.",
        caveats=[
            "This grid shows only the optimized negative-bitline waveforms; the baseline comparison is summarized separately in the grouped delay bar chart.",
        ],
    )


def integrate_supply_energy(csv_path: Path, window_ns: tuple[float, float]) -> float:
    signals = load_waveform_csv(csv_path)
    current = signals["/V0/PLUS"]
    vdd = signals["/VDD"]
    charge_c = integrate_window_ns(current["time_ns"], current["value"], *window_ns)
    avg_vdd_v = integrate_window_ns(vdd["time_ns"], vdd["value"], *window_ns) / ((window_ns[1] - window_ns[0]) * 1e-9)
    return -avg_vdd_v * charge_c * 1e18


def measure_negative_bitline_delay(
    label: str,
    vdd: float,
    baseline_csv: Path,
    optimized_csv: Path,
) -> DelayMeasurement:
    baseline_signals = load_waveform_csv(baseline_csv)
    optimized_signals = load_waveform_csv(optimized_csv)

    baseline_delay_ps, _, _ = measure_write_delay(
        baseline_signals["/Q"]["time_ns"],
        baseline_signals["/WL"]["value"],
        baseline_signals["/Q"]["value"],
        vdd,
        window_ns=(2.5, 9.0),
        q_rising=True,
    )
    optimized_delay_ps, wl_cross_ns, q_cross_ns = measure_write_delay(
        optimized_signals["/Q"]["time_ns"],
        optimized_signals["/WL"]["value"],
        optimized_signals["/Q"]["value"],
        vdd,
        window_ns=(20.0, 28.0),
        q_rising=False,
    )

    return DelayMeasurement(
        label=label,
        vdd=vdd,
        baseline_delay_ps=baseline_delay_ps,
        optimized_delay_ps=optimized_delay_ps,
        time_ns=optimized_signals["/Q"]["time_ns"],
        wl_v=optimized_signals["/WL"]["value"],
        bl_v=optimized_signals["/BL"]["value"],
        q_v=optimized_signals["/Q"]["value"],
        wl_cross_ns=wl_cross_ns,
        q_cross_ns=q_cross_ns,
    )


def measure_write_delay(
    time_ns: np.ndarray,
    wl_v: np.ndarray,
    q_v: np.ndarray,
    vdd: float,
    *,
    window_ns: tuple[float, float],
    q_rising: bool,
) -> tuple[float, float, float]:
    mask = (time_ns >= window_ns[0]) & (time_ns <= window_ns[1])
    if not np.any(mask):
        raise ValueError(f"No samples found in delay window {window_ns}")
    t_window = time_ns[mask]
    wl_window = wl_v[mask]
    q_window = q_v[mask]
    half_vdd = 0.5 * vdd
    wl_cross_ns = first_crossing_ns(t_window, wl_window, half_vdd, rising=True)
    q_cross_ns = first_crossing_ns(t_window, q_window, half_vdd, rising=q_rising)
    if wl_cross_ns is None or q_cross_ns is None:
        raise ValueError(f"Could not find 50 percent crossings in window {window_ns}")
    return (q_cross_ns - wl_cross_ns) * 1000.0, wl_cross_ns, q_cross_ns


def measure_read_disturb(csv_path: Path, *, pulse_window_ns: tuple[float, float]) -> float:
    signals = load_waveform_csv(csv_path)
    time_ns = signals["/QB"]["time_ns"]
    qb_v = signals["/QB"]["value"]
    wl_v = signals["/WL"]["value"]
    wl_threshold = 0.5 * float(np.max(wl_v))

    pulse_start_ns, pulse_end_ns = detect_pulse_window_ns(time_ns, wl_v, pulse_window_ns, wl_threshold)
    reference_window_ns = (max(0.0, pulse_start_ns - 0.50), pulse_start_ns - 0.10)
    reference_mask = (time_ns >= reference_window_ns[0]) & (time_ns <= reference_window_ns[1])
    pulse_mask = (time_ns >= pulse_start_ns) & (time_ns <= pulse_end_ns)
    if not np.any(reference_mask) or not np.any(pulse_mask):
        raise ValueError(f"Could not form read-disturb windows for {csv_path}")

    reference_qb_v = float(np.mean(qb_v[reference_mask]))
    peak_qb_v = float(np.max(qb_v[pulse_mask]))
    return (peak_qb_v - reference_qb_v) * 1000.0


def detect_pulse_window_ns(
    time_ns: np.ndarray,
    waveform_v: np.ndarray,
    search_window_ns: tuple[float, float],
    threshold_v: float,
) -> tuple[float, float]:
    mask = (time_ns >= search_window_ns[0]) & (time_ns <= search_window_ns[1])
    if not np.any(mask):
        raise ValueError(f"No samples in search window {search_window_ns}")
    t_window = time_ns[mask]
    y_window = waveform_v[mask]
    pulse_start_ns = first_crossing_ns(t_window, y_window, threshold_v, rising=True)
    pulse_end_ns = first_crossing_ns(t_window, y_window, threshold_v, rising=False)
    if pulse_start_ns is None or pulse_end_ns is None or pulse_end_ns <= pulse_start_ns:
        return search_window_ns
    return pulse_start_ns, pulse_end_ns


def first_crossing_ns(
    time_ns: np.ndarray,
    waveform_v: np.ndarray,
    threshold_v: float,
    *,
    rising: bool,
) -> float | None:
    if rising:
        indices = np.where((waveform_v[:-1] < threshold_v) & (waveform_v[1:] >= threshold_v))[0]
    else:
        indices = np.where((waveform_v[:-1] > threshold_v) & (waveform_v[1:] <= threshold_v))[0]
    if len(indices) == 0:
        return None
    index = int(indices[0])
    x0 = float(time_ns[index])
    x1 = float(time_ns[index + 1])
    y0 = float(waveform_v[index])
    y1 = float(waveform_v[index + 1])
    if y1 == y0:
        return x0
    return x0 + (threshold_v - y0) * (x1 - x0) / (y1 - y0)


def integrate_window_ns(
    time_ns: np.ndarray,
    values: np.ndarray,
    start_ns: float,
    end_ns: float,
) -> float:
    time_s = np.asarray(time_ns, dtype=float) * 1e-9
    values = np.asarray(values, dtype=float)
    start_s = start_ns * 1e-9
    end_s = end_ns * 1e-9
    mask = (time_s >= start_s) & (time_s <= end_s)
    time_sel = time_s[mask]
    value_sel = values[mask]
    if time_sel.size == 0:
        raise ValueError(f"No samples found between {start_ns} ns and {end_ns} ns")

    if time_sel[0] > start_s:
        start_value = np.interp(start_s, time_s, values)
        time_sel = np.insert(time_sel, 0, start_s)
        value_sel = np.insert(value_sel, 0, start_value)
    else:
        time_sel[0] = start_s
        value_sel[0] = np.interp(start_s, time_s, values)

    if time_sel[-1] < end_s:
        end_value = np.interp(end_s, time_s, values)
        time_sel = np.append(time_sel, end_s)
        value_sel = np.append(value_sel, end_value)
    else:
        time_sel[-1] = end_s
        value_sel[-1] = np.interp(end_s, time_s, values)

    try:
        return float(np.trapezoid(value_sel, time_sel))
    except AttributeError:
        return float(np.trapz(value_sel, time_sel))


def format_butterfly_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0.0, AXIS_LIMIT)
    ax.set_ylim(0.0, AXIS_LIMIT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("V1 (V)")
    ax.set_ylabel("V2 (V)")
    ax.set_xticks(AXIS_TICKS)
    ax.set_yticks(AXIS_TICKS)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    format_axis_common(ax)


def format_bar_axis(ax: plt.Axes) -> None:
    format_axis_common(ax)


def format_waveform_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    format_axis_common(ax)


def format_axis_common(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", top=True, right=True, length=4.0, width=0.9)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def annotate_bars(ax: plt.Axes, bars, value_formatter: str, y_max: float) -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            float(bar.get_x() + bar.get_width() / 2.0),
            value + 0.028 * y_max,
            value_formatter.format(value),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print_saved_path(png_path)
    print_saved_path(pdf_path)


def print_saved_path(path: Path) -> None:
    print(path.relative_to(REPO_ROOT))


def write_figure_guide(path: Path, entries: list[GuideEntry]) -> None:
    lines = [
        "# Bitcell Figure Guide",
        "",
        "Generated by `scripts/plot_bitcell_report_figures.py` from the raw CSV data under `sim_data/`.",
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                f"## `{entry.relative_png.as_posix()}`",
                "",
                f"1. Figure filename: `{entry.relative_png.as_posix()}` and matching `.pdf` saved alongside it",
                f"2. Brief description of what the figure shows: {entry.description}",
                "3. Source data file or files used:",
            ]
        )
        lines.extend([f"   - `{source.relative_to(REPO_ROOT).as_posix()}`" for source in entry.source_files])
        lines.append("4. Key processing steps applied:")
        lines.extend([f"   - {step}" for step in entry.processing_steps])
        lines.append(f"5. What the figure helps the reader understand: {entry.takeaway}")
        lines.append("6. Any important caveats or assumptions:")
        lines.extend([f"   - {caveat}" for caveat in entry.caveats])
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def verify_guide_entries(guide_path: Path, entries: list[GuideEntry]) -> None:
    guide_text = guide_path.read_text(encoding="utf-8")
    for entry in entries:
        if entry.relative_png.as_posix() not in guide_text:
            raise ValueError(f"Guide is missing {entry.relative_png.as_posix()}")
        if not (OUTPUT_ROOT / entry.relative_png).exists():
            raise ValueError(f"Missing generated PNG {entry.relative_png.as_posix()}")
        if not (OUTPUT_ROOT / entry.relative_png.with_suffix(".pdf")).exists():
            raise ValueError(f"Missing generated PDF for {entry.relative_png.as_posix()}")


if __name__ == "__main__":
    main()
