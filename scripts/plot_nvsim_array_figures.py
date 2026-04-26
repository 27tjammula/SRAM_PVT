#!/usr/bin/env python3
"""Generate report-facing array figures and report index files."""

from __future__ import annotations

import argparse
import csv
import os
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
REPORT_FIGURES_ROOT = REPO_ROOT / "sram_analysis_plots" / "report_figures"
ARRAY_FIGURES_DIR = REPORT_FIGURES_ROOT / "array"
ARRAY_GUIDE_PATH = ARRAY_FIGURES_DIR / "figure_guide.md"
FIGURE_INDEX_PATH = REPORT_FIGURES_ROOT / "figure_index.md"
REPORT_OUTLINE_PATH = REPO_ROOT / "sram_analysis_plots" / "report_outline.md"
LOCAL_TEX_BIN = REPO_ROOT / ".tex" / "TinyTeX" / "bin" / "universal-darwin"

PNG_DPI = 400
SUMMARY_FIGSIZE = (7.2, 4.6)
BREAKDOWN_FIGSIZE = (7.2, 3.8)
DELTA_FIGSIZE = (5.8, 4.0)

NVSIM_FILL = "#4c72b0"
PROXY_FILL = "#dd8452"
CADENCE_FILL = "#dfeedd"
BREAKDOWN_COLORS = {
    "Row decode": "#4c72b0",
    "Bitline path": "#dd8452",
    "Other": "#c5c9d3",
}


@dataclass(frozen=True)
class GuideEntry:
    relative_png: Path
    description: str
    source_files: list[Path]
    processing_steps: list[str]
    takeaway: str
    caveats: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report-ready array figures from fresh NVSim outputs.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Directory containing the fresh array CSV outputs.")
    parser.add_argument("--figures-dir", type=Path, default=ARRAY_FIGURES_DIR, help="Directory for generated array figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_row = read_single_row(data_dir / "nvsim_baseline_macro.csv")
    breakdown_rows = read_csv_rows(data_dir / "nvsim_baseline_latency_breakdown.csv")
    bridge_rows = read_csv_rows(data_dir / "array_bridge_case_summary.csv")

    guide_entries = [
        plot_baseline_macro_summary(figures_dir / "baseline_nvsim_macro_summary", baseline_row),
        plot_baseline_latency_breakdown(figures_dir / "baseline_nvsim_latency_breakdown", breakdown_rows),
        plot_high_vt_leakage_proxy(figures_dir / "high_vt_leakage_proxy", bridge_rows),
        plot_negative_bl_write_latency_proxy(figures_dir / "negative_bl_write_latency_proxy", bridge_rows),
    ]

    write_figure_guide(ARRAY_GUIDE_PATH, guide_entries)
    write_figure_index(FIGURE_INDEX_PATH)
    write_report_outline(REPORT_OUTLINE_PATH)
    verify_outputs(guide_entries)

    print(ARRAY_GUIDE_PATH.relative_to(REPO_ROOT))
    print(FIGURE_INDEX_PATH.relative_to(REPO_ROOT))
    print(REPORT_OUTLINE_PATH.relative_to(REPO_ROOT))


def read_single_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


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


def plot_baseline_macro_summary(base_path: Path, row: dict[str, str]) -> GuideEntry:
    enable_local_tex_path()
    summary_rows = baseline_macro_summary_rows(row)
    tex_path = base_path.with_suffix(".tex")
    write_baseline_macro_tex(tex_path, summary_rows)
    print(tex_path.relative_to(REPO_ROOT))

    rc_overrides = {
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{booktabs}",
        "font.family": "serif",
        "font.size": 11.0,
    }
    with plt.rc_context(rc_overrides):
        fig, ax = plt.subplots(figsize=SUMMARY_FIGSIZE)
        ax.axis("off")
        ax.text(
            0.5,
            0.94,
            r"\textbf{\large Baseline NVSim Macro Summary}",
            transform=ax.transAxes,
            ha="center",
            va="top",
        )
        ax.text(
            0.5,
            0.50,
            build_baseline_macro_tabular(summary_rows, single_line=True),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11.0,
        )
        ax.text(
            0.5,
            0.04,
            r"\textit{\small All rows in this table are direct NVSim estimates from the fresh baseline run.}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
        )
        save_figure(fig, base_path)
        plt.close(fig)
    return GuideEntry(
        relative_png=base_path.with_suffix(".png").relative_to(REPORT_FIGURES_ROOT),
        description="Compact table-style summary of the fresh baseline NVSim macro used as the direct array anchor for the report.",
        source_files=[DATA_DIR / "nvsim_baseline_macro.csv"],
        processing_steps=[
            "Loaded the single-row baseline macro CSV generated by the fresh NVSim baseline run.",
            "Emitted a booktabs LaTeX table source (`.tex`) alongside the PNG/PDF so the report can `\\input{}` the typeset table directly.",
            "Rendered the matplotlib figure with `text.usetex=True` (project-local TinyTeX) so the raster preview matches the LaTeX typography.",
        ],
        takeaway="Provides the direct baseline macro snapshot once, without spending figure space repeating unchanged quantities in comparison plots.",
        caveats=[
            "This figure reflects only the fresh baseline NVSim run, not the bridged optimization cases.",
            "The matplotlib render and the `.tex` source share the same booktabs content, but the report should `\\input{baseline_nvsim_macro_summary.tex}` rather than embed the rasterized PNG.",
        ],
    )


def baseline_macro_summary_rows(row: dict[str, str]) -> list[tuple[str, str]]:
    return [
        ("Capacity", rf"{row['capacity_bytes']} B"),
        ("Physical array", rf"{row['rows']} $\times$ {row['columns']}"),
        ("Word width", rf"{row['word_width_bits']} bits"),
        ("Column selection", rf"{row['column_mux_factor']}:1"),
        ("Area", rf"{float(row['area_mm2']):.6f} mm$^2$"),
        ("Read latency", rf"{float(row['read_latency_ns']):.4f} ns"),
        ("Write latency", rf"{float(row['write_latency_ns']):.4f} ns"),
        ("Read energy", rf"{float(row['read_energy_pj']):.4f} pJ"),
        ("Write energy", rf"{float(row['write_energy_pj']):.4f} pJ"),
        ("Leakage", rf"{float(row['leakage_power_uw']):.4f} $\mu$W"),
    ]


def build_baseline_macro_tabular(rows: list[tuple[str, str]], *, single_line: bool = False) -> str:
    parts = [
        r"\begin{tabular}{@{}lr@{}}",
        r"\toprule",
        r"\textbf{Baseline macro quantity} & \textbf{Value} \\",
        r"\midrule",
    ]
    for label, value in rows:
        parts.append(f"{label} & {value} \\\\")
    parts.extend([r"\bottomrule", r"\end{tabular}"])
    return (" " if single_line else "\n").join(parts)


def write_baseline_macro_tex(tex_path: Path, rows: list[tuple[str, str]]) -> None:
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    body = build_baseline_macro_tabular(rows)
    header = (
        "% Generated by scripts/plot_nvsim_array_figures.py\n"
        "% Source: sram_analysis_plots/array/data/nvsim_baseline_macro.csv\n"
        "% Requires: \\usepackage{booktabs}\n"
    )
    tex_path.write_text(header + body + "\n")


def enable_local_tex_path() -> None:
    pdflatex_path = LOCAL_TEX_BIN / "pdflatex"
    if not pdflatex_path.exists():
        raise FileNotFoundError(
            f"Project-local TinyTeX not found at {pdflatex_path}. "
            "See memory/project_local_pdflatex.md for the install steps."
        )
    bin_str = str(LOCAL_TEX_BIN)
    current_path = os.environ.get("PATH", "")
    if bin_str not in current_path.split(os.pathsep):
        os.environ["PATH"] = bin_str + os.pathsep + current_path


def plot_baseline_latency_breakdown(base_path: Path, rows: list[dict[str, str]]) -> GuideEntry:
    configure_matplotlib()
    ordered_accesses = ["Read", "Write"]
    components = ["Row decode", "Bitline path", "Other"]
    values = {
        access: [float(next(row["latency_ns"] for row in rows if row["access"] == access and row["component"] == component)) for component in components]
        for access in ordered_accesses
    }
    fig, ax = plt.subplots(figsize=BREAKDOWN_FIGSIZE)
    y = np.arange(len(ordered_accesses), dtype=float)
    left = np.zeros(len(ordered_accesses), dtype=float)
    for component_index, component in enumerate(components):
        component_values = np.asarray([values[access][component_index] for access in ordered_accesses], dtype=float)
        ax.barh(
            y,
            component_values,
            left=left,
            color=BREAKDOWN_COLORS[component],
            height=0.52,
            label=component,
        )
        for index, value in enumerate(component_values):
            if value > 0.04:
                ax.text(left[index] + value / 2.0, y[index], f"{value:.3f}", ha="center", va="center", fontsize=8.3)
        left += component_values
    ax.set_yticks(y, ordered_accesses)
    ax.set_xlabel("Latency (ns)")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.invert_yaxis()
    format_axis(ax)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax.set_title("Baseline NVSim Latency Breakdown")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    save_figure(fig, base_path)
    plt.close(fig)
    return GuideEntry(
        relative_png=base_path.with_suffix(".png").relative_to(REPORT_FIGURES_ROOT),
        description="Row-decode, bitline-path, and residual latency breakdown for the baseline NVSim read and write accesses.",
        source_files=[DATA_DIR / "nvsim_baseline_latency_breakdown.csv"],
        processing_steps=[
            "Loaded the aggregated read and write breakdown CSV from the fresh baseline run.",
            "Stacked the three requested breakdown components on separate read and write bars.",
            "Annotated the in-bar nanosecond contributions to keep the figure usable without a separate table.",
        ],
        takeaway="Shows which internal blocks dominate the baseline access time before any optimization bridge is applied.",
        caveats=[
            "The `Other` segment is the residual to total latency after the explicit row-decode and bitline-path terms are removed.",
        ],
    )


def plot_high_vt_leakage_proxy(base_path: Path, rows: list[dict[str, str]]) -> GuideEntry:
    configure_matplotlib()
    baseline_row = find_case_row(rows, "Baseline")
    high_vt_row = find_case_row(rows, "High Vt")
    return plot_two_case_delta(
        base_path=base_path,
        title="High-Vt Leakage Estimate",
        y_label="Leakage power (uW)",
        baseline_case="Baseline",
        optimized_case="High Vt",
        baseline_value=float(baseline_row["macro_leakage_power_uw"]),
        optimized_value=float(high_vt_row["macro_leakage_power_uw"]),
        baseline_source=baseline_row["macro_leakage_power_uw_evidence_type"],
        optimized_source=high_vt_row["macro_leakage_power_uw_evidence_type"],
        optimized_note="Scaled from TT hold-window energy ratio",
        source_files=[DATA_DIR / "nvsim_baseline_macro.csv", DATA_DIR / "array_bridge_case_summary.csv"],
        processing_steps=[
            "Loaded the fresh baseline macro leakage estimate and the High-Vt bridged leakage row from the array bridge summary CSV.",
            "Dropped unchanged macro quantities and plotted only the leakage value that actually changes in the High-Vt bridge.",
            "Annotated the percent change so the figure reads as a result rather than a raw table dump.",
        ],
        takeaway="Shows the only macro-level quantity that changes in the High-Vt bridge: a lower leakage estimate.",
        caveats=[
            "The High-Vt bar is a derived proxy from the TT hold-window energy ratio, not a direct second NVSim topology run.",
        ],
    )


def plot_negative_bl_write_latency_proxy(base_path: Path, rows: list[dict[str, str]]) -> GuideEntry:
    baseline_row = find_case_row(rows, "Baseline")
    negative_bl_row = find_case_row(rows, "Negative BL")
    return plot_two_case_delta(
        base_path=base_path,
        title="Negative-BL Write-Latency Estimate",
        y_label="Write latency (ns)",
        baseline_case="Baseline",
        optimized_case="Negative BL",
        baseline_value=float(baseline_row["macro_write_latency_ns"]),
        optimized_value=float(negative_bl_row["macro_write_latency_ns"]),
        baseline_source=baseline_row["macro_write_latency_ns_evidence_type"],
        optimized_source=negative_bl_row["macro_write_latency_ns_evidence_type"],
        optimized_note="Scaled from TT Cadence write-delay ratio",
        source_files=[DATA_DIR / "nvsim_baseline_macro.csv", DATA_DIR / "array_bridge_case_summary.csv"],
        processing_steps=[
            "Loaded the fresh baseline macro write-latency estimate and the Negative-BL bridged write-latency row from the array bridge summary CSV.",
            "Removed all unchanged macro quantities and plotted only the write-latency shift that the Negative-BL bridge actually claims.",
            "Annotated the percent change so the chart focuses on the defended array-level benefit.",
        ],
        takeaway="Shows the one macro quantity we are willing to move for Negative BL in the current bridge: write latency.",
        caveats=[
            "The Negative-BL bar is a derived proxy from the TT Cadence write-delay ratio because this NVSim flow does not model explicit negative-bitline assist waveforms.",
        ],
    )


def plot_two_case_delta(
    *,
    base_path: Path,
    title: str,
    y_label: str,
    baseline_case: str,
    optimized_case: str,
    baseline_value: float,
    optimized_value: float,
    baseline_source: str,
    optimized_source: str,
    optimized_note: str,
    source_files: list[Path],
    processing_steps: list[str],
    takeaway: str,
    caveats: list[str],
) -> GuideEntry:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=DELTA_FIGSIZE)
    x = np.arange(2, dtype=float)
    values = np.asarray([baseline_value, optimized_value], dtype=float)
    colors = [NVSIM_FILL, PROXY_FILL]
    bars = ax.bar(x, values, width=0.56, color=colors)
    y_max = max(values)
    ax.set_ylabel(y_label)
    ax.set_xticks(
        x,
        [
            f"{baseline_case}\n{short_source_label(baseline_source)}",
            f"{optimized_case}\n{short_source_label(optimized_source)}",
        ],
    )
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    format_axis(ax)
    annotate_delta_bars(ax, bars, y_max)
    delta_percent = 100.0 * (optimized_value - baseline_value) / baseline_value if baseline_value else 0.0
    ax.text(
        0.5,
        max(values) * 1.07,
        f"{delta_percent:+.1f}% vs baseline",
        ha="center",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color="#2f7f4f" if delta_percent < 0.0 else "#b23a48",
    )
    ax.text(
        0.98,
        0.96,
        optimized_note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.1,
    )
    ax.set_ylim(0.0, max(values) * 1.22)
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, base_path)
    plt.close(fig)
    return GuideEntry(
        relative_png=base_path.with_suffix(".png").relative_to(REPORT_FIGURES_ROOT),
        description=f"Two-case comparison focused only on the macro quantity that changes in the {optimized_case} bridge.",
        source_files=source_files,
        processing_steps=processing_steps,
        takeaway=takeaway,
        caveats=caveats,
    )


def short_source_label(evidence_type: str) -> str:
    mapping = {
        "Estimated by NVSim": "NVSim",
        "Derived proxy": "Proxy",
        "Measured in Cadence": "Cadence",
        "NA": "NA",
    }
    return mapping[evidence_type]


def annotate_delta_bars(ax: plt.Axes, bars, y_max: float) -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            float(bar.get_x() + bar.get_width() / 2.0),
            value + 0.03 * y_max,
            f"{value:.4g}",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
        )


def find_case_row(rows: list[dict[str, str]], case: str) -> dict[str, str]:
    return next(row for row in rows if row["case"] == case)


def format_axis(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", top=True, right=True, length=4.0, width=0.9)
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    png_path = base_path.with_suffix(".png")
    pdf_path = base_path.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=PNG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(png_path.relative_to(REPO_ROOT))
    print(pdf_path.relative_to(REPO_ROOT))


def write_figure_guide(path: Path, entries: list[GuideEntry]) -> None:
    lines = [
        "# Array Figure Guide",
        "",
        "Generated by `scripts/plot_nvsim_array_figures.py` from the fresh CSV outputs under `sram_analysis_plots/array/data`.",
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
        for source in entry.source_files:
            lines.append(f"   - `{source.relative_to(REPO_ROOT).as_posix()}`")
        lines.append("4. Key processing steps applied:")
        for step in entry.processing_steps:
            lines.append(f"   - {step}")
        lines.extend(
            [
                f"5. What the figure helps the reader understand: {entry.takeaway}",
                "6. Any important caveats or assumptions:",
            ]
        )
        for caveat in entry.caveats:
            lines.append(f"   - {caveat}")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n")


def write_figure_index(path: Path) -> None:
    main_text_rows = [
        ["1", "bitcell/bitcell_results_summary.png", "Compact baseline stability snapshot plus TT optimization headlines."],
        ["2", "bitcell/high_vt/high_vt_tradeoff_summary.png", "High-Vt energy benefit and read-SNM penalty."],
        ["3", "bitcell/negative_bitline/negative_bitline_write_delay_by_corner.png", "Negative-BL write-delay comparison by corner."],
        ["4", "bitcell/wordline_underdrive/wordline_underdrive_read_snm_comparison.png", "WL-underdrive read-SNM comparison."],
        ["5", "bitcell/wordline_underdrive/wordline_underdrive_read_disturb_comparison.png", "WL-underdrive read-disturb comparison."],
        ["6", "array/baseline_nvsim_macro_summary.png", "One-table baseline macro snapshot for the direct NVSim anchor."],
        ["7", "array/baseline_nvsim_latency_breakdown.png", "Row-decode / bitline-path / other latency breakdown."],
        ["8", "array/high_vt_leakage_proxy.png", "Only macro-level High-Vt change: bridged leakage estimate versus baseline."],
        ["9", "array/negative_bl_write_latency_proxy.png", "Only macro-level Negative-BL change: bridged write-latency estimate versus baseline."],
    ]
    appendix_rows = [
        ["A1", "bitcell/baseline/baseline_hold_snm_all_corners.png", "Baseline hold-SNM butterfly overlays."],
        ["A2", "bitcell/baseline/baseline_read_snm_all_corners.png", "Baseline read-SNM butterfly overlays."],
        ["A3", "bitcell/baseline/baseline_write_nm_all_corners.png", "Baseline WNM butterfly overlays."],
        ["A4", "bitcell/high_vt/high_vt_hold_snm_all_corners.png", "High-Vt hold-SNM butterfly overlays."],
        ["A5", "bitcell/high_vt/high_vt_read_snm_all_corners.png", "High-Vt read-SNM butterfly overlays."],
        ["A6", "bitcell/negative_bitline/negative_bitline_write_delay_grid.png", "Negative-BL waveform delay grid."],
        ["A7", "bitcell/wordline_underdrive/wordline_underdrive_read_snm_all_corners.png", "WL-underdrive all-corner read-SNM overlay."],
        ["A8", "bitcell/high_vt/high_vt_hold_leakage_energy_by_corner.png", "Detailed High-Vt hold-window energy bars."],
    ]
    lines = [
        "# Figure Index",
        "",
        "This file maps the planned report figures to the generated assets under `sram_analysis_plots/report_figures/`.",
        "",
        "## Main Text",
        "",
        markdown_table(["Figure", "Asset", "Role in report"], main_text_rows),
        "",
        "## Appendix",
        "",
        markdown_table(["Figure", "Asset", "Role in report"], appendix_rows),
    ]
    path.write_text("\n".join(lines) + "\n")


def write_report_outline(path: Path) -> None:
    lines = [
        "# SRAM Report Outline",
        "",
        "## 1. Bitcell Methodology And Extraction Rules",
        "",
        "- Reference `report_figures/bitcell/bitcell_summary.md` for the canonical metrics table.",
        "- State explicitly that read/hold SNM come from limiting-eye square fits and WNM uses the current diagonal-corner convention.",
        "- Describe the WL-underdrive read-disturb value as a pulse-window transient metric.",
        "",
        "## 2. Baseline Bitcell Stability And Writability",
        "",
        "- Main figure: `report_figures/bitcell/bitcell_results_summary.png`.",
        "- Appendix support: baseline hold/read/WNM overlays.",
        "",
        "## 3. Optimization-Specific Bitcell Results",
        "",
        "- High Vt: `report_figures/bitcell/high_vt/high_vt_tradeoff_summary.png` plus appendix overlays.",
        "- Negative BL: `report_figures/bitcell/negative_bitline/negative_bitline_write_delay_by_corner.png` and waveform grid appendix.",
        "- WL underdrive: read-SNM and read-disturb comparison figures plus the all-corner overlay appendix.",
        "",
        "## 4. Baseline Array-Level NVSim Model",
        "",
        "- Main figures: `report_figures/array/baseline_nvsim_macro_summary.png` and `report_figures/array/baseline_nvsim_latency_breakdown.png`.",
        "- Cite `array/data/nvsim_baseline_macro.csv` and `array/data/nvsim_baseline_latency_breakdown.csv` directly.",
        "",
        "## 5. Bridge Methodology",
        "",
        "- Explain the three evidence labels: `Estimated by NVSim`, `Derived proxy`, and `Measured in Cadence`.",
        "- Point to `array/docs/bridge_methodology.md` and `array/data/array_bridge_detail.csv`.",
        "",
        "## 6. Optimization-Aware Macro Discussion",
        "",
        "- Main figures: `report_figures/array/high_vt_leakage_proxy.png` and `report_figures/array/negative_bl_write_latency_proxy.png`.",
        "- Emphasize that High-Vt changes leakage only and Negative BL changes write latency only in the current bridge.",
        "- Keep WL underdrive in the bitcell results section because its defended benefits are cell-level stability metrics, not a justified NVSim macro shift.",
        "",
        "## 7. Limitations And Evidence Boundaries",
        "",
        "- Note that the archived `old_nvsim/` material is not part of the live report evidence.",
        "- Note that Negative BL and WL underdrive were not directly simulated as full-array NVSim waveform cases.",
        "- Note that TT nominal is the bridge anchor while corner robustness remains in the bitcell section.",
    ]
    path.write_text("\n".join(lines) + "\n")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


def verify_outputs(entries: list[GuideEntry]) -> None:
    for entry in entries:
        png_path = REPORT_FIGURES_ROOT / entry.relative_png
        pdf_path = png_path.with_suffix(".pdf")
        if not png_path.exists() or not pdf_path.exists():
            raise FileNotFoundError(f"Missing expected figure outputs for {entry.relative_png}")
    for path in (ARRAY_GUIDE_PATH, FIGURE_INDEX_PATH, REPORT_OUTLINE_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing expected documentation output: {path}")


if __name__ == "__main__":
    main()
