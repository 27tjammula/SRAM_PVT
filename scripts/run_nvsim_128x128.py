#!/usr/bin/env python3
"""Run and summarize NVSim for the 2 KB 128x128 45 nm SRAM configs.

The script runs the native NVSim roadmap configs and also generates calibrated
per-corner configs from the Cadence-derived bitcell metrics in rw_metrics.csv.
"""

from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NVSIM_DIR = REPO_ROOT / "nvsim"
OUTPUT_DIR = REPO_ROOT / "sram_analysis_plots" / "nvsim_128x128_45nm"
RW_METRICS_CSV = REPO_ROOT / "sram_analysis_plots" / "rw_metrics.csv"
CALIBRATED_BASE_ROADMAP = "LSTP"

ROADMAP_CONFIGS = [
    ("HP", "2kb_128x128_45nm_hp.cfg", "hp.txt"),
    ("LSTP", "2kb_128x128_45nm_lstp.cfg", "lstp.txt"),
    ("LOP", "2kb_128x128_45nm_lop.cfg", "lop.txt"),
]


@dataclass(frozen=True)
class NvsimMetrics:
    label: str
    subarray_rows: int
    subarray_columns: int
    area_um2: float
    area_efficiency_percent: float
    read_latency_ps: float
    write_latency_ps: float
    predecoder_latency_ps: float
    subarray_read_latency_ps: float
    row_decoder_latency_ps: float
    bitline_latency_ps: float
    senseamp_latency_ps: float
    precharge_latency_ps: float
    charge_latency_ps: float
    read_energy_pj: float
    write_energy_pj: float
    row_decoder_read_energy_pj: float
    row_decoder_write_energy_pj: float
    precharge_energy_pj: float
    senseamp_energy_pj: float
    leakage_power_uw: float


@dataclass(frozen=True)
class CornerSourceMetrics:
    corner: str
    vdd_v: float
    write_delay_ps: float
    standby_leakage_na: float


@dataclass(frozen=True)
class CornerCalibration:
    corner: str
    base_roadmap: str
    vdd_override_v: float
    ion_scale: float
    ioff_scale: float
    config_name: str
    output_name: str


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    roadmap_metrics: list[NvsimMetrics] = []

    for roadmap, config_name, output_name in ROADMAP_CONFIGS:
        output_path = OUTPUT_DIR / output_name
        run_nvsim(config_name, output_path)
        roadmap_metrics.append(parse_output(roadmap, output_path.read_text()))

    write_csv(roadmap_metrics, OUTPUT_DIR / "summary.csv", label_header="roadmap")
    write_markdown(roadmap_metrics, OUTPUT_DIR / "summary.md")
    print("Roadmap configs:")
    print(format_markdown_table(roadmap_metrics, label_header="roadmap"))

    calibrations = build_corner_calibrations()
    calibrated_metrics: list[NvsimMetrics] = []
    for calibration in calibrations:
        write_calibrated_config(calibration)
        output_path = OUTPUT_DIR / calibration.output_name
        run_nvsim(calibration.config_name, output_path)
        calibrated_metrics.append(parse_output(calibration.corner, output_path.read_text()))

    write_calibrated_csv(calibrations, calibrated_metrics, OUTPUT_DIR / "calibrated_summary.csv")
    write_calibrated_markdown(calibrations, calibrated_metrics, OUTPUT_DIR / "calibrated_summary.md")
    print("\nCadence-calibrated corner configs:")
    print(format_calibrated_table(calibrations, calibrated_metrics))
    print(f"\nWrote NVSim outputs to {OUTPUT_DIR}")


def run_nvsim(config_name: str, output_path: Path) -> None:
    completed = subprocess.run(
        ["./nvsim", config_name],
        cwd=NVSIM_DIR,
        check=True,
        text=True,
        capture_output=True,
    )
    output_path.write_text(completed.stdout)


def parse_output(label: str, text: str) -> NvsimMetrics:
    rows, columns = subarray_size(text)
    return NvsimMetrics(
        label=label,
        subarray_rows=rows,
        subarray_columns=columns,
        area_um2=number_after(text, r"Total Area = .* = ([0-9.]+)um\^2"),
        area_efficiency_percent=number_after(text, r"Area Efficiency = ([0-9.]+)%"),
        read_latency_ps=time_after(text, r" -  Read Latency = ([0-9.]+)([a-zA-Z]+)"),
        write_latency_ps=time_after(text, r" - Write Latency = ([0-9.]+)([a-zA-Z]+)"),
        predecoder_latency_ps=time_after(text, r"Predecoder Latency = ([0-9.]+)([a-zA-Z]+)"),
        subarray_read_latency_ps=time_after(text, r"Subarray Latency   = ([0-9.]+)([a-zA-Z]+)"),
        row_decoder_latency_ps=time_after(text, r"Row Decoder Latency = ([0-9.]+)([a-zA-Z]+)"),
        bitline_latency_ps=time_after(text, r"Bitline Latency     = ([0-9.]+)([a-zA-Z]+)"),
        senseamp_latency_ps=time_after(text, r"Senseamp Latency    = ([0-9.]+)([a-zA-Z]+)"),
        precharge_latency_ps=time_after(text, r"Precharge Latency   = ([0-9.]+)([a-zA-Z]+)"),
        charge_latency_ps=time_after(text, r"Charge Latency      = ([0-9.]+)([a-zA-Z]+)"),
        read_energy_pj=energy_after(text, r" -  Read Dynamic Energy = ([0-9.]+)([a-zA-Z]+)"),
        write_energy_pj=energy_after(text, r" - Write Dynamic Energy = ([0-9.]+)([a-zA-Z]+)"),
        row_decoder_read_energy_pj=energy_after(text, r"Row Decoder Dynamic Energy = ([0-9.]+)([a-zA-Z]+)"),
        row_decoder_write_energy_pj=energy_after_last(text, r"Row Decoder Dynamic Energy = ([0-9.]+)([a-zA-Z]+)"),
        precharge_energy_pj=energy_after(text, r"Precharge Dynamic Energy   = ([0-9.]+)([a-zA-Z]+)"),
        senseamp_energy_pj=energy_after(text, r"Senseamp Dynamic Energy    = ([0-9.]+)([a-zA-Z]+)"),
        leakage_power_uw=power_after(text, r" - Leakage Power = ([0-9.]+)([a-zA-Z]+)"),
    )


def subarray_size(text: str) -> tuple[int, int]:
    match = re.search(r"Subarray Size\s+: ([0-9]+) Rows x ([0-9]+) Columns", text)
    if not match:
        raise ValueError("Could not parse subarray size")
    return int(match.group(1)), int(match.group(2))


def build_corner_calibrations() -> list[CornerCalibration]:
    source_metrics = read_corner_metrics()
    if "tt" not in source_metrics:
        raise ValueError("rw_metrics.csv must include a tt row to use as the calibration baseline")

    baseline = source_metrics["tt"]
    if baseline.write_delay_ps <= 0 or baseline.standby_leakage_na <= 0:
        raise ValueError("tt write delay and leakage must be positive")

    calibrations: list[CornerCalibration] = []
    for corner, metric in source_metrics.items():
        if metric.write_delay_ps <= 0 or metric.standby_leakage_na <= 0:
            raise ValueError(f"{corner} write delay and leakage must be positive")
        safe_corner = corner.replace("/", "_")
        calibrations.append(
            CornerCalibration(
                corner=corner,
                base_roadmap=CALIBRATED_BASE_ROADMAP,
                vdd_override_v=metric.vdd_v,
                ion_scale=baseline.write_delay_ps / metric.write_delay_ps,
                ioff_scale=metric.standby_leakage_na / baseline.standby_leakage_na,
                config_name=f"2kb_128x128_45nm_cal_{safe_corner}.cfg",
                output_name=f"cal_{safe_corner}.txt",
            )
        )
    return calibrations


def read_corner_metrics() -> dict[str, CornerSourceMetrics]:
    if not RW_METRICS_CSV.exists():
        raise FileNotFoundError(f"Could not find {RW_METRICS_CSV}")

    metrics: dict[str, CornerSourceMetrics] = {}
    with RW_METRICS_CSV.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            corner = row["corner"]
            metrics[corner] = CornerSourceMetrics(
                corner=corner,
                vdd_v=float(row["vdd_V"]),
                write_delay_ps=float(row["write_delay_ps"]),
                standby_leakage_na=float(row["standby_leakage_nA"]),
            )
    return metrics


def write_calibrated_config(calibration: CornerCalibration) -> None:
    config_path = NVSIM_DIR / calibration.config_name
    config_text = f"""// 2 KB SRAM array: 128 rows x 128 columns, 45 nm {calibration.corner} calibrated corner
// Generated by scripts/run_nvsim_128x128.py from sram_analysis_plots/rw_metrics.csv.
// Calibration model: common {calibration.base_roadmap} base, VDD override plus Ion/Ioff multipliers.
-DesignTarget: RAM
-CacheAccessMode: Normal
-OptimizationTarget: Area
-EnablePruning: Yes

-ProcessNode: 45
-Capacity (KB): 2
-WordWidth (bit): 128

-DeviceRoadmap: {calibration.base_roadmap}
-VddOverride (V): {calibration.vdd_override_v:.6g}
-IonScale: {calibration.ion_scale:.6f}
-IoffScale: {calibration.ioff_scale:.6f}

-LocalWireType: LocalAggressive
-LocalWireRepeaterType: RepeatedNone
-LocalWireUseLowSwing: No

-GlobalWireType: GlobalAggressive
-GlobalWireRepeaterType: RepeatedNone
-GlobalWireUseLowSwing: No

-Routing: H-tree
-InternalSensing: true
-MemoryCellInputFile: SRAM.cell
-Temperature (K): 350

-BufferDesignOptimization: latency

// Force one 128x128 subarray: 1 bank, 1 mat, 1 subarray, no column muxing.
-ForceBank (Total AxB, Active CxD): 1x1, 1x1
-ForceMat (Total AxB, Active CxD): 1x1, 1x1
-ForceMuxSenseAmp: 1
-ForceMuxOutputLev1: 1
-ForceMuxOutputLev2: 1
"""
    config_path.write_text(config_text)


def number_after(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse pattern: {pattern}")
    return float(match.group(1))


def time_after(text: str, pattern: str) -> float:
    return convert_with_unit(text, pattern, {"ps": 1.0, "ns": 1e3, "us": 1e6})


def energy_after(text: str, pattern: str) -> float:
    return convert_with_unit(text, pattern, {"fJ": 1e-3, "pJ": 1.0, "nJ": 1e3})


def energy_after_last(text: str, pattern: str) -> float:
    return convert_with_unit(text, pattern, {"fJ": 1e-3, "pJ": 1.0, "nJ": 1e3}, last=True)


def power_after(text: str, pattern: str) -> float:
    return convert_with_unit(text, pattern, {"pW": 1e-6, "nW": 1e-3, "uW": 1.0, "mW": 1e3})


def convert_with_unit(text: str, pattern: str, units: dict[str, float], last: bool = False) -> float:
    matches = list(re.finditer(pattern, text))
    if not matches:
        raise ValueError(f"Could not parse pattern: {pattern}")
    match = matches[-1] if last else matches[0]
    value = float(match.group(1))
    unit = match.group(2)
    if unit not in units:
        raise ValueError(f"Unsupported unit {unit!r} for pattern {pattern}")
    return value * units[unit]


def write_csv(metrics: list[NvsimMetrics], path: Path, label_header: str) -> None:
    rows = [metric_row(metric, label_header) for metric in metrics]
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(metrics: list[NvsimMetrics], path: Path) -> None:
    lines = [
        "# NVSim 128x128 45 nm SRAM Summary",
        "",
        "Generated from the dedicated `2kb_128x128_45nm_*.cfg` files in `nvsim/`.",
        "Each config forces one 128-row by 128-column subarray with no column muxing.",
        "",
        format_markdown_table(metrics, label_header="roadmap"),
        "",
        "Raw outputs:",
        "- `hp.txt`",
        "- `lstp.txt`",
        "- `lop.txt`",
        "",
    ]
    path.write_text("\n".join(lines))


def write_calibrated_csv(
    calibrations: list[CornerCalibration], metrics: list[NvsimMetrics], path: Path
) -> None:
    rows = calibrated_rows(calibrations, metrics)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_calibrated_markdown(
    calibrations: list[CornerCalibration], metrics: list[NvsimMetrics], path: Path
) -> None:
    raw_outputs = "\n".join(f"- `{calibration.output_name}`" for calibration in calibrations)
    config_files = "\n".join(f"- `nvsim/{calibration.config_name}`" for calibration in calibrations)
    lines = [
        "# Cadence-Calibrated NVSim 128x128 45 nm SRAM Summary",
        "",
        "Generated from `sram_analysis_plots/rw_metrics.csv` using TT as the calibration baseline.",
        f"All calibrated corners use the same `{CALIBRATED_BASE_ROADMAP}` NVSim device roadmap, then apply corner-specific VDD, Ion, and Ioff scaling.",
        "",
        format_calibrated_table(calibrations, metrics),
        "",
        "Generated configs:",
        config_files,
        "",
        "Raw outputs:",
        raw_outputs,
        "",
        "Calibration formulas:",
        "- `IonScale = tt_write_delay_ps / corner_write_delay_ps`",
        "- `IoffScale = corner_standby_leakage_nA / tt_standby_leakage_nA`",
        "- `VddOverride = corner_vdd_V`",
        "",
    ]
    path.write_text("\n".join(lines))


def metric_row(metric: NvsimMetrics, label_header: str) -> dict[str, str]:
    return {
        label_header: metric.label,
        "subarray": f"{metric.subarray_rows}x{metric.subarray_columns}",
        "area_um2": f"{metric.area_um2:.3f}",
        "area_eff_%": f"{metric.area_efficiency_percent:.3f}",
        "read_lat_ps": f"{metric.read_latency_ps:.3f}",
        "write_lat_ps": f"{metric.write_latency_ps:.3f}",
        "bitline_lat_ps": f"{metric.bitline_latency_ps:.3f}",
        "rowdec_lat_ps": f"{metric.row_decoder_latency_ps:.3f}",
        "read_energy_pJ": f"{metric.read_energy_pj:.3f}",
        "write_energy_pJ": f"{metric.write_energy_pj:.3f}",
        "leakage_uW": f"{metric.leakage_power_uw:.3f}",
    }


def calibrated_rows(
    calibrations: list[CornerCalibration], metrics: list[NvsimMetrics]
) -> list[dict[str, str]]:
    metrics_by_corner = {metric.label: metric for metric in metrics}
    rows: list[dict[str, str]] = []
    for calibration in calibrations:
        metric = metrics_by_corner[calibration.corner]
        row = {
            "corner": calibration.corner,
            "base": calibration.base_roadmap,
            "vdd_V": f"{calibration.vdd_override_v:.3f}",
            "ion_scale": f"{calibration.ion_scale:.3f}",
            "ioff_scale": f"{calibration.ioff_scale:.3f}",
        }
        metric_values = metric_row(metric, label_header="corner")
        metric_values.pop("corner")
        row.update(metric_values)
        rows.append(row)
    return rows


def format_markdown_table(metrics: list[NvsimMetrics], label_header: str) -> str:
    rows = [metric_row(metric, label_header) for metric in metrics]
    return format_rows_table(rows)


def format_calibrated_table(
    calibrations: list[CornerCalibration], metrics: list[NvsimMetrics]
) -> str:
    return format_rows_table(calibrated_rows(calibrations, metrics))


def format_rows_table(rows: list[dict[str, str]]) -> str:
    headers = list(rows[0].keys())
    widths = {header: max(len(header), *(len(row[header]) for row in rows)) for header in headers}
    lines = [
        "| " + " | ".join(header.ljust(widths[header]) for header in headers) + " |",
        "| " + " | ".join("-" * widths[header] for header in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[header].ljust(widths[header]) for header in headers) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
