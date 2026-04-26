#!/usr/bin/env python3
"""Run a fresh NVSim baseline and build the bridged array-report artifacts."""

from __future__ import annotations

import csv
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plot_bitcell_report_figures import collect_bitcell_metrics, corner_label, format_float


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DATA_ROOT = REPO_ROOT / "sim_data"
ARRAY_ROOT = REPO_ROOT / "sram_analysis_plots" / "array"
DATA_DIR = ARRAY_ROOT / "data"
DOCS_DIR = ARRAY_ROOT / "docs"
LOGS_DIR = ARRAY_ROOT / "logs"
NVSIM_DIR = REPO_ROOT / "nvsim"
NVSIM_EXECUTABLE = NVSIM_DIR / "nvsim"
TOP_LEVEL_SUMMARY_PATH = REPO_ROOT / "sram_analysis_plots" / "array_extrapolation_128x128_45nm.md"


@dataclass(frozen=True)
class SramMacroConfig:
    capacity_bytes: int = 2048
    total_bits: int = 16384
    rows: int = 128
    columns: int = 128
    word_width_bits: int = 8
    banks: int = 1
    technology_node_nm: int = 45
    device_roadmap: str = "LSTP"
    memory_cell_input_file: str = "SRAM.cell"
    temperature_k: int = 350
    local_wire_type: str = "LocalAggressive"
    local_wire_repeater_type: str = "RepeatedNone"
    local_wire_use_low_swing: bool = False
    global_wire_type: str = "GlobalAggressive"
    global_wire_repeater_type: str = "RepeatedNone"
    global_wire_use_low_swing: bool = False
    routing: str = "H-tree"
    internal_sensing: bool = True
    buffer_design_optimization: str = "latency"
    bank_rows: int = 1
    bank_columns: int = 1
    bank_active_columns: int = 1
    bank_active_rows: int = 1
    mat_rows: int = 1
    mat_columns: int = 1
    mat_active_columns: int = 1
    mat_active_rows: int = 1
    mux_sense_amp: int = 16
    mux_output_level1: int = 1
    mux_output_level2: int = 1

    @property
    def capacity_kb(self) -> int:
        return self.capacity_bytes // 1024

    @property
    def column_select_count(self) -> int:
        return self.columns // self.word_width_bits

    @property
    def row_address_bits(self) -> int:
        return int(math.log2(self.rows))

    def validate(self) -> None:
        if self.capacity_bytes * 8 != self.total_bits:
            raise ValueError("capacity_bytes does not match total_bits")
        if self.rows * self.columns != self.total_bits:
            raise ValueError("rows * columns does not match total_bits")
        if self.banks != 1:
            raise ValueError("This report flow expects one bank")
        if self.column_select_count != 16:
            raise ValueError("This report flow expects a 16:1 column selection")
        if self.row_address_bits != 7:
            raise ValueError("This report flow expects 7 row-address bits")


@dataclass(frozen=True)
class NvsimMetrics:
    subarray_rows: int
    subarray_columns: int
    area_um2: float
    area_efficiency_percent: float
    read_latency_ps: float
    write_latency_ps: float
    predecoder_latency_ps: float
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


def main() -> None:
    config = SramMacroConfig()
    config.validate()
    ensure_output_structure()

    if not NVSIM_EXECUTABLE.exists():
        raise FileNotFoundError(f"Expected NVSim executable not found: {NVSIM_EXECUTABLE}")

    bitcell_metrics = collect_bitcell_metrics(SIM_DATA_ROOT)
    baseline_text, baseline_metrics, config_path, output_path = run_baseline_nvsim(config)
    validate_baseline_metrics(config, baseline_metrics)

    baseline_macro_rows = [build_baseline_macro_row(config, baseline_metrics, config_path, output_path)]
    latency_breakdown_rows = build_latency_breakdown_rows(baseline_metrics)
    bridge_detail_rows = build_bridge_detail_rows(config, baseline_metrics, bitcell_metrics)
    bridge_case_rows = build_bridge_case_summary_rows(baseline_metrics, bitcell_metrics)

    write_csv_rows(DATA_DIR / "nvsim_baseline_macro.csv", baseline_macro_rows)
    write_csv_rows(DATA_DIR / "nvsim_baseline_latency_breakdown.csv", latency_breakdown_rows)
    write_csv_rows(DATA_DIR / "array_bridge_detail.csv", bridge_detail_rows)
    write_csv_rows(DATA_DIR / "array_bridge_case_summary.csv", bridge_case_rows)

    write_array_methodology_md(config, baseline_metrics)
    write_bridge_methodology_md(config, bridge_detail_rows)
    write_phase_summary_md(config, baseline_metrics, bridge_case_rows)
    write_array_extrapolation_md(config, baseline_metrics, bridge_case_rows, baseline_text)

    print(DATA_DIR.relative_to(REPO_ROOT) / "nvsim_baseline_macro.csv")
    print(DATA_DIR.relative_to(REPO_ROOT) / "nvsim_baseline_latency_breakdown.csv")
    print(DATA_DIR.relative_to(REPO_ROOT) / "array_bridge_detail.csv")
    print(DATA_DIR.relative_to(REPO_ROOT) / "array_bridge_case_summary.csv")
    print(DOCS_DIR.relative_to(REPO_ROOT) / "array_methodology.md")
    print(DOCS_DIR.relative_to(REPO_ROOT) / "bridge_methodology.md")
    print(DOCS_DIR.relative_to(REPO_ROOT) / "nvsim_phase_summary.md")
    print(TOP_LEVEL_SUMMARY_PATH.relative_to(REPO_ROOT))


def ensure_output_structure() -> None:
    for path in (ARRAY_ROOT, DATA_DIR, DOCS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "configs").mkdir(parents=True, exist_ok=True)


def run_baseline_nvsim(config: SramMacroConfig) -> tuple[str, NvsimMetrics, Path, Path]:
    config_text = build_nvsim_config_text(config)
    config_path = LOGS_DIR / "configs" / "baseline.cfg"
    output_path = LOGS_DIR / "baseline_nvsim_output.txt"
    config_path.write_text(config_text)

    completed = subprocess.run(
        [str(NVSIM_EXECUTABLE), str(config_path)],
        cwd=NVSIM_DIR,
        text=True,
        capture_output=True,
        timeout=120,
    )
    raw_text = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
    output_path.write_text(raw_text)
    if completed.returncode != 0:
        raise RuntimeError(f"NVSim baseline run failed. See {output_path}")

    metrics = parse_nvsim_output(completed.stdout)
    return completed.stdout, metrics, config_path, output_path


def build_nvsim_config_text(config: SramMacroConfig) -> str:
    lines = [
        "// Generated by scripts/run_nvsim_array_analysis.py",
        f"// Fresh baseline for the {config.capacity_bytes}-byte SRAM report flow",
        "-DesignTarget: RAM",
        "-CacheAccessMode: Normal",
        "-OptimizationTarget: Area",
        "-EnablePruning: Yes",
        "",
        f"-ProcessNode: {config.technology_node_nm}",
        f"-Capacity (KB): {config.capacity_kb}",
        f"-WordWidth (bit): {config.word_width_bits}",
        "",
        f"-DeviceRoadmap: {config.device_roadmap}",
        "",
        f"-LocalWireType: {config.local_wire_type}",
        f"-LocalWireRepeaterType: {config.local_wire_repeater_type}",
        f"-LocalWireUseLowSwing: {'Yes' if config.local_wire_use_low_swing else 'No'}",
        "",
        f"-GlobalWireType: {config.global_wire_type}",
        f"-GlobalWireRepeaterType: {config.global_wire_repeater_type}",
        f"-GlobalWireUseLowSwing: {'Yes' if config.global_wire_use_low_swing else 'No'}",
        "",
        f"-Routing: {config.routing}",
        f"-InternalSensing: {'true' if config.internal_sensing else 'false'}",
        f"-MemoryCellInputFile: {config.memory_cell_input_file}",
        f"-Temperature (K): {config.temperature_k}",
        "",
        f"-BufferDesignOptimization: {config.buffer_design_optimization}",
        "",
        f"-ForceBank (Total AxB, Active CxD): {config.bank_rows}x{config.bank_columns}, {config.bank_active_columns}x{config.bank_active_rows}",
        f"-ForceMat (Total AxB, Active CxD): {config.mat_rows}x{config.mat_columns}, {config.mat_active_columns}x{config.mat_active_rows}",
        f"-ForceMuxSenseAmp: {config.mux_sense_amp}",
        f"-ForceMuxOutputLev1: {config.mux_output_level1}",
        f"-ForceMuxOutputLev2: {config.mux_output_level2}",
        "",
        f"// Physical organization target: {config.rows} rows x {config.columns} columns = {config.total_bits} bits",
        f"// Column selection target: {config.column_select_count}:1 for {config.word_width_bits}-bit words",
        f"// Row address target: A4-A10 => {config.rows} rows",
    ]
    return "\n".join(lines) + "\n"


def parse_nvsim_output(text: str) -> NvsimMetrics:
    rows, columns = subarray_size(text)
    return NvsimMetrics(
        subarray_rows=rows,
        subarray_columns=columns,
        area_um2=number_after(text, r"Total Area = .* = ([0-9.]+)um\^2"),
        area_efficiency_percent=number_after(text, r"Area Efficiency = ([0-9.]+)%"),
        read_latency_ps=time_after(text, r" -  Read Latency = ([0-9.]+)([a-zA-Z]+)"),
        write_latency_ps=time_after(text, r" - Write Latency = ([0-9.]+)([a-zA-Z]+)"),
        predecoder_latency_ps=time_after(text, r"Predecoder Latency = ([0-9.]+)([a-zA-Z]+)"),
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
        raise ValueError("Could not parse NVSim subarray size")
    return int(match.group(1)), int(match.group(2))


def number_after(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not parse NVSim pattern: {pattern}")
    return float(match.group(1))


def time_after(text: str, pattern: str) -> float:
    return convert_with_units(text, pattern, {"ps": 1.0, "ns": 1e3, "us": 1e6})


def energy_after(text: str, pattern: str) -> float:
    return convert_with_units(text, pattern, {"fJ": 1e-3, "pJ": 1.0, "nJ": 1e3})


def energy_after_last(text: str, pattern: str) -> float:
    return convert_with_units(text, pattern, {"fJ": 1e-3, "pJ": 1.0, "nJ": 1e3}, last=True)


def power_after(text: str, pattern: str) -> float:
    return convert_with_units(text, pattern, {"pW": 1e-6, "nW": 1e-3, "uW": 1.0, "mW": 1e3})


def convert_with_units(text: str, pattern: str, units: dict[str, float], *, last: bool = False) -> float:
    matches = list(re.finditer(pattern, text))
    if not matches:
        raise ValueError(f"Could not parse NVSim pattern: {pattern}")
    match = matches[-1] if last else matches[0]
    unit = match.group(2)
    if unit not in units:
        raise ValueError(f"Unsupported unit {unit!r} in pattern {pattern}")
    return float(match.group(1)) * units[unit]


def validate_baseline_metrics(config: SramMacroConfig, metrics: NvsimMetrics) -> None:
    if metrics.subarray_rows != config.rows or metrics.subarray_columns != config.columns:
        raise ValueError(
            f"NVSim produced {metrics.subarray_rows}x{metrics.subarray_columns}, expected {config.rows}x{config.columns}"
        )
    for name, value in {
        "area_um2": metrics.area_um2,
        "read_latency_ps": metrics.read_latency_ps,
        "write_latency_ps": metrics.write_latency_ps,
        "read_energy_pj": metrics.read_energy_pj,
        "write_energy_pj": metrics.write_energy_pj,
        "leakage_power_uw": metrics.leakage_power_uw,
    }.items():
        if value < 0.0:
            raise ValueError(f"Invalid NVSim baseline metric: {name} is negative")


def build_baseline_macro_row(
    config: SramMacroConfig,
    metrics: NvsimMetrics,
    config_path: Path,
    output_path: Path,
) -> dict[str, str]:
    return {
        "capacity_bytes": str(config.capacity_bytes),
        "total_bits": str(config.total_bits),
        "rows": str(config.rows),
        "columns": str(config.columns),
        "word_width_bits": str(config.word_width_bits),
        "column_mux_factor": str(config.column_select_count),
        "banks": str(config.banks),
        "technology_node_nm": str(config.technology_node_nm),
        "device_roadmap": config.device_roadmap,
        "subarray_rows": str(metrics.subarray_rows),
        "subarray_columns": str(metrics.subarray_columns),
        "area_um2": format_float(metrics.area_um2),
        "area_mm2": format_float(metrics.area_um2 / 1e6),
        "area_efficiency_percent": format_float(metrics.area_efficiency_percent),
        "read_latency_ns": format_float(metrics.read_latency_ps / 1000.0),
        "write_latency_ns": format_float(metrics.write_latency_ps / 1000.0),
        "read_energy_pj": format_float(metrics.read_energy_pj),
        "write_energy_pj": format_float(metrics.write_energy_pj),
        "leakage_power_uw": format_float(metrics.leakage_power_uw),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "raw_output_path": str(output_path.relative_to(REPO_ROOT)),
    }


def build_latency_breakdown_rows(metrics: NvsimMetrics) -> list[dict[str, str]]:
    read_decode_ns = (metrics.predecoder_latency_ps + metrics.row_decoder_latency_ps) / 1000.0
    read_bitline_ns = (metrics.bitline_latency_ps + metrics.senseamp_latency_ps) / 1000.0
    read_other_ns = max(0.0, metrics.read_latency_ps / 1000.0 - read_decode_ns - read_bitline_ns)

    write_decode_ns = (metrics.predecoder_latency_ps + metrics.row_decoder_latency_ps) / 1000.0
    write_bitline_ns = metrics.charge_latency_ps / 1000.0
    write_other_ns = max(0.0, metrics.write_latency_ps / 1000.0 - write_decode_ns - write_bitline_ns)

    rows: list[dict[str, str]] = []
    for access, total_ns, row_decode_ns, bitline_ns, other_ns in (
        ("Read", metrics.read_latency_ps / 1000.0, read_decode_ns, read_bitline_ns, read_other_ns),
        ("Write", metrics.write_latency_ps / 1000.0, write_decode_ns, write_bitline_ns, write_other_ns),
    ):
        for component, latency_ns, rule in (
            ("Row decode", row_decode_ns, "Predecoder + row decoder"),
            ("Bitline path", bitline_ns, "Read: bitline + sense amp; Write: charge latency"),
            ("Other", other_ns, "Residual to total latency"),
        ):
            rows.append(
                {
                    "access": access,
                    "component": component,
                    "latency_ns": format_float(latency_ns),
                    "share_percent": format_float(100.0 * latency_ns / total_ns if total_ns else 0.0),
                    "aggregation_rule": rule,
                    "evidence_type": "Estimated by NVSim",
                }
            )
    return rows


def build_bridge_detail_rows(
    config: SramMacroConfig,
    baseline_metrics: NvsimMetrics,
    bitcell_metrics: dict[str, Any],
) -> list[dict[str, str]]:
    baseline_write_ratio = (
        bitcell_metrics["negative_bl_write_delay_ps"]["tt"]["optimized_value"]
        / bitcell_metrics["negative_bl_write_delay_ps"]["tt"]["baseline_value"]
    )
    high_vt_leakage_ratio = (
        bitcell_metrics["high_vt_hold_energy_aj"]["tt"]["optimized_value"]
        / bitcell_metrics["high_vt_hold_energy_aj"]["tt"]["baseline_value"]
    )

    baseline_area_mm2 = baseline_metrics.area_um2 / 1e6
    baseline_read_latency_ns = baseline_metrics.read_latency_ps / 1000.0
    baseline_write_latency_ns = baseline_metrics.write_latency_ps / 1000.0

    rows: list[dict[str, str]] = []
    case_names = ["Baseline", "High Vt", "Negative BL", "WL Underdrive"]
    for case in case_names:
        rows.extend(
            macro_quantity_rows(
                case=case,
                area_mm2=baseline_area_mm2,
                read_latency_ns=baseline_read_latency_ns,
                write_latency_ns=(
                    baseline_write_latency_ns * baseline_write_ratio if case == "Negative BL" else baseline_write_latency_ns
                ),
                read_energy_pj=baseline_metrics.read_energy_pj,
                write_energy_pj=baseline_metrics.write_energy_pj,
                leakage_power_uw=(
                    baseline_metrics.leakage_power_uw * high_vt_leakage_ratio
                    if case == "High Vt"
                    else baseline_metrics.leakage_power_uw
                ),
                write_latency_type="Derived proxy" if case == "Negative BL" else "Estimated by NVSim",
                leakage_type="Derived proxy" if case == "High Vt" else "Estimated by NVSim",
                baseline_metrics=baseline_metrics,
                high_vt_leakage_ratio=high_vt_leakage_ratio,
                negative_bl_write_ratio=baseline_write_ratio,
            )
        )
        rows.extend(bitcell_support_rows(case, bitcell_metrics))

    rows.insert(
        0,
        {
            "case": "Report anchor",
            "quantity_group": "Methodology",
            "quantity_name": "Bridge anchor",
            "value": "TT nominal",
            "unit": "label",
            "evidence_type": "Measured in Cadence",
            "reference_case": "NA",
            "reference_value": "NA",
            "delta_percent": "NA",
            "derivation": "TT nominal bitcell values are the report-body bridge anchor; corner robustness remains in the bitcell section.",
            "source_files": "sim_data/...; sram_analysis_plots/report_figures/bitcell/bitcell_summary.csv",
            "notes": f"Array target fixed at {config.rows}x{config.columns}, {config.word_width_bits}-bit words, {config.column_select_count}:1 column selection.",
        },
    )
    return rows


def macro_quantity_rows(
    *,
    case: str,
    area_mm2: float,
    read_latency_ns: float,
    write_latency_ns: float,
    read_energy_pj: float,
    write_energy_pj: float,
    leakage_power_uw: float,
    write_latency_type: str,
    leakage_type: str,
    baseline_metrics: NvsimMetrics,
    high_vt_leakage_ratio: float,
    negative_bl_write_ratio: float,
) -> list[dict[str, str]]:
    shared_source = "sram_analysis_plots/array/data/nvsim_baseline_macro.csv"
    baseline_write_latency_ns = baseline_metrics.write_latency_ps / 1000.0
    baseline_leakage_uw = baseline_metrics.leakage_power_uw
    return [
        bridge_row(case, "Macro", "Area", area_mm2, "mm^2", "Estimated by NVSim", "NA", "NA", "NA", f"Direct baseline NVSim area estimate reused for {case}.", shared_source),
        bridge_row(case, "Macro", "Read latency", read_latency_ns, "ns", "Estimated by NVSim", "Baseline", baseline_metrics.read_latency_ps / 1000.0 if case != "Baseline" else "NA", 0.0 if case != "Baseline" else "NA", f"Direct baseline NVSim read latency reused for {case}.", shared_source),
        bridge_row(case, "Macro", "Write latency", write_latency_ns, "ns", write_latency_type, "Baseline", baseline_write_latency_ns if case != "Baseline" else "NA", percent_delta_value(baseline_write_latency_ns, write_latency_ns) if case != "Baseline" else "NA", write_latency_derivation(case, negative_bl_write_ratio), shared_source),
        bridge_row(case, "Macro", "Read energy", read_energy_pj, "pJ", "Estimated by NVSim", "Baseline", baseline_metrics.read_energy_pj if case != "Baseline" else "NA", 0.0 if case != "Baseline" else "NA", f"Direct baseline NVSim read energy reused for {case}.", shared_source),
        bridge_row(case, "Macro", "Write energy", write_energy_pj, "pJ", "Estimated by NVSim", "Baseline", baseline_metrics.write_energy_pj if case != "Baseline" else "NA", 0.0 if case != "Baseline" else "NA", f"Direct baseline NVSim write energy reused for {case}.", shared_source),
        bridge_row(case, "Macro", "Leakage power", leakage_power_uw, "uW", leakage_type, "Baseline", baseline_leakage_uw if case != "Baseline" else "NA", percent_delta_value(baseline_leakage_uw, leakage_power_uw) if case != "Baseline" else "NA", leakage_derivation(case, high_vt_leakage_ratio), shared_source),
    ]


def write_latency_derivation(case: str, negative_bl_write_ratio: float) -> str:
    if case == "Negative BL":
        return f"Baseline macro write latency scaled by the TT Cadence write-delay ratio ({negative_bl_write_ratio:.4f})."
    return f"No supported NVSim assist knob for {case}; baseline write latency is intentionally unchanged."


def leakage_derivation(case: str, high_vt_leakage_ratio: float) -> str:
    if case == "High Vt":
        return f"Baseline macro leakage scaled by the TT hold-window energy ratio ({high_vt_leakage_ratio:.4f}) as a leakage-oriented proxy."
    return f"No direct leakage change is applied for {case}; baseline leakage is intentionally reused."


def bridge_row(
    case: str,
    quantity_group: str,
    quantity_name: str,
    value: float | str,
    unit: str,
    evidence_type: str,
    reference_case: str,
    reference_value: float | str,
    delta_percent: float | str,
    derivation: str,
    source_files: str,
) -> dict[str, str]:
    return {
        "case": case,
        "quantity_group": quantity_group,
        "quantity_name": quantity_name,
        "value": format_optional(value),
        "unit": unit,
        "evidence_type": evidence_type,
        "reference_case": reference_case,
        "reference_value": format_optional(reference_value),
        "delta_percent": format_optional(delta_percent),
        "derivation": derivation,
        "source_files": source_files,
        "notes": "",
    }


def bitcell_support_rows(case: str, bitcell_metrics: dict[str, Any]) -> list[dict[str, str]]:
    source_table = "sram_analysis_plots/report_figures/bitcell/bitcell_summary.csv"
    rows: list[dict[str, str]] = []
    if case == "Baseline":
        rows.extend(
            [
                bridge_row(case, "Bitcell support", "Hold SNM", bitcell_metrics["baseline_hold_snm_mv"]["tt"], "mV", "Measured in Cadence", "NA", "NA", "NA", "TT nominal baseline hold-SNM support.", source_table),
                bridge_row(case, "Bitcell support", "Read SNM", bitcell_metrics["baseline_read_snm_mv"]["tt"], "mV", "Measured in Cadence", "NA", "NA", "NA", "TT nominal baseline read-SNM support.", source_table),
                bridge_row(case, "Bitcell support", "Write Noise Margin", bitcell_metrics["baseline_write_nm_mv"]["tt"], "mV", "Measured in Cadence", "NA", "NA", "NA", "TT nominal baseline WNM support.", source_table),
            ]
        )
    elif case == "High Vt":
        rows.extend(
            [
                bridge_row(case, "Bitcell support", "Hold SNM", bitcell_metrics["high_vt_hold_snm_mv"]["tt"], "mV", "Measured in Cadence", "Baseline", bitcell_metrics["baseline_hold_snm_mv"]["tt"], percent_delta_value(bitcell_metrics["baseline_hold_snm_mv"]["tt"], bitcell_metrics["high_vt_hold_snm_mv"]["tt"]), "High-Vt hold-SNM tradeoff measured directly in Cadence.", source_table),
                bridge_row(case, "Bitcell support", "Read SNM", bitcell_metrics["high_vt_read_snm_mv"]["tt"], "mV", "Measured in Cadence", "Baseline", bitcell_metrics["baseline_read_snm_mv"]["tt"], percent_delta_value(bitcell_metrics["baseline_read_snm_mv"]["tt"], bitcell_metrics["high_vt_read_snm_mv"]["tt"]), "High-Vt read-SNM penalty measured directly in Cadence.", source_table),
                bridge_row(case, "Bitcell support", "Hold-window supply energy", bitcell_metrics["high_vt_hold_energy_aj"]["tt"]["optimized_value"], "aJ", "Measured in Cadence", "Baseline", bitcell_metrics["high_vt_hold_energy_aj"]["tt"]["baseline_value"], bitcell_metrics["high_vt_hold_energy_aj"]["tt"]["delta_percent"], "6-9 ns hold-window supply energy from the TT transient bench.", source_table),
            ]
        )
    elif case == "Negative BL":
        rows.append(
            bridge_row(case, "Bitcell support", "Write delay", bitcell_metrics["negative_bl_write_delay_ps"]["tt"]["optimized_value"], "ps", "Measured in Cadence", "Baseline", bitcell_metrics["negative_bl_write_delay_ps"]["tt"]["baseline_value"], bitcell_metrics["negative_bl_write_delay_ps"]["tt"]["delta_percent"], "TT assisted reverse-write WL-to-Q delay measured directly in Cadence.", source_table)
        )
    elif case == "WL Underdrive":
        rows.extend(
            [
                bridge_row(case, "Bitcell support", "Read SNM", bitcell_metrics["wl_underdrive_read_snm_mv"]["tt"], "mV", "Measured in Cadence", "Baseline", bitcell_metrics["baseline_read_snm_mv"]["tt"], percent_delta_value(bitcell_metrics["baseline_read_snm_mv"]["tt"], bitcell_metrics["wl_underdrive_read_snm_mv"]["tt"]), "Wordline-underdrive read-SNM improvement measured directly in Cadence.", source_table),
                bridge_row(case, "Bitcell support", "Read disturb", bitcell_metrics["wl_underdrive_read_disturb_mv"]["tt"]["optimized_value"], "mV", "Measured in Cadence", "Baseline", bitcell_metrics["wl_underdrive_read_disturb_mv"]["tt"]["baseline_value"], bitcell_metrics["wl_underdrive_read_disturb_mv"]["tt"]["delta_percent"], "Pulse-window transient read-disturb metric measured directly in Cadence.", source_table),
            ]
        )
    return rows


def build_bridge_case_summary_rows(
    baseline_metrics: NvsimMetrics,
    bitcell_metrics: dict[str, Any],
) -> list[dict[str, str]]:
    baseline_write_ratio = (
        bitcell_metrics["negative_bl_write_delay_ps"]["tt"]["optimized_value"]
        / bitcell_metrics["negative_bl_write_delay_ps"]["tt"]["baseline_value"]
    )
    high_vt_leakage_ratio = (
        bitcell_metrics["high_vt_hold_energy_aj"]["tt"]["optimized_value"]
        / bitcell_metrics["high_vt_hold_energy_aj"]["tt"]["baseline_value"]
    )
    rows: list[dict[str, str]] = []
    baseline_common = {
        "macro_area_mm2": baseline_metrics.area_um2 / 1e6,
        "macro_read_latency_ns": baseline_metrics.read_latency_ps / 1000.0,
        "macro_write_latency_ns": baseline_metrics.write_latency_ps / 1000.0,
        "macro_read_energy_pj": baseline_metrics.read_energy_pj,
        "macro_write_energy_pj": baseline_metrics.write_energy_pj,
        "macro_leakage_power_uw": baseline_metrics.leakage_power_uw,
    }
    rows.append(
        case_summary_row(
            case="Baseline",
            macro_values=baseline_common,
            macro_sources={
                "macro_area_mm2": "Estimated by NVSim",
                "macro_read_latency_ns": "Estimated by NVSim",
                "macro_write_latency_ns": "Estimated by NVSim",
                "macro_read_energy_pj": "Estimated by NVSim",
                "macro_write_energy_pj": "Estimated by NVSim",
                "macro_leakage_power_uw": "Estimated by NVSim",
            },
            support_values={
                "support_hold_snm_mv": bitcell_metrics["baseline_hold_snm_mv"]["tt"],
                "support_read_snm_mv": bitcell_metrics["baseline_read_snm_mv"]["tt"],
                "support_write_nm_mv": bitcell_metrics["baseline_write_nm_mv"]["tt"],
            },
            support_sources={
                "support_hold_snm_mv": "Measured in Cadence",
                "support_read_snm_mv": "Measured in Cadence",
                "support_write_nm_mv": "Measured in Cadence",
            },
            headline_metric="Baseline nominal support",
            headline_delta_percent="NA",
            headline_note="TT hold SNM, read SNM, and WNM support the direct NVSim baseline macro.",
            bridge_notes="All macro quantities come directly from the fresh baseline NVSim run.",
        )
    )
    rows.append(
        case_summary_row(
            case="High Vt",
            macro_values={
                **baseline_common,
                "macro_leakage_power_uw": baseline_common["macro_leakage_power_uw"] * high_vt_leakage_ratio,
            },
            macro_sources={
                "macro_area_mm2": "Estimated by NVSim",
                "macro_read_latency_ns": "Estimated by NVSim",
                "macro_write_latency_ns": "Estimated by NVSim",
                "macro_read_energy_pj": "Estimated by NVSim",
                "macro_write_energy_pj": "Estimated by NVSim",
                "macro_leakage_power_uw": "Derived proxy",
            },
            support_values={
                "support_hold_snm_mv": bitcell_metrics["high_vt_hold_snm_mv"]["tt"],
                "support_read_snm_mv": bitcell_metrics["high_vt_read_snm_mv"]["tt"],
                "support_hold_window_energy_aj": bitcell_metrics["high_vt_hold_energy_aj"]["tt"]["optimized_value"],
            },
            support_sources={
                "support_hold_snm_mv": "Measured in Cadence",
                "support_read_snm_mv": "Measured in Cadence",
                "support_hold_window_energy_aj": "Measured in Cadence",
            },
            headline_metric="Hold-window supply energy",
            headline_delta_percent=bitcell_metrics["high_vt_hold_energy_aj"]["tt"]["delta_percent"],
            headline_note="Leakage-oriented gain with a read-SNM penalty at TT nominal.",
            bridge_notes="Only leakage is changed at macro level; timing, dynamic energy, and area intentionally remain baseline.",
        )
    )
    rows.append(
        case_summary_row(
            case="Negative BL",
            macro_values={
                **baseline_common,
                "macro_write_latency_ns": baseline_common["macro_write_latency_ns"] * baseline_write_ratio,
            },
            macro_sources={
                "macro_area_mm2": "Estimated by NVSim",
                "macro_read_latency_ns": "Estimated by NVSim",
                "macro_write_latency_ns": "Derived proxy",
                "macro_read_energy_pj": "Estimated by NVSim",
                "macro_write_energy_pj": "Estimated by NVSim",
                "macro_leakage_power_uw": "Estimated by NVSim",
            },
            support_values={
                "support_write_delay_ps": bitcell_metrics["negative_bl_write_delay_ps"]["tt"]["optimized_value"],
            },
            support_sources={
                "support_write_delay_ps": "Measured in Cadence",
            },
            headline_metric="Write delay",
            headline_delta_percent=bitcell_metrics["negative_bl_write_delay_ps"]["tt"]["delta_percent"],
            headline_note="The bridge changes only macro write latency using the TT Cadence delay ratio.",
            bridge_notes="Negative BL has no direct NVSim assist knob here, so read path, energy, leakage, and area remain baseline.",
        )
    )
    rows.append(
        case_summary_row(
            case="WL Underdrive",
            macro_values=baseline_common,
            macro_sources={
                "macro_area_mm2": "Estimated by NVSim",
                "macro_read_latency_ns": "Estimated by NVSim",
                "macro_write_latency_ns": "Estimated by NVSim",
                "macro_read_energy_pj": "Estimated by NVSim",
                "macro_write_energy_pj": "Estimated by NVSim",
                "macro_leakage_power_uw": "Estimated by NVSim",
            },
            support_values={
                "support_read_snm_mv": bitcell_metrics["wl_underdrive_read_snm_mv"]["tt"],
                "support_read_disturb_mv": bitcell_metrics["wl_underdrive_read_disturb_mv"]["tt"]["optimized_value"],
            },
            support_sources={
                "support_read_snm_mv": "Measured in Cadence",
                "support_read_disturb_mv": "Measured in Cadence",
            },
            headline_metric="Read SNM",
            headline_delta_percent=percent_delta_value(
                bitcell_metrics["baseline_read_snm_mv"]["tt"],
                bitcell_metrics["wl_underdrive_read_snm_mv"]["tt"],
            ),
            headline_note="Read-SNM improvement is attached as support evidence, so this case stays in the bitcell section rather than becoming an array-result comparison.",
            bridge_notes="WL underdrive is intentionally not forced into a new NVSim timing or energy case in this report.",
        )
    )
    return rows


def case_summary_row(
    *,
    case: str,
    macro_values: dict[str, float],
    macro_sources: dict[str, str],
    support_values: dict[str, float],
    support_sources: dict[str, str],
    headline_metric: str,
    headline_delta_percent: float | str,
    headline_note: str,
    bridge_notes: str,
) -> dict[str, str]:
    row = {
        "case": case,
        "headline_metric": headline_metric,
        "headline_delta_percent": format_optional(headline_delta_percent),
        "headline_note": headline_note,
        "bridge_notes": bridge_notes,
    }
    for name, value in macro_values.items():
        row[name] = format_float(value)
        row[f"{name}_evidence_type"] = macro_sources[name]
    for name in (
        "support_hold_snm_mv",
        "support_read_snm_mv",
        "support_write_nm_mv",
        "support_hold_window_energy_aj",
        "support_write_delay_ps",
        "support_read_disturb_mv",
    ):
        if name in support_values:
            row[name] = format_float(support_values[name])
            row[f"{name}_evidence_type"] = support_sources[name]
        else:
            row[name] = "NA"
            row[f"{name}_evidence_type"] = "NA"
    return row


def percent_delta_value(baseline_value: float, optimized_value: float) -> float:
    if baseline_value == 0.0:
        return 0.0
    return 100.0 * (optimized_value - baseline_value) / baseline_value


def format_optional(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return format_float(value)


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_array_methodology_md(config: SramMacroConfig, baseline_metrics: NvsimMetrics) -> None:
    text = f"""# Array Methodology

This phase rebuilds the array-level analysis from scratch around one direct NVSim baseline and an explicit bridge back to the Cadence bitcell evidence.

## Direct NVSim Baseline

The only direct macro run in this refreshed flow is a `{config.capacity_bytes}`-byte SRAM with:

1. `{config.rows}` rows x `{config.columns}` columns.
2. `{config.word_width_bits}`-bit words.
3. `{config.column_select_count}:1` column selection through `ForceMuxSenseAmp = {config.mux_sense_amp}`.
4. `45 nm` technology and the `LSTP` roadmap.

NVSim reported:

1. Area: `{baseline_metrics.area_um2 / 1e6:.6f} mm^2`
2. Read latency: `{baseline_metrics.read_latency_ps / 1000.0:.6f} ns`
3. Write latency: `{baseline_metrics.write_latency_ps / 1000.0:.6f} ns`
4. Read energy: `{baseline_metrics.read_energy_pj:.6f} pJ`
5. Write energy: `{baseline_metrics.write_energy_pj:.6f} pJ`
6. Leakage: `{baseline_metrics.leakage_power_uw:.6f} uW`

## Why The Flow Uses One Direct Baseline

This report now keeps the direct NVSim output separate from any optimization-specific interpretation:

1. Baseline macro quantities come directly from NVSim.
2. High-Vt leakage is a derived proxy from Cadence hold-window energy, not a fresh macro topology run.
3. Negative BL write latency is a derived proxy from Cadence write-delay improvement, not a direct assist-waveform simulation in NVSim.
4. WL underdrive remains baseline at macro level and is supported only by Cadence read-stability evidence.
5. Because WL underdrive does not change a defended macro quantity in this flow, it stays in the bitcell-results story instead of getting a dedicated array-result figure.

## Output Files

1. `sram_analysis_plots/array/data/nvsim_baseline_macro.csv`
2. `sram_analysis_plots/array/data/nvsim_baseline_latency_breakdown.csv`
3. `sram_analysis_plots/array/data/array_bridge_detail.csv`
4. `sram_analysis_plots/array/data/array_bridge_case_summary.csv`

## Evidence Boundary

The archived material in `old_nvsim/` is intentionally treated as reference context only. The live report pipeline should cite only the fresh files listed above.
"""
    (DOCS_DIR / "array_methodology.md").write_text(text)


def write_bridge_methodology_md(config: SramMacroConfig, bridge_rows: list[dict[str, str]]) -> None:
    case_rules = [
        ["Baseline", "All macro columns direct from the fresh NVSim baseline. TT bitcell hold/read/WNM are attached as nominal support."],
        ["High Vt", "Only macro leakage changes, using the TT hold-window energy ratio as a derived proxy. Area, latency, and dynamic energy stay baseline."],
        ["Negative BL", "Only macro write latency changes, using the TT Cadence write-delay ratio as a derived proxy. Other macro columns stay baseline."],
        ["WL Underdrive", "No macro timing or energy change is forced. Baseline macro columns are reused while read SNM and read disturb remain Cadence support, so the result stays in the bitcell section for emphasis."],
    ]
    quantity_counts = {
        evidence_type: sum(1 for row in bridge_rows if row["evidence_type"] == evidence_type)
        for evidence_type in ("Measured in Cadence", "Estimated by NVSim", "Derived proxy")
    }
    text = "\n".join(
        [
            "# Bridge Methodology",
            "",
            "Every bridge row is labeled as exactly one of: `Measured in Cadence`, `Estimated by NVSim`, or `Derived proxy`.",
            "",
            "## Locked Organization",
            "",
            f"- Capacity: `{config.capacity_bytes}` bytes",
            f"- Physical organization: `{config.rows} x {config.columns}`",
            f"- Word width: `{config.word_width_bits}` bits",
            f"- Column selection: `{config.column_select_count}:1`",
            "",
            "## Case Rules",
            "",
            markdown_table(["Case", "Rule"], case_rules),
            "",
            "## Evidence-Type Inventory",
            "",
            markdown_table(
                ["Evidence type", "Row count"],
                [
                    [evidence_type, str(quantity_counts[evidence_type])]
                    for evidence_type in ("Measured in Cadence", "Estimated by NVSim", "Derived proxy")
                ],
            ),
            "",
            "## Important Caveats",
            "",
            "- TT nominal is the report-body bridge anchor. Corner robustness remains in the bitcell section.",
            "- The wordline-underdrive read-disturb metric is explicitly treated as a pulse-window transient metric.",
            "- The write-noise-margin value remains tied to the current project WNM convention used in the butterfly plots.",
        ]
    )
    (DOCS_DIR / "bridge_methodology.md").write_text(text + "\n")


def write_phase_summary_md(
    config: SramMacroConfig,
    baseline_metrics: NvsimMetrics,
    bridge_case_rows: list[dict[str, str]],
) -> None:
    baseline_row = next(row for row in bridge_case_rows if row["case"] == "Baseline")
    lines = [
        "# NVSim Phase Summary",
        "",
        f"The refreshed array phase now uses one direct `{config.capacity_kb} KB` NVSim baseline for the `{config.rows} x {config.columns}` macro and then bridges the optimizations with explicit source labels.",
        "",
        "## Baseline Macro",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["Area", f"{baseline_metrics.area_um2 / 1e6:.6f} mm^2"],
                ["Read latency", f"{baseline_metrics.read_latency_ps / 1000.0:.6f} ns"],
                ["Write latency", f"{baseline_metrics.write_latency_ps / 1000.0:.6f} ns"],
                ["Read energy", f"{baseline_metrics.read_energy_pj:.6f} pJ"],
                ["Write energy", f"{baseline_metrics.write_energy_pj:.6f} pJ"],
                ["Leakage", f"{baseline_metrics.leakage_power_uw:.6f} uW"],
            ],
        ),
        "",
            "## Optimization Bridge Headlines",
            "",
            markdown_table(
                ["Case", "Headline metric", "Delta", "Bridge note"],
                [
                    [row["case"], row["headline_metric"], row["headline_delta_percent"], row["headline_note"]]
                    for row in bridge_case_rows
                ],
            ),
            "",
            "Only the High-Vt leakage proxy and Negative-BL write-latency proxy are promoted to dedicated array-result figures. WL underdrive remains in the bitcell section because its defended benefit is still cell-level stability.",
            "",
            "## Files To Cite",
            "",
            "- `sram_analysis_plots/array/data/nvsim_baseline_macro.csv`",
        "- `sram_analysis_plots/array/data/nvsim_baseline_latency_breakdown.csv`",
        "- `sram_analysis_plots/array/data/array_bridge_detail.csv`",
        "- `sram_analysis_plots/array/data/array_bridge_case_summary.csv`",
    ]
    (DOCS_DIR / "nvsim_phase_summary.md").write_text("\n".join(lines) + "\n")


def write_array_extrapolation_md(
    config: SramMacroConfig,
    baseline_metrics: NvsimMetrics,
    bridge_case_rows: list[dict[str, str]],
    baseline_text: str,
) -> None:
    lines = [
        "# Array Extrapolation: 128 x 128 SRAM at 45 nm",
        "",
        "This note is the clean restart point for the array section of the report. It references only the fresh NVSim baseline and the new bridge tables.",
        "",
        "## Target Macro",
        "",
        f"- Capacity: `{config.capacity_bytes}` bytes (`{config.total_bits}` bits)",
        f"- Physical organization: `{config.rows} x {config.columns}`",
        f"- Word width: `{config.word_width_bits}` bits",
        f"- Column selection: `{config.column_select_count}:1`",
        f"- NVSim roadmap: `{config.device_roadmap}`",
        "",
        "## Fresh Baseline NVSim Result",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["Area", f"{baseline_metrics.area_um2 / 1e6:.6f} mm^2"],
                ["Read latency", f"{baseline_metrics.read_latency_ps / 1000.0:.6f} ns"],
                ["Write latency", f"{baseline_metrics.write_latency_ps / 1000.0:.6f} ns"],
                ["Read energy", f"{baseline_metrics.read_energy_pj:.6f} pJ"],
                ["Write energy", f"{baseline_metrics.write_energy_pj:.6f} pJ"],
                ["Leakage", f"{baseline_metrics.leakage_power_uw:.6f} uW"],
            ],
        ),
        "",
        "## How The Optimizations Enter The Array Story",
        "",
        markdown_table(
            ["Case", "Macro treatment", "Bitcell support carried into the report"],
            [
                [
                    row["case"],
                    row["bridge_notes"],
                    row["headline_note"],
                ]
                for row in bridge_case_rows
            ],
        ),
        "",
        "## Evidence Boundary",
        "",
        "- `Baseline` macro values are `Estimated by NVSim`.",
        "- `High Vt` leakage and `Negative BL` write-latency updates are `Derived proxy` rows only.",
        "- Stability, disturb, and delay support quantities remain `Measured in Cadence`.",
        "- WL underdrive is intentionally kept as a bitcell-level result in the report narrative because no defended NVSim macro quantity changes.",
        "",
        "## Primary Files",
        "",
        "- `sram_analysis_plots/array/data/nvsim_baseline_macro.csv`",
        "- `sram_analysis_plots/array/data/nvsim_baseline_latency_breakdown.csv`",
        "- `sram_analysis_plots/array/data/array_bridge_detail.csv`",
        "- `sram_analysis_plots/array/data/array_bridge_case_summary.csv`",
        "- `sram_analysis_plots/report_figures/array/`",
        "",
        "## Raw NVSim Output Snippet",
        "",
        "```text",
        "\n".join(baseline_text.strip().splitlines()[:40]),
        "```",
    ]
    TOP_LEVEL_SUMMARY_PATH.write_text("\n".join(lines) + "\n")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator_line, *body_lines])


if __name__ == "__main__":
    main()
