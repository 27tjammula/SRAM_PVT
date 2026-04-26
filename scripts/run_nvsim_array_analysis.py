#!/usr/bin/env python3
"""Run the 2 KB SRAM NVSim array analysis and write report-ready outputs."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from generate_structured_sram_plots import load_waveform_csv
from plot_bitcell_report_figures import (
    integrate_supply_energy,
    load_snm_overlay_case,
    load_wnm_overlay_case,
    measure_negative_bitline_delay,
    measure_read_disturb,
    measure_write_delay,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DATA_ROOT = REPO_ROOT / "sim_data"
BITCELL_GUIDE_PATH = REPO_ROOT / "sram_analysis_plots" / "report_figures" / "bitcell" / "figure_guide.md"
ARRAY_ROOT = REPO_ROOT / "sram_analysis_plots" / "array"
DATA_DIR = ARRAY_ROOT / "data"
FIGURES_DIR = ARRAY_ROOT / "figures"
DOCS_DIR = ARRAY_ROOT / "docs"
LOGS_DIR = ARRAY_ROOT / "logs"
NVSIM_DIR = REPO_ROOT / "nvsim"

CASE_ORDER = ["Baseline", "High Vt", "Negative BL", "WL Underdrive"]
CASE_SLUG = {
    "Baseline": "baseline",
    "High Vt": "high_vt",
    "Negative BL": "negative_bl",
    "WL Underdrive": "wl_underdrive",
}
FOUR_CORNER_LABELS = {
    "ff": "FF 1.2 V",
    "tt": "TT 1.0 V",
    "ss_1V": "SS 1.0 V",
    "ss_0.8V": "SS 0.8 V",
}
THREE_CORNER_LABELS = {
    "ff": "FF 1.2 V",
    "tt": "TT 1.0 V",
    "ss_1V": "SS 1.0 V",
}


@dataclass(frozen=True)
class SramMacroConfig:
    capacity_bytes: int = 2048
    total_bits: int = 16384
    rows: int = 128
    columns: int = 128
    word_width_bits: int = 8
    banks: int = 1
    technology_node_nm: int = 45
    memory_type: str = "SRAM"
    device_roadmap: str = "LSTP"
    local_wire_type: str = "LocalAggressive"
    local_wire_repeater_type: str = "RepeatedNone"
    local_wire_use_low_swing: bool = False
    global_wire_type: str = "GlobalAggressive"
    global_wire_repeater_type: str = "RepeatedNone"
    global_wire_use_low_swing: bool = False
    routing: str = "H-tree"
    internal_sensing: bool = True
    memory_cell_input_file: str = "SRAM.cell"
    temperature_k: int = 350
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
    def column_select_count(self) -> int:
        return self.columns // self.word_width_bits

    @property
    def column_select_bits(self) -> int:
        return int(math.log2(self.column_select_count))

    @property
    def row_select_bits(self) -> int:
        return int(math.log2(self.rows))

    def validate(self) -> None:
        if self.capacity_bytes * 8 != self.total_bits:
            raise ValueError(
                f"Invalid macro config: capacity_bytes*8={self.capacity_bytes * 8} but total_bits={self.total_bits}"
            )
        if self.rows * self.columns != self.total_bits:
            raise ValueError(
                f"Invalid macro config: rows*columns={self.rows * self.columns} but total_bits={self.total_bits}"
            )
        if self.word_width_bits != 8:
            raise ValueError(f"Invalid macro config: word_width_bits must be 8, got {self.word_width_bits}")
        if self.banks != 1:
            raise ValueError(f"Invalid macro config: banks must be 1, got {self.banks}")
        if self.columns % self.word_width_bits != 0:
            raise ValueError(
                f"Invalid macro config: columns={self.columns} is not divisible by word_width_bits={self.word_width_bits}"
            )
        if self.column_select_count != 16:
            raise ValueError(
                f"Invalid macro config: expected 16 column selections from A0-A3, got {self.column_select_count}"
            )
        if self.row_select_bits != 7:
            raise ValueError(f"Invalid macro config: expected 7 row bits from A4-A10, got {self.row_select_bits}")


@dataclass(frozen=True)
class NvsimMetrics:
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
class CaseRun:
    case: str
    applied_macro_scaling: str
    cadence_supported_trend: str
    notes: str
    config_text: str
    raw_output_path: Path
    config_path: Path
    metrics: NvsimMetrics


def main() -> None:
    config = SramMacroConfig()
    config.validate()
    ensure_output_structure()

    if not BITCELL_GUIDE_PATH.exists():
        raise FileNotFoundError(f"Required bitcell figure guide not found: {BITCELL_GUIDE_PATH}")

    nvsim_executable = ensure_nvsim_executable(LOGS_DIR)
    bitcell_data = extract_bitcell_metrics()
    case_runs = run_all_cases(config, nvsim_executable, bitcell_data)

    write_baseline_macro_csv(config, case_runs[0])
    case_rows = build_case_comparison_rows(config, case_runs, bitcell_data)
    bitcell_rows = build_bitcell_summary_rows(bitcell_data)
    combined_rows = build_combined_summary_rows(case_rows, bitcell_data)

    write_csv(DATA_DIR / "nvsim_case_comparison.csv", case_rows)
    write_json(DATA_DIR / "nvsim_case_comparison.json", {"cases": case_rows})
    write_csv(DATA_DIR / "bitcell_metrics_summary.csv", bitcell_rows)
    write_json(DATA_DIR / "bitcell_metrics_summary.json", {"metrics": bitcell_rows})
    write_csv(DATA_DIR / "array_analysis_combined_summary.csv", combined_rows)

    write_array_methodology_md(config, case_runs)
    write_bitcell_to_array_assumptions_md(config, case_rows)
    write_phase_summary_md(config, case_runs, case_rows)

    validate_csv_outputs(
        [
            DATA_DIR / "nvsim_baseline_macro.csv",
            DATA_DIR / "nvsim_case_comparison.csv",
            DATA_DIR / "bitcell_metrics_summary.csv",
            DATA_DIR / "array_analysis_combined_summary.csv",
        ]
    )

    print_terminal_summary(config, case_runs, case_rows)


def ensure_output_structure() -> None:
    for path in (ARRAY_ROOT, DATA_DIR, FIGURES_DIR, DOCS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    (LOGS_DIR / "configs").mkdir(parents=True, exist_ok=True)


def ensure_nvsim_executable(log_dir: Path) -> Path:
    log_path = log_dir / "nvsim_build.log"
    candidates = [
        NVSIM_DIR / "nvsim_local.exe",
        NVSIM_DIR / "nvsim.exe",
    ]
    for candidate in candidates:
        if candidate.exists() and is_runnable_nvsim(candidate):
            log_path.write_text(f"Reused existing NVSim executable: {candidate}\n")
            return candidate

    compiler = shutil.which("g++")
    if compiler is None:
        raise RuntimeError("Could not find g++ to build a local Windows NVSim executable")

    sources = sorted(path.name for path in NVSIM_DIR.glob("*.cpp"))
    command = [compiler, "-O2", "-std=c++11", "-Wall", *sources, "-o", "nvsim_local.exe"]
    completed = subprocess.run(command, cwd=NVSIM_DIR, text=True, capture_output=True)
    build_log = [
        "Command:",
        " ".join(command),
        "",
        "stdout:",
        completed.stdout,
        "",
        "stderr:",
        completed.stderr,
        "",
        f"return_code: {completed.returncode}",
    ]
    log_path.write_text("\n".join(build_log))
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to build NVSim locally. See {log_path}")

    executable = NVSIM_DIR / "nvsim_local.exe"
    if not is_runnable_nvsim(executable):
        raise RuntimeError(f"Built NVSim executable is not runnable: {executable}")
    return executable


def is_runnable_nvsim(executable: Path) -> bool:
    test_config = NVSIM_DIR / "test.cfg"
    try:
        completed = subprocess.run(
            [str(executable), str(test_config)],
            cwd=NVSIM_DIR,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0 and "Subarray Size" in completed.stdout


def extract_bitcell_metrics() -> dict[str, Any]:
    baseline_hold = {
        "ff": load_snm_overlay_case(SIM_DATA_ROOT / "baseline" / "hold" / "holdSNM_ff.csv", "ff", "FF 1.2 V").margin_mv,
        "tt": load_snm_overlay_case(SIM_DATA_ROOT / "baseline" / "hold" / "holdSNM_tt.csv", "tt", "TT 1.0 V").margin_mv,
        "ss_1V": load_snm_overlay_case(SIM_DATA_ROOT / "baseline" / "hold" / "holdSNM_ss.csv", "ss", "SS 1.0 V").margin_mv,
    }
    baseline_read = {
        "ff": load_snm_overlay_case(SIM_DATA_ROOT / "baseline" / "read" / "ReadSNM_ff.csv", "ff", "FF 1.2 V").margin_mv,
        "tt": load_snm_overlay_case(SIM_DATA_ROOT / "baseline" / "read" / "ReadSNM_tt.csv", "tt", "TT 1.0 V").margin_mv,
        "ss_1V": load_snm_overlay_case(SIM_DATA_ROOT / "baseline" / "read" / "ReadSNM_ss.csv", "ss", "SS 1.0 V").margin_mv,
    }
    baseline_write_nm = {
        "ff": load_wnm_overlay_case(SIM_DATA_ROOT / "baseline" / "write" / "writenm_ff.csv", "ff", "FF 1.2 V").margin_mv,
        "tt": load_wnm_overlay_case(SIM_DATA_ROOT / "baseline" / "write" / "writenm_tt.csv", "tt", "TT 1.0 V").margin_mv,
        "ss_1V": load_wnm_overlay_case(SIM_DATA_ROOT / "baseline" / "write" / "writenm_ss.csv", "ss", "SS 1.0 V").margin_mv,
    }

    high_vt_hold = {
        "ff": load_snm_overlay_case(
            SIM_DATA_ROOT / "optimized" / "high_vt" / "hold_snm_opt_vt_ff.csv", "ff", "FF 1.2 V"
        ).margin_mv,
        "tt": load_snm_overlay_case(
            SIM_DATA_ROOT / "optimized" / "high_vt" / "hold_snm_opt_vt_tt.csv", "tt", "TT 1.0 V"
        ).margin_mv,
        "ss_1V": load_snm_overlay_case(
            SIM_DATA_ROOT / "optimized" / "high_vt" / "hold_snm_opt_vt_ss.csv", "ss", "SS 1.0 V"
        ).margin_mv,
    }
    high_vt_read = {
        "ff": load_snm_overlay_case(
            SIM_DATA_ROOT / "optimized" / "high_vt" / "opt_Vt_readSNM_ff.csv", "ff", "FF 1.2 V"
        ).margin_mv,
        "tt": load_snm_overlay_case(
            SIM_DATA_ROOT / "optimized" / "high_vt" / "opt_Vt_readSNM_tt.csv", "tt", "TT 1.0 V"
        ).margin_mv,
        "ss_1V": load_snm_overlay_case(
            SIM_DATA_ROOT / "optimized" / "high_vt" / "opt_Vt_readSNM_ss.csv", "ss", "SS 1.0 V"
        ).margin_mv,
    }

    negative_bl_write_nm = {
        "tt": load_wnm_overlay_case(
            SIM_DATA_ROOT / "optimized" / "negative_bitline" / "negbl_writenm_tt.csv", "tt", "TT 1.0 V"
        ).margin_mv,
    }
    wordline_underdrive_read = {
        "ff": load_snm_overlay_case(
            SIM_DATA_ROOT / "optimized" / "wordline_underdrive" / "read_snm_opt_ff.csv", "ff", "FF 1.2 V"
        ).margin_mv,
        "tt": load_snm_overlay_case(
            SIM_DATA_ROOT / "optimized" / "wordline_underdrive" / "read_snm_opt_tt.csv", "tt", "TT 1.0 V"
        ).margin_mv,
        "ss_1V": load_snm_overlay_case(
            SIM_DATA_ROOT / "optimized" / "wordline_underdrive" / "read_snm_opt_ss.csv", "ss", "SS 1.0 V"
        ).margin_mv,
    }

    baseline_transient = extract_transient_case_metrics(
        SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_tt.csv",
        vdd=1.0,
        read_window_ns=(13.0, 16.0),
        q_rising=True,
    )
    high_vt_transient = extract_transient_case_metrics(
        SIM_DATA_ROOT / "optimized" / "high_vt" / "trans_vt_opt_tt.csv",
        vdd=1.0,
        read_window_ns=(13.0, 16.0),
        q_rising=True,
    )
    wl_underdrive_transient = extract_transient_case_metrics(
        SIM_DATA_ROOT / "optimized" / "wordline_underdrive" / "trans_WLunderopt_tt.csv",
        vdd=1.0,
        read_window_ns=(15.0, 18.0),
        q_rising=True,
    )
    negative_bl_delay_tt = measure_negative_bitline_delay(
        "TT 1.0 V",
        1.0,
        SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_tt.csv",
        SIM_DATA_ROOT / "optimized" / "negative_bitline" / "trans_negBLopt_tt.csv",
    )

    high_vt_hold_energy = extract_high_vt_hold_energy_trend()
    negative_bl_delay = extract_negative_bl_delay_trend()
    wl_underdrive_read_disturb = extract_wl_underdrive_read_disturb_trend()

    return {
        "baseline_hold_snm_mv": baseline_hold,
        "baseline_read_snm_mv": baseline_read,
        "baseline_write_nm_mv": baseline_write_nm,
        "high_vt_hold_snm_mv": high_vt_hold,
        "high_vt_read_snm_mv": high_vt_read,
        "negative_bl_write_nm_mv": negative_bl_write_nm,
        "wl_underdrive_read_snm_mv": wordline_underdrive_read,
        "baseline_tt_transient": baseline_transient,
        "high_vt_tt_transient": high_vt_transient,
        "wl_underdrive_tt_transient": wl_underdrive_transient,
        "negative_bl_tt_delay": {
            "baseline_delay_ps": negative_bl_delay_tt.baseline_delay_ps,
            "optimized_delay_ps": negative_bl_delay_tt.optimized_delay_ps,
            "delta_percent": percent_delta(
                negative_bl_delay_tt.baseline_delay_ps,
                negative_bl_delay_tt.optimized_delay_ps,
            ),
        },
        "high_vt_hold_energy_trend": high_vt_hold_energy,
        "negative_bl_delay_trend": negative_bl_delay,
        "wl_underdrive_read_disturb_trend": wl_underdrive_read_disturb,
        "high_vt_ioff_scale_tt": high_vt_hold_energy["tt"]["optimized_value"] / high_vt_hold_energy["tt"]["baseline_value"],
    }


def extract_transient_case_metrics(
    csv_path: Path,
    *,
    vdd: float,
    read_window_ns: tuple[float, float],
    q_rising: bool,
) -> dict[str, float]:
    signals = load_waveform_csv(csv_path)
    write_delay_ps, _, _ = measure_write_delay(
        signals["/Q"]["time_ns"],
        signals["/WL"]["value"],
        signals["/Q"]["value"],
        vdd,
        window_ns=(2.5, 9.0),
        q_rising=q_rising,
    )
    read_disturb_mv = measure_read_disturb(csv_path, pulse_window_ns=read_window_ns)
    return {
        "write_delay_ps": write_delay_ps,
        "read_disturb_mv": read_disturb_mv,
    }


def extract_high_vt_hold_energy_trend() -> dict[str, dict[str, float]]:
    cases = {
        "ff": (
            SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_ff.csv",
            SIM_DATA_ROOT / "optimized" / "high_vt" / "trans_vt_opt_ff.csv",
        ),
        "tt": (
            SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_tt.csv",
            SIM_DATA_ROOT / "optimized" / "high_vt" / "trans_vt_opt_tt.csv",
        ),
        "ss_1V": (
            SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_ss_1.csv",
            SIM_DATA_ROOT / "optimized" / "high_vt" / "trans_vt_opt_ss_1V.csv",
        ),
        "ss_0.8V": (
            SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_ss_08.csv",
            SIM_DATA_ROOT / "optimized" / "high_vt" / "trans_vt_opt_ss_0.8V.csv",
        ),
    }
    results: dict[str, dict[str, float]] = {}
    for corner, (baseline_csv, optimized_csv) in cases.items():
        baseline_value = integrate_supply_energy(baseline_csv, (6.0, 9.0))
        optimized_value = integrate_supply_energy(optimized_csv, (6.0, 9.0))
        results[corner] = {
            "baseline_value": baseline_value,
            "optimized_value": optimized_value,
            "delta_value": optimized_value - baseline_value,
            "delta_percent": percent_delta(baseline_value, optimized_value),
        }
    return results


def extract_negative_bl_delay_trend() -> dict[str, dict[str, float]]:
    cases = {
        "ff": (
            1.2,
            SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_ff.csv",
            SIM_DATA_ROOT / "optimized" / "negative_bitline" / "trans_negBLopt_ff.csv",
        ),
        "tt": (
            1.0,
            SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_tt.csv",
            SIM_DATA_ROOT / "optimized" / "negative_bitline" / "trans_negBLopt_tt.csv",
        ),
        "ss_1V": (
            1.0,
            SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_ss_1.csv",
            SIM_DATA_ROOT / "optimized" / "negative_bitline" / "trans_negBLopt_ss_1V.csv",
        ),
        "ss_0.8V": (
            0.8,
            SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_ss_08.csv",
            SIM_DATA_ROOT / "optimized" / "negative_bitline" / "trans_negBLopt_ss_0.8V.csv",
        ),
    }
    results: dict[str, dict[str, float]] = {}
    for corner, (vdd, baseline_csv, optimized_csv) in cases.items():
        measurement = measure_negative_bitline_delay(corner, vdd, baseline_csv, optimized_csv)
        results[corner] = {
            "baseline_value": measurement.baseline_delay_ps,
            "optimized_value": measurement.optimized_delay_ps,
            "delta_value": measurement.optimized_delay_ps - measurement.baseline_delay_ps,
            "delta_percent": percent_delta(measurement.baseline_delay_ps, measurement.optimized_delay_ps),
        }
    return results


def extract_wl_underdrive_read_disturb_trend() -> dict[str, dict[str, float]]:
    cases = {
        "ff": (
            SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_ff.csv",
            SIM_DATA_ROOT / "optimized" / "wordline_underdrive" / "trans_WLunderopt_ff.csv",
        ),
        "tt": (
            SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_tt.csv",
            SIM_DATA_ROOT / "optimized" / "wordline_underdrive" / "trans_WLunderopt_tt.csv",
        ),
        "ss_1V": (
            SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_ss_1.csv",
            SIM_DATA_ROOT / "optimized" / "wordline_underdrive" / "trans_WLunderopt_ss_1V.csv",
        ),
        "ss_0.8V": (
            SIM_DATA_ROOT / "baseline" / "rwtrans" / "rw_ss_08.csv",
            SIM_DATA_ROOT / "optimized" / "wordline_underdrive" / "trans_WLunderopt_ss_0.8V.csv",
        ),
    }
    results: dict[str, dict[str, float]] = {}
    for corner, (baseline_csv, optimized_csv) in cases.items():
        baseline_value = measure_read_disturb(baseline_csv, pulse_window_ns=(13.0, 16.0))
        optimized_value = measure_read_disturb(optimized_csv, pulse_window_ns=(15.0, 18.0))
        results[corner] = {
            "baseline_value": baseline_value,
            "optimized_value": optimized_value,
            "delta_value": optimized_value - baseline_value,
            "delta_percent": percent_delta(baseline_value, optimized_value),
        }
    return results


def run_all_cases(
    config: SramMacroConfig,
    nvsim_executable: Path,
    bitcell_data: dict[str, Any],
) -> list[CaseRun]:
    cases = [
        {
            "case": "Baseline",
            "applied_macro_scaling": "none",
            "cadence_supported_trend": "Cadence baseline TT bitcell metrics attached as nominal support data.",
            "notes": "Reference NVSim macro topology with 8-bit words and 16x column selection. No assist or device scaling applied.",
            "scales": {},
        },
        {
            "case": "High Vt",
            "applied_macro_scaling": f"IoffScale={bitcell_data['high_vt_ioff_scale_tt']:.3f} from TT 6-9 ns hold-window energy ratio",
            "cadence_supported_trend": (
                f"TT hold-window supply energy {bitcell_data['high_vt_hold_energy_trend']['tt']['delta_percent']:+.1f}% "
                f"and TT read SNM {percent_delta(bitcell_data['baseline_read_snm_mv']['tt'], bitcell_data['high_vt_read_snm_mv']['tt']):+.1f}% "
                "versus baseline."
            ),
            "notes": "NVSim supports leakage scaling through IoffScale, so only macro leakage is adjusted. Latency, energy, and area remain the same topology estimate.",
            "scales": {"ioff_scale": bitcell_data["high_vt_ioff_scale_tt"]},
        },
        {
            "case": "Negative BL",
            "applied_macro_scaling": "none",
            "cadence_supported_trend": (
                f"TT assisted write delay {bitcell_data['negative_bl_tt_delay']['delta_percent']:+.1f}% "
                f"({bitcell_data['negative_bl_tt_delay']['baseline_delay_ps']:.1f} ps to "
                f"{bitcell_data['negative_bl_tt_delay']['optimized_delay_ps']:.1f} ps) versus baseline."
            ),
            "notes": "This NVSim build does not model negative bitline assist directly, so the baseline macro is reused and the assist benefit is reported from Cadence bitcell transients.",
            "scales": {},
        },
        {
            "case": "WL Underdrive",
            "applied_macro_scaling": "none",
            "cadence_supported_trend": (
                f"TT read SNM {percent_delta(bitcell_data['baseline_read_snm_mv']['tt'], bitcell_data['wl_underdrive_read_snm_mv']['tt']):+.1f}% "
                f"and TT read disturb {bitcell_data['wl_underdrive_read_disturb_trend']['tt']['delta_percent']:+.1f}% versus baseline."
            ),
            "notes": "This NVSim build does not model reduced read wordline swing directly, so the baseline macro is reused and read-stability improvements are reported from Cadence.",
            "scales": {},
        },
    ]

    runs: list[CaseRun] = []
    for case in cases:
        config_text = build_nvsim_config_text(config, case["scales"])
        slug = CASE_SLUG[case["case"]]
        config_path = LOGS_DIR / "configs" / f"{slug}.cfg"
        raw_output_path = LOGS_DIR / f"{slug}_nvsim_output.txt"
        config_path.write_text(config_text)
        completed = subprocess.run(
            [str(nvsim_executable), str(config_path)],
            cwd=NVSIM_DIR,
            text=True,
            capture_output=True,
            timeout=120,
        )
        raw_output_path.write_text(completed.stdout + ("\n" + completed.stderr if completed.stderr else ""))
        if completed.returncode != 0:
            raise RuntimeError(f"NVSim failed for case {case['case']}. See {raw_output_path}")
        metrics = parse_nvsim_output(completed.stdout)
        validate_macro_metrics(metrics)
        if metrics.subarray_rows != config.rows or metrics.subarray_columns != config.columns:
            raise ValueError(
                f"NVSim case {case['case']} produced {metrics.subarray_rows}x{metrics.subarray_columns}, expected {config.rows}x{config.columns}"
            )
        runs.append(
            CaseRun(
                case=case["case"],
                applied_macro_scaling=case["applied_macro_scaling"],
                cadence_supported_trend=case["cadence_supported_trend"],
                notes=case["notes"],
                config_text=config_text,
                raw_output_path=raw_output_path,
                config_path=config_path,
                metrics=metrics,
            )
        )
    return runs


def build_nvsim_config_text(config: SramMacroConfig, scales: dict[str, float]) -> str:
    lines = [
        "// Generated by scripts/run_nvsim_array_analysis.py",
        f"// 2 KB {config.memory_type} macro: {config.rows} rows x {config.columns} columns, {config.word_width_bits}-bit words",
        "-DesignTarget: RAM",
        "-CacheAccessMode: Normal",
        "-OptimizationTarget: Area",
        "-EnablePruning: Yes",
        "",
        f"-ProcessNode: {config.technology_node_nm}",
        f"-Capacity (KB): {config.capacity_bytes // 1024}",
        f"-WordWidth (bit): {config.word_width_bits}",
        "",
        f"-DeviceRoadmap: {config.device_roadmap}",
    ]
    if "vdd_override" in scales:
        lines.append(f"-VddOverride (V): {scales['vdd_override']:.6g}")
    if "ion_scale" in scales:
        lines.append(f"-IonScale: {scales['ion_scale']:.6f}")
    if "ioff_scale" in scales:
        lines.append(f"-IoffScale: {scales['ioff_scale']:.6f}")
    lines.extend(
        [
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
            f"// Column selection: A0-A3 => {config.column_select_count}:1 mux for {config.word_width_bits}-bit words",
            f"// Row selection: A4-A10 => {config.rows} rows",
        ]
    )
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
        raise ValueError("Could not parse subarray size from NVSim output")
    return int(match.group(1)), int(match.group(2))


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


def convert_with_unit(text: str, pattern: str, units: dict[str, float], *, last: bool = False) -> float:
    matches = list(re.finditer(pattern, text))
    if not matches:
        raise ValueError(f"Could not parse pattern: {pattern}")
    match = matches[-1] if last else matches[0]
    value = float(match.group(1))
    unit = match.group(2)
    if unit not in units:
        raise ValueError(f"Unsupported unit {unit!r} for pattern {pattern}")
    return value * units[unit]


def validate_macro_metrics(metrics: NvsimMetrics) -> None:
    numeric_values = {
        "area_um2": metrics.area_um2,
        "read_latency_ps": metrics.read_latency_ps,
        "write_latency_ps": metrics.write_latency_ps,
        "read_energy_pj": metrics.read_energy_pj,
        "write_energy_pj": metrics.write_energy_pj,
        "leakage_power_uw": metrics.leakage_power_uw,
    }
    for name, value in numeric_values.items():
        if value < 0:
            raise ValueError(f"Invalid NVSim metric: {name} is negative ({value})")


def build_case_comparison_rows(
    config: SramMacroConfig,
    case_runs: list[CaseRun],
    bitcell_data: dict[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for run in case_runs:
        tt_hold = value_for_case_tt(run.case, bitcell_data, "hold")
        tt_read = value_for_case_tt(run.case, bitcell_data, "read")
        tt_wnm = value_for_case_tt(run.case, bitcell_data, "write_nm")
        tt_write_delay = value_for_case_tt(run.case, bitcell_data, "write_delay")
        tt_read_disturb = value_for_case_tt(run.case, bitcell_data, "read_disturb")
        rows.append(
            {
                "case": run.case,
                "capacity_bytes": str(config.capacity_bytes),
                "total_bits": str(config.total_bits),
                "rows": str(config.rows),
                "columns": str(config.columns),
                "word_width_bits": str(config.word_width_bits),
                "banks": str(config.banks),
                "technology_node_nm": str(config.technology_node_nm),
                "read_latency_ns": format_float(run.metrics.read_latency_ps / 1000.0),
                "write_latency_ns": format_float(run.metrics.write_latency_ps / 1000.0),
                "read_energy_pj": format_float(run.metrics.read_energy_pj),
                "write_energy_pj": format_float(run.metrics.write_energy_pj),
                "leakage_power_uw": format_float(run.metrics.leakage_power_uw),
                "area_um2": format_float(run.metrics.area_um2),
                "area_mm2": format_float(run.metrics.area_um2 / 1e6),
                "bitcell_hold_snm_mv_tt": format_optional(tt_hold),
                "bitcell_read_snm_mv_tt": format_optional(tt_read),
                "bitcell_write_nm_mv_tt": format_optional(tt_wnm),
                "write_delay_ps_tt": format_optional(tt_write_delay),
                "read_disturb_mv_tt": format_optional(tt_read_disturb),
                "applied_macro_scaling": run.applied_macro_scaling,
                "cadence_supported_trend": run.cadence_supported_trend,
                "notes": run.notes,
            }
        )
    return rows


def value_for_case_tt(case: str, bitcell_data: dict[str, Any], metric_family: str) -> float | None:
    if case == "Baseline":
        if metric_family == "hold":
            return bitcell_data["baseline_hold_snm_mv"]["tt"]
        if metric_family == "read":
            return bitcell_data["baseline_read_snm_mv"]["tt"]
        if metric_family == "write_nm":
            return bitcell_data["baseline_write_nm_mv"]["tt"]
        if metric_family == "write_delay":
            return bitcell_data["baseline_tt_transient"]["write_delay_ps"]
        if metric_family == "read_disturb":
            return bitcell_data["baseline_tt_transient"]["read_disturb_mv"]
    if case == "High Vt":
        if metric_family == "hold":
            return bitcell_data["high_vt_hold_snm_mv"]["tt"]
        if metric_family == "read":
            return bitcell_data["high_vt_read_snm_mv"]["tt"]
        if metric_family == "write_nm":
            return None
        if metric_family == "write_delay":
            return bitcell_data["high_vt_tt_transient"]["write_delay_ps"]
        if metric_family == "read_disturb":
            return bitcell_data["high_vt_tt_transient"]["read_disturb_mv"]
    if case == "Negative BL":
        if metric_family == "hold":
            return None
        if metric_family == "read":
            return None
        if metric_family == "write_nm":
            return bitcell_data["negative_bl_write_nm_mv"]["tt"]
        if metric_family == "write_delay":
            return bitcell_data["negative_bl_tt_delay"]["optimized_delay_ps"]
        if metric_family == "read_disturb":
            return bitcell_data["baseline_tt_transient"]["read_disturb_mv"]
    if case == "WL Underdrive":
        if metric_family == "hold":
            return None
        if metric_family == "read":
            return bitcell_data["wl_underdrive_read_snm_mv"]["tt"]
        if metric_family == "write_nm":
            return None
        if metric_family == "write_delay":
            return None
        if metric_family == "read_disturb":
            return bitcell_data["wl_underdrive_tt_transient"]["read_disturb_mv"]
    return None


def build_bitcell_summary_rows(bitcell_data: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(
        margin_rows(
            metric_group="Hold SNM",
            case="Baseline",
            values=bitcell_data["baseline_hold_snm_mv"],
            unit="mV",
            source_files=[
                "sim_data/baseline/hold/holdSNM_ff.csv",
                "sim_data/baseline/hold/holdSNM_tt.csv",
                "sim_data/baseline/hold/holdSNM_ss.csv",
            ],
            notes="Largest-square SNM extraction from the Cadence butterfly plots.",
        )
    )
    rows.extend(
        margin_rows(
            metric_group="Read SNM",
            case="Baseline",
            values=bitcell_data["baseline_read_snm_mv"],
            unit="mV",
            source_files=[
                "sim_data/baseline/read/ReadSNM_ff.csv",
                "sim_data/baseline/read/ReadSNM_tt.csv",
                "sim_data/baseline/read/ReadSNM_ss.csv",
            ],
            notes="Largest-square read SNM extraction from the Cadence butterfly plots.",
        )
    )
    rows.extend(
        margin_rows(
            metric_group="Write Noise Margin",
            case="Baseline",
            values=bitcell_data["baseline_write_nm_mv"],
            unit="mV",
            source_files=[
                "sim_data/baseline/write/writenm_ff.csv",
                "sim_data/baseline/write/writenm_tt.csv",
                "sim_data/baseline/write/writenm_ss.csv",
            ],
            notes="Diagonal-corner WNM extraction from the Cadence write butterfly plots.",
        )
    )
    rows.extend(
        comparative_rows(
            metric_group="Hold SNM",
            case="High Vt",
            baseline_values=bitcell_data["baseline_hold_snm_mv"],
            optimized_values=bitcell_data["high_vt_hold_snm_mv"],
            unit="mV",
            source_files=[
                "sim_data/optimized/high_vt/hold_snm_opt_vt_ff.csv",
                "sim_data/optimized/high_vt/hold_snm_opt_vt_tt.csv",
                "sim_data/optimized/high_vt/hold_snm_opt_vt_ss.csv",
            ],
            notes="High-Vt hold SNM compared against the baseline bitcell.",
        )
    )
    rows.extend(
        comparative_rows(
            metric_group="Read SNM",
            case="High Vt",
            baseline_values=bitcell_data["baseline_read_snm_mv"],
            optimized_values=bitcell_data["high_vt_read_snm_mv"],
            unit="mV",
            source_files=[
                "sim_data/optimized/high_vt/opt_Vt_readSNM_ff.csv",
                "sim_data/optimized/high_vt/opt_Vt_readSNM_tt.csv",
                "sim_data/optimized/high_vt/opt_Vt_readSNM_ss.csv",
            ],
            notes="High-Vt read SNM compared against the baseline bitcell.",
        )
    )
    rows.extend(
        comparative_rows(
            metric_group="Hold-Window Supply Energy",
            case="High Vt",
            baseline_values={corner: data["baseline_value"] for corner, data in bitcell_data["high_vt_hold_energy_trend"].items()},
            optimized_values={corner: data["optimized_value"] for corner, data in bitcell_data["high_vt_hold_energy_trend"].items()},
            unit="aJ",
            source_files=[
                "sim_data/baseline/rwtrans/rw_ff.csv",
                "sim_data/baseline/rwtrans/rw_tt.csv",
                "sim_data/baseline/rwtrans/rw_ss_1.csv",
                "sim_data/baseline/rwtrans/rw_ss_08.csv",
                "sim_data/optimized/high_vt/trans_vt_opt_ff.csv",
                "sim_data/optimized/high_vt/trans_vt_opt_tt.csv",
                "sim_data/optimized/high_vt/trans_vt_opt_ss_1V.csv",
                "sim_data/optimized/high_vt/trans_vt_opt_ss_0.8V.csv",
            ],
            notes="Integrated 6-9 ns supply energy used as the leakage-related High-Vt trend.",
        )
    )
    rows.extend(
        comparative_rows(
            metric_group="Write Delay",
            case="Negative BL",
            baseline_values={corner: data["baseline_value"] for corner, data in bitcell_data["negative_bl_delay_trend"].items()},
            optimized_values={corner: data["optimized_value"] for corner, data in bitcell_data["negative_bl_delay_trend"].items()},
            unit="ps",
            source_files=[
                "sim_data/baseline/rwtrans/rw_ff.csv",
                "sim_data/baseline/rwtrans/rw_tt.csv",
                "sim_data/baseline/rwtrans/rw_ss_1.csv",
                "sim_data/baseline/rwtrans/rw_ss_08.csv",
                "sim_data/optimized/negative_bitline/trans_negBLopt_ff.csv",
                "sim_data/optimized/negative_bitline/trans_negBLopt_tt.csv",
                "sim_data/optimized/negative_bitline/trans_negBLopt_ss_1V.csv",
                "sim_data/optimized/negative_bitline/trans_negBLopt_ss_0.8V.csv",
            ],
            notes="Assisted reverse-write WL-to-Q delay from the Cadence transient runs.",
        )
    )
    rows.extend(
        comparative_rows(
            metric_group="Read SNM",
            case="WL Underdrive",
            baseline_values=bitcell_data["baseline_read_snm_mv"],
            optimized_values=bitcell_data["wl_underdrive_read_snm_mv"],
            unit="mV",
            source_files=[
                "sim_data/optimized/wordline_underdrive/read_snm_opt_ff.csv",
                "sim_data/optimized/wordline_underdrive/read_snm_opt_tt.csv",
                "sim_data/optimized/wordline_underdrive/read_snm_opt_ss.csv",
            ],
            notes="Wordline-underdrive read SNM compared against the baseline bitcell.",
        )
    )
    rows.extend(
        comparative_rows(
            metric_group="Read Disturb",
            case="WL Underdrive",
            baseline_values={corner: data["baseline_value"] for corner, data in bitcell_data["wl_underdrive_read_disturb_trend"].items()},
            optimized_values={corner: data["optimized_value"] for corner, data in bitcell_data["wl_underdrive_read_disturb_trend"].items()},
            unit="mV",
            source_files=[
                "sim_data/optimized/wordline_underdrive/trans_WLunderopt_ff.csv",
                "sim_data/optimized/wordline_underdrive/trans_WLunderopt_tt.csv",
                "sim_data/optimized/wordline_underdrive/trans_WLunderopt_ss_1V.csv",
                "sim_data/optimized/wordline_underdrive/trans_WLunderopt_ss_0.8V.csv",
            ],
            notes="QB read-disturb magnitude measured during the read pulse.",
        )
    )
    rows.extend(
        margin_rows(
            metric_group="Write Noise Margin",
            case="Negative BL",
            values=bitcell_data["negative_bl_write_nm_mv"],
            unit="mV",
            source_files=["sim_data/optimized/negative_bitline/negbl_writenm_tt.csv"],
            notes="Static write-noise-margin result available only for the TT negative-bitline case.",
        )
    )
    return rows


def margin_rows(
    *,
    metric_group: str,
    case: str,
    values: dict[str, float],
    unit: str,
    source_files: list[str],
    notes: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for corner, value in values.items():
        rows.append(
            {
                "metric_group": metric_group,
                "case": case,
                "corner": corner,
                "corner_label": FOUR_CORNER_LABELS.get(corner, THREE_CORNER_LABELS.get(corner, corner)),
                "value": format_float(value),
                "unit": unit,
                "reference_case": "NA",
                "reference_value": "NA",
                "reference_unit": "NA",
                "delta_value": "NA",
                "delta_unit": "NA",
                "delta_percent": "NA",
                "source_files": "; ".join(source_files),
                "notes": notes,
            }
        )
    return rows


def comparative_rows(
    *,
    metric_group: str,
    case: str,
    baseline_values: dict[str, float],
    optimized_values: dict[str, float],
    unit: str,
    source_files: list[str],
    notes: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    shared_corners = sorted(set(baseline_values) & set(optimized_values))
    for corner in shared_corners:
        baseline_value = baseline_values[corner]
        optimized_value = optimized_values[corner]
        rows.append(
            {
                "metric_group": metric_group,
                "case": case,
                "corner": corner,
                "corner_label": FOUR_CORNER_LABELS.get(corner, THREE_CORNER_LABELS.get(corner, corner)),
                "value": format_float(optimized_value),
                "unit": unit,
                "reference_case": "Baseline",
                "reference_value": format_float(baseline_value),
                "reference_unit": unit,
                "delta_value": format_float(optimized_value - baseline_value),
                "delta_unit": unit,
                "delta_percent": format_float(percent_delta(baseline_value, optimized_value)),
                "source_files": "; ".join(source_files),
                "notes": notes,
            }
        )
    return rows


def build_combined_summary_rows(
    case_rows: list[dict[str, str]],
    bitcell_data: dict[str, Any],
) -> list[dict[str, str]]:
    summary_rows: list[dict[str, str]] = []
    for row in case_rows:
        primary_name = "NA"
        primary_value = "NA"
        primary_unit = "NA"
        primary_delta_percent = "NA"
        if row["case"] == "High Vt":
            primary_name = "Hold-window supply energy"
            primary_value = format_float(bitcell_data["high_vt_hold_energy_trend"]["tt"]["optimized_value"])
            primary_unit = "aJ"
            primary_delta_percent = format_float(bitcell_data["high_vt_hold_energy_trend"]["tt"]["delta_percent"])
        elif row["case"] == "Negative BL":
            primary_name = "Write delay"
            primary_value = row["write_delay_ps_tt"]
            primary_unit = "ps"
            primary_delta_percent = format_float(bitcell_data["negative_bl_delay_trend"]["tt"]["delta_percent"])
        elif row["case"] == "WL Underdrive":
            primary_name = "Read disturb"
            primary_value = row["read_disturb_mv_tt"]
            primary_unit = "mV"
            primary_delta_percent = format_float(bitcell_data["wl_underdrive_read_disturb_trend"]["tt"]["delta_percent"])
        summary_rows.append(
            {
                "case": row["case"],
                "macro_read_latency_ns": row["read_latency_ns"],
                "macro_write_latency_ns": row["write_latency_ns"],
                "macro_read_energy_pj": row["read_energy_pj"],
                "macro_write_energy_pj": row["write_energy_pj"],
                "macro_leakage_power_uw": row["leakage_power_uw"],
                "macro_area_mm2": row["area_mm2"],
                "tt_hold_snm_mv": row["bitcell_hold_snm_mv_tt"],
                "tt_read_snm_mv": row["bitcell_read_snm_mv_tt"],
                "tt_write_nm_mv": row["bitcell_write_nm_mv_tt"],
                "tt_write_delay_ps": row["write_delay_ps_tt"],
                "tt_read_disturb_mv": row["read_disturb_mv_tt"],
                "primary_trend_name": primary_name,
                "primary_trend_value": primary_value,
                "primary_trend_unit": primary_unit,
                "primary_trend_delta_percent": primary_delta_percent,
                "applied_macro_scaling": row["applied_macro_scaling"],
                "notes": row["notes"],
            }
        )
    return summary_rows


def write_baseline_macro_csv(config: SramMacroConfig, baseline_run: CaseRun) -> None:
    row = {
        "case": baseline_run.case,
        "capacity_bytes": str(config.capacity_bytes),
        "total_bits": str(config.total_bits),
        "rows": str(config.rows),
        "columns": str(config.columns),
        "word_width_bits": str(config.word_width_bits),
        "banks": str(config.banks),
        "technology_node_nm": str(config.technology_node_nm),
        "device_roadmap": config.device_roadmap,
        "column_mux_factor": str(config.column_select_count),
        "subarray_rows": str(baseline_run.metrics.subarray_rows),
        "subarray_columns": str(baseline_run.metrics.subarray_columns),
        "read_latency_ns": format_float(baseline_run.metrics.read_latency_ps / 1000.0),
        "write_latency_ns": format_float(baseline_run.metrics.write_latency_ps / 1000.0),
        "predecoder_latency_ns": format_float(baseline_run.metrics.predecoder_latency_ps / 1000.0),
        "subarray_read_latency_ns": format_float(baseline_run.metrics.subarray_read_latency_ps / 1000.0),
        "row_decoder_latency_ns": format_float(baseline_run.metrics.row_decoder_latency_ps / 1000.0),
        "bitline_latency_ns": format_float(baseline_run.metrics.bitline_latency_ps / 1000.0),
        "senseamp_latency_ns": format_float(baseline_run.metrics.senseamp_latency_ps / 1000.0),
        "precharge_latency_ns": format_float(baseline_run.metrics.precharge_latency_ps / 1000.0),
        "charge_latency_ns": format_float(baseline_run.metrics.charge_latency_ps / 1000.0),
        "read_energy_pj": format_float(baseline_run.metrics.read_energy_pj),
        "write_energy_pj": format_float(baseline_run.metrics.write_energy_pj),
        "row_decoder_read_energy_pj": format_float(baseline_run.metrics.row_decoder_read_energy_pj),
        "row_decoder_write_energy_pj": format_float(baseline_run.metrics.row_decoder_write_energy_pj),
        "precharge_energy_pj": format_float(baseline_run.metrics.precharge_energy_pj),
        "senseamp_energy_pj": format_float(baseline_run.metrics.senseamp_energy_pj),
        "leakage_power_uw": format_float(baseline_run.metrics.leakage_power_uw),
        "area_um2": format_float(baseline_run.metrics.area_um2),
        "area_mm2": format_float(baseline_run.metrics.area_um2 / 1e6),
        "area_efficiency_percent": format_float(baseline_run.metrics.area_efficiency_percent),
        "notes": baseline_run.notes,
    }
    write_csv(DATA_DIR / "nvsim_baseline_macro.csv", [row])


def write_array_methodology_md(config: SramMacroConfig, case_runs: list[CaseRun]) -> None:
    baseline = case_runs[0].metrics
    text = f"""# Array Methodology

This phase estimates the full 2 KB SRAM macro with NVSim while keeping the Cadence Virtuoso bitcell simulations as the circuit-level evidence.

## Why We Did Not Run a Full 128 x 128 Transistor-Level Simulation

Running a full transistor-level simulation for a `{config.total_bits}`-bit SRAM macro would require simulating `{config.total_bits}` storage devices plus the row decoder, column selection, sense amplifiers, precharge network, and write drivers in one transient environment. That is much heavier than the bitcell testbenches already completed in Cadence, especially across multiple corners and optimization cases.

## Why Cadence Plus NVSim Is Reasonable Here

Cadence and NVSim answer different but complementary questions:

1. Cadence bitcell simulations capture local stability and assist behavior such as hold SNM, read SNM, write noise margin, read disturb, and write-assist delay changes.
2. NVSim provides a macro-level estimate of area, latency, dynamic energy, and leakage once the SRAM is organized as a 2 KB array with explicit word width and muxing.

## What Cadence Contributes

The Cadence flow supplies:

1. Baseline hold SNM, read SNM, and write noise margin.
2. High-Vt hold SNM, read SNM, and hold-window supply-energy trends.
3. Negative-bitline write-delay improvement.
4. Wordline-underdrive read-SNM and read-disturb improvement.

These results are documented in `{BITCELL_GUIDE_PATH.relative_to(REPO_ROOT)}` and are kept separate from the NVSim macro outputs.

## What NVSim Contributes

The NVSim flow models the `{config.capacity_bytes}`-byte macro as:

1. `{config.rows}` rows by `{config.columns}` columns.
2. `{config.word_width_bits}`-bit words.
3. `{config.banks}` bank.
4. One physical `{baseline.subarray_rows} x {baseline.subarray_columns}` subarray.
5. A `{config.column_select_count}:1` column selection implied by `A0` to `A3`.

From that organization NVSim reports:

1. Total area.
2. Read latency.
3. Write latency.
4. Read energy.
5. Write energy.
6. Leakage power.
7. Internal latency and energy breakdowns such as row-decoder, bitline, precharge, and sense-amp terms.

## Assumptions

1. The macro topology is fixed across Baseline, High Vt, Negative BL, and WL Underdrive.
2. High Vt is represented in NVSim only through leakage scaling because that mapping is directly supported by the local NVSim build through `IoffScale`.
3. Negative BL and WL Underdrive are not directly modeled in NVSim because the implementation does not expose explicit assist-waveform controls for those cases.
4. When direct NVSim support is absent, the baseline macro estimate is reused and the Cadence trend is reported separately.

## Remaining Limitations

1. NVSim does not replace extracted post-layout parasitics for this specific design.
2. The assist techniques are still supported primarily by bitcell-level Cadence data, not by full macro transient waveforms.
3. Statistical variation and yield are outside the scope of this nominal-corner macro estimate.
"""
    (DOCS_DIR / "array_methodology.md").write_text(text)


def write_bitcell_to_array_assumptions_md(config: SramMacroConfig, case_rows: list[dict[str, str]]) -> None:
    text = f"""# Bitcell To Array Assumptions

## Exact SRAM Organization

1. Capacity: `{config.capacity_bytes}` bytes.
2. Total bits: `{config.total_bits}`.
3. Physical array: `{config.rows}` rows x `{config.columns}` columns.
4. Word width: `{config.word_width_bits}` bits.
5. Banks: `{config.banks}`.
6. Column selection from Cadence addressing: `A0` to `A3`, which implies a `{config.column_select_count}:1` selection across each `{config.columns}`-column row.
7. Row selection from Cadence addressing: `A4` to `A10`, which maps to `{config.rows}` rows.

## Technology Node

1. Technology node used for NVSim: `{config.technology_node_nm}` nm.
2. The repository evidence for this choice is the gpdk45 Cadence flow plus the existing 45 nm NVSim setup that was already archived in `old NVSIM/`.

## Voltage And Corner Assumptions

1. The macro topology estimate uses one nominal NVSim baseline roadmap configuration.
2. Cadence PVT behavior is attached as supporting evidence through the bitcell metrics rather than forcing every circuit metric into NVSim.
3. TT bitcell values are used as the representative nominal support values in `nvsim_case_comparison.csv`.

## How Each Optimization Is Represented

1. Baseline: direct NVSim macro estimate with the 2 KB organization above.
2. High Vt: same topology with NVSim leakage scaling only, using the TT Cadence high-Vt hold-window energy ratio as `IoffScale`.
3. Negative BL: baseline macro reused because the local NVSim model does not expose explicit negative-bitline assist controls.
4. WL Underdrive: baseline macro reused because the local NVSim model does not expose explicit read wordline underdrive controls.

## Directly Measured From Cadence

1. Hold SNM.
2. Read SNM.
3. Write noise margin.
4. Write delay trends.
5. Read disturb trends.
6. High-Vt hold-window supply-energy trend.

## Estimated By NVSim

1. Macro area.
2. Read latency.
3. Write latency.
4. Read energy.
5. Write energy.
6. Leakage power.
7. Reported row-decoder, bitline, precharge, and sense-amp subcomponent estimates.

## Qualitative Or Semi-Quantitative Trends

1. Negative BL is reported as a Cadence-backed assisted-write improvement that is not directly injected into the macro model.
2. WL Underdrive is reported as a Cadence-backed read-stability improvement that is not directly injected into the macro timing or energy model.
3. High Vt is only partially mapped into NVSim because leakage scaling is supported but the full transistor-level device tradeoff is still characterized by Cadence.

## Case Summary

"""
    for row in case_rows:
        text += (
            f"- **{row['case']}**: macro scaling = `{row['applied_macro_scaling']}`. "
            f"Cadence support = {row['cadence_supported_trend']}\n"
        )
    (DOCS_DIR / "bitcell_to_array_assumptions.md").write_text(text)


def write_phase_summary_md(
    config: SramMacroConfig,
    case_runs: list[CaseRun],
    case_rows: list[dict[str, str]],
) -> None:
    baseline = case_runs[0].metrics
    lines = [
        "# NVSim Phase Summary",
        "",
        "## Generated Files",
        "",
        "Data:",
        "- `data/nvsim_baseline_macro.csv`",
        "- `data/nvsim_case_comparison.csv`",
        "- `data/bitcell_metrics_summary.csv`",
        "- `data/array_analysis_combined_summary.csv`",
        "- `data/nvsim_case_comparison.json`",
        "- `data/bitcell_metrics_summary.json`",
        "",
        "Docs:",
        "- `docs/array_methodology.md`",
        "- `docs/bitcell_to_array_assumptions.md`",
        "- `docs/nvsim_phase_summary.md`",
        "",
        "Logs:",
        "- `logs/nvsim_build.log`",
        "- `logs/*_nvsim_output.txt`",
        "- `logs/configs/*.cfg`",
        "",
        "Figures are generated by `scripts/plot_nvsim_array_figures.py` into `figures/`.",
        "",
        "## Key Baseline NVSim Estimates",
        "",
        f"1. Area: `{baseline.area_um2:.3f} um^2` (`{baseline.area_um2 / 1e6:.6f} mm^2`).",
        f"2. Read latency: `{baseline.read_latency_ps / 1000.0:.3f} ns`.",
        f"3. Write latency: `{baseline.write_latency_ps / 1000.0:.3f} ns`.",
        f"4. Read energy: `{baseline.read_energy_pj:.3f} pJ`.",
        f"5. Write energy: `{baseline.write_energy_pj:.3f} pJ`.",
        f"6. Leakage power: `{baseline.leakage_power_uw:.6f} uW`.",
        "",
        "## Key Optimization Trends",
        "",
    ]
    for row in case_rows[1:]:
        lines.append(f"- **{row['case']}**: {row['cadence_supported_trend']} Macro scaling = `{row['applied_macro_scaling']}`.")
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "1. The NVSim topology estimate and the Cadence stability metrics are intentionally reported side-by-side instead of being forced into one unsupported model.",
            "2. Negative BL and WL Underdrive remain bitcell-backed assist trends, not full-array transient simulations.",
            "3. The High Vt macro case only applies leakage scaling because that is the clearest supported mapping in this local NVSim build.",
            "",
            "## Report-Ready Paragraph",
            "",
            (
                f"A 2 KB SRAM macro estimate was generated with the repository's NVSim flow using a `{config.rows} x {config.columns}` "
                f"physical array, `{config.word_width_bits}`-bit words, and `{config.column_select_count}:1` column selection to match the Cadence "
                f"addressing scheme. The baseline NVSim result predicts `{baseline.read_latency_ps / 1000.0:.3f} ns` read latency, "
                f"`{baseline.write_latency_ps / 1000.0:.3f} ns` write latency, `{baseline.read_energy_pj:.3f} pJ` read energy, "
                f"`{baseline.write_energy_pj:.3f} pJ` write energy, `{baseline.leakage_power_uw:.6f} uW` leakage, and `{baseline.area_um2 / 1e6:.6f} mm^2` area. "
                "Cadence bitcell simulations were kept as the source of stability and assist evidence: High Vt primarily contributes a leakage-related trend, "
                "Negative BL provides a strong assisted-write delay reduction, and WL Underdrive improves read stability by increasing read SNM and reducing read disturb. "
                "Where the local NVSim implementation did not directly support those assist techniques, the baseline macro estimate was retained and the Cadence measured trend was reported separately."
            ),
            "",
        ]
    )
    (DOCS_DIR / "nvsim_phase_summary.md").write_text("\n".join(lines))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def validate_csv_outputs(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"Expected nonempty CSV output: {path}")


def print_terminal_summary(
    config: SramMacroConfig,
    case_runs: list[CaseRun],
    case_rows: list[dict[str, str]],
) -> None:
    print("Completed NVSim 2 KB SRAM array analysis")
    print(f"Output root: {ARRAY_ROOT.relative_to(REPO_ROOT)}")
    print(
        f"Config: {config.capacity_bytes} B, {config.rows}x{config.columns}, "
        f"{config.word_width_bits}-bit word, {config.column_select_count}:1 column select"
    )
    print("Cases:")
    for run, row in zip(case_runs, case_rows):
        print(
            f"  {run.case}: read={float(row['read_latency_ns']):.3f} ns, "
            f"write={float(row['write_latency_ns']):.3f} ns, "
            f"leakage={float(row['leakage_power_uw']):.6f} uW, "
            f"area={float(row['area_mm2']):.6f} mm^2"
        )


def percent_delta(baseline_value: float, new_value: float) -> float:
    return 100.0 * (new_value - baseline_value) / baseline_value


def format_float(value: float) -> str:
    return f"{value:.6f}"


def format_optional(value: float | None) -> str:
    return "NA" if value is None else format_float(value)


if __name__ == "__main__":
    main()
