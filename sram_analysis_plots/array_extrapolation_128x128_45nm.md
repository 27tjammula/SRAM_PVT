# Array Extrapolation: 128 x 128 SRAM at 45 nm

This note is the clean restart point for the array section of the report. It references only the fresh NVSim baseline and the new bridge tables.

## Target Macro

- Capacity: `2048` bytes (`16384` bits)
- Physical organization: `128 x 128`
- Word width: `8` bits
- Column selection: `16:1`
- NVSim roadmap: `LSTP`

## Fresh Baseline NVSim Result

| Metric | Value |
| --- | --- |
| Area | 0.006197 mm^2 |
| Read latency | 0.882077 ns |
| Write latency | 0.882077 ns |
| Read energy | 1.354000 pJ |
| Write energy | 0.321000 pJ |
| Leakage | 0.485401 uW |

## How The Optimizations Enter The Array Story

| Case | Macro treatment | Bitcell support carried into the report |
| --- | --- | --- |
| Baseline | All macro quantities come directly from the fresh baseline NVSim run. | TT hold SNM, read SNM, and WNM support the direct NVSim baseline macro. |
| High Vt | Only leakage is changed at macro level; timing, dynamic energy, and area intentionally remain baseline. | Leakage-oriented gain with a read-SNM penalty at TT nominal. |
| Negative BL | Negative BL has no direct NVSim assist knob here, so read path, energy, leakage, and area remain baseline. | The bridge changes only macro write latency using the TT Cadence delay ratio. |
| WL Underdrive | WL underdrive is intentionally not forced into a new NVSim timing or energy case in this report. | Read-SNM improvement is attached as support evidence, so this case stays in the bitcell section rather than becoming an array-result comparison. |

## Evidence Boundary

- `Baseline` macro values are `Estimated by NVSim`.
- `High Vt` leakage and `Negative BL` write-latency updates are `Derived proxy` rows only.
- Stability, disturb, and delay support quantities remain `Measured in Cadence`.
- WL underdrive is intentionally kept as a bitcell-level result in the report narrative because no defended NVSim macro quantity changes.

## Primary Files

- `sram_analysis_plots/array/data/nvsim_baseline_macro.csv`
- `sram_analysis_plots/array/data/nvsim_baseline_latency_breakdown.csv`
- `sram_analysis_plots/array/data/array_bridge_detail.csv`
- `sram_analysis_plots/array/data/array_bridge_case_summary.csv`
- `sram_analysis_plots/report_figures/array/`

## Raw NVSim Output Snippet

```text
User-defined configuration file (/Users/krishnachemudupati/School/ESE5760/SRAM_PVT/sram_analysis_plots/array/logs/configs/baseline.cfg) is loaded

Memory Cell: SRAM
Cell Area (F^2)    : 146.000 (14.600Fx10.000F)
Cell Aspect Ratio  : 1.460
SRAM Cell Access Transistor Width: 1.310F
SRAM Cell NMOS Width: 2.080F
SRAM Cell PMOS Width: 1.230F

====================
DESIGN SPECIFICATION
====================
Design Target: Random Access Memory
Capacity   : 2KB
Data Width : 8Bits (1Bytes)

Searching for the best solution that is optimized for area ...

=============
CONFIGURATION
=============
Bank Organization: 1 x 1
 - Row Activation   : 1 / 1
 - Column Activation: 1 / 1
Mat Organization: 1 x 1
 - Row Activation   : 1 / 1
 - Column Activation: 1 / 1
 - Subarray Size    : 128 Rows x 128 Columns
Mux Level:
 - Senseamp Mux      : 16
 - Output Level-1 Mux: 1
 - Output Level-2 Mux: 1
Local Wire:
 - Wire Type : Local Aggressive
 - Repeater Type: No Repeaters
 - Low Swing : No
Global Wire:
 - Wire Type : Global Aggressive
 - Repeater Type: No Repeaters
 - Low Swing : No
```
