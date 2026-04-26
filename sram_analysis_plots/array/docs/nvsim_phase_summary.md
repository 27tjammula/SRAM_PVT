# NVSim Phase Summary

The refreshed array phase now uses one direct `2 KB` NVSim baseline for the `128 x 128` macro and then bridges the optimizations with explicit source labels.

## Baseline Macro

| Metric | Value |
| --- | --- |
| Area | 0.006197 mm^2 |
| Read latency | 0.882077 ns |
| Write latency | 0.882077 ns |
| Read energy | 1.354000 pJ |
| Write energy | 0.321000 pJ |
| Leakage | 0.485401 uW |

## Optimization Bridge Headlines

| Case | Headline metric | Delta | Bridge note |
| --- | --- | --- | --- |
| Baseline | Baseline nominal support | NA | TT hold SNM, read SNM, and WNM support the direct NVSim baseline macro. |
| High Vt | Hold-window supply energy | -13.902855 | Leakage-oriented gain with a read-SNM penalty at TT nominal. |
| Negative BL | Write delay | -54.465664 | The bridge changes only macro write latency using the TT Cadence delay ratio. |
| WL Underdrive | Read SNM | 39.937103 | Read-SNM improvement is attached as support evidence, so this case stays in the bitcell section rather than becoming an array-result comparison. |

Only the High-Vt leakage proxy and Negative-BL write-latency proxy are promoted to dedicated array-result figures. WL underdrive remains in the bitcell section because its defended benefit is still cell-level stability.

## Files To Cite

- `sram_analysis_plots/array/data/nvsim_baseline_macro.csv`
- `sram_analysis_plots/array/data/nvsim_baseline_latency_breakdown.csv`
- `sram_analysis_plots/array/data/array_bridge_detail.csv`
- `sram_analysis_plots/array/data/array_bridge_case_summary.csv`
