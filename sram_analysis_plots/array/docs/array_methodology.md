# Array Methodology

This phase rebuilds the array-level analysis from scratch around one direct NVSim baseline and an explicit bridge back to the Cadence bitcell evidence.

## Direct NVSim Baseline

The only direct macro run in this refreshed flow is a `2048`-byte SRAM with:

1. `128` rows x `128` columns.
2. `8`-bit words.
3. `16:1` column selection through `ForceMuxSenseAmp = 16`.
4. `45 nm` technology and the `LSTP` roadmap.

NVSim reported:

1. Area: `0.006197 mm^2`
2. Read latency: `0.882077 ns`
3. Write latency: `0.882077 ns`
4. Read energy: `1.354000 pJ`
5. Write energy: `0.321000 pJ`
6. Leakage: `0.485401 uW`

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
