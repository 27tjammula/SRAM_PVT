# SRAM Report Outline

## 1. Bitcell Methodology And Extraction Rules

- Reference `report_figures/bitcell/bitcell_summary.md` for the canonical metrics table.
- State explicitly that read/hold SNM come from limiting-eye square fits and WNM uses the current diagonal-corner convention.
- Describe the WL-underdrive read-disturb value as a pulse-window transient metric.

## 2. Baseline Bitcell Stability And Writability

- Main figure: `report_figures/bitcell/bitcell_results_summary.png`.
- Appendix support: baseline hold/read/WNM overlays.

## 3. Optimization-Specific Bitcell Results

- High Vt: `report_figures/bitcell/high_vt/high_vt_tradeoff_summary.png` plus appendix overlays.
- Negative BL: `report_figures/bitcell/negative_bitline/negative_bitline_write_delay_by_corner.png` and waveform grid appendix.
- WL underdrive: read-SNM and read-disturb comparison figures plus the all-corner overlay appendix.

## 4. Baseline Array-Level NVSim Model

- Main figures: `report_figures/array/baseline_nvsim_macro_summary.png` and `report_figures/array/baseline_nvsim_latency_breakdown.png`.
- Cite `array/data/nvsim_baseline_macro.csv` and `array/data/nvsim_baseline_latency_breakdown.csv` directly.

## 5. Bridge Methodology

- Explain the three evidence labels: `Estimated by NVSim`, `Derived proxy`, and `Measured in Cadence`.
- Point to `array/docs/bridge_methodology.md` and `array/data/array_bridge_detail.csv`.

## 6. Optimization-Aware Macro Discussion

- Main figures: `report_figures/array/high_vt_leakage_proxy.png` and `report_figures/array/negative_bl_write_latency_proxy.png`.
- Emphasize that High-Vt changes leakage only and Negative BL changes write latency only in the current bridge.
- Keep WL underdrive in the bitcell results section because its defended benefits are cell-level stability metrics, not a justified NVSim macro shift.

## 7. Limitations And Evidence Boundaries

- Note that the archived `old_nvsim/` material is not part of the live report evidence.
- Note that Negative BL and WL underdrive were not directly simulated as full-array NVSim waveform cases.
- Note that TT nominal is the bridge anchor while corner robustness remains in the bitcell section.
