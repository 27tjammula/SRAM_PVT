# Bitcell To Array Assumptions

## Exact SRAM Organization

1. Capacity: `2048` bytes.
2. Total bits: `16384`.
3. Physical array: `128` rows x `128` columns.
4. Word width: `8` bits.
5. Banks: `1`.
6. Column selection from Cadence addressing: `A0` to `A3`, which implies a `16:1` selection across each `128`-column row.
7. Row selection from Cadence addressing: `A4` to `A10`, which maps to `128` rows.

## Technology Node

1. Technology node used for NVSim: `45` nm.
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

- **Baseline**: macro scaling = `none`. Cadence support = Cadence baseline TT bitcell metrics attached as nominal support data.
- **High Vt**: macro scaling = `IoffScale=0.861 from TT 6-9 ns hold-window energy ratio`. Cadence support = TT hold-window supply energy -13.9% and TT read SNM -10.2% versus baseline.
- **Negative BL**: macro scaling = `none`. Cadence support = TT assisted write delay -54.5% (21.2 ps to 9.6 ps) versus baseline.
- **WL Underdrive**: macro scaling = `none`. Cadence support = TT read SNM +39.9% and TT read disturb -57.3% versus baseline.
