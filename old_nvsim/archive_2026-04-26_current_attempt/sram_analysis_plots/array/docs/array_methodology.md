# Array Methodology

This phase estimates the full 2 KB SRAM macro with NVSim while keeping the Cadence Virtuoso bitcell simulations as the circuit-level evidence.

## Why We Did Not Run a Full 128 x 128 Transistor-Level Simulation

Running a full transistor-level simulation for a `16384`-bit SRAM macro would require simulating `16384` storage devices plus the row decoder, column selection, sense amplifiers, precharge network, and write drivers in one transient environment. That is much heavier than the bitcell testbenches already completed in Cadence, especially across multiple corners and optimization cases.

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

These results are documented in `sram_analysis_plots\report_figures\bitcell\figure_guide.md` and are kept separate from the NVSim macro outputs.

## What NVSim Contributes

The NVSim flow models the `2048`-byte macro as:

1. `128` rows by `128` columns.
2. `8`-bit words.
3. `1` bank.
4. One physical `128 x 128` subarray.
5. A `16:1` column selection implied by `A0` to `A3`.

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
