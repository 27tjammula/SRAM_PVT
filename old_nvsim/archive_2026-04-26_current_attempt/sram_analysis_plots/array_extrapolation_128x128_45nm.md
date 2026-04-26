# 128x128 45 nm SRAM Array Extrapolation

This document extrapolates the bitcell-level simulation results to a 2 KB SRAM
organized as:

```text
128 rows x 128 columns = 16,384 bits = 2 KB
```

The source bitcell metrics are from `rw_metrics.md`. Those values are left
unchanged; this document separates direct extrapolations from lower-bound or
best-estimate array-level quantities.

## Assumptions

- Process: 45 nm.
- Array organization: 128 rows by 128 columns.
- One access activates one full 128-bit row.
- The bitcell transient results are intrinsic/local bitcell results.
- Peripheral circuits are not included in the bitcell CSVs.
- Estimated bitline capacitance per column side: 15 fF to 50 fF.
- Estimated 128-cell wordline capacitance: 25 fF to 80 fF.
- Estimated read bitline swing: 50 mV to 150 mV.

The capacitance assumptions are first-order planning estimates. Post-layout
extraction or a testbench with explicit BL/WL capacitance is needed for final
array energy and delay.

## Bitcell Source Results

| corner | VDD (V) | write delay (ps) | read disturb (mV) | bitcell leakage (nA) | bitcell write energy (fJ) | bitcell read energy (fJ) |
| ------ | ------: | ---------------: | ----------------: | -------------------: | ------------------------: | -----------------------: |
| ff     | 1.2     | 15.182           | 189.362           | 0.442965             | 1.204                     | 0.188                    |
| ss_08  | 0.8     | 85.623           | 61.397            | 0.011356             | 0.764                     | 0.032                    |
| ss_1   | 1.0     | 34.436           | 100.405           | 0.016752             | 0.687                     | 0.063                    |
| tt     | 1.0     | 21.167           | 119.438           | 0.022536             | 1.055                     | 0.070                    |

## Directly Scalable Result: Standby Leakage

Standby leakage is the cleanest bitcell-to-array extrapolation:

```text
I_leak_array_cells = 16,384 * I_leak_bitcell
```

| corner | bitcell leakage (nA) | 128x128 cell leakage (uA) |
| ------ | -------------------: | ------------------------: |
| ff     | 0.442965             | 7.258                     |
| ss_08  | 0.011356             | 0.186                     |
| ss_1   | 0.016752             | 0.274                     |
| tt     | 0.022536             | 0.369                     |

These values include only the 16,384 storage bitcells. Add decoder, precharge,
write-driver, sense-amplifier, and control leakage separately.

## Bitcell Results That Remain Sufficient

### Hold and Read SNM

SNM is a bitcell stability metric, so it is not multiplied by array size. The
holdSNM and readSNM butterfly plots are the correct nominal-cell evidence for
stability.

For a full 128x128 array, the missing piece is statistical variation:

```text
P(array passes) = P(single cell passes)^16384
```

That requires Monte Carlo or mismatch-aware corners. The nominal bitcell SNM
plots are sufficient for deterministic corner comparison, but not sufficient
for yield prediction.

### Read Disturb

The measured read disturb is also a bitcell-local stability result. It is
useful as-is for comparing PVT corners. A real array may change the disturb
because the bitline is no longer ideal; BL capacitance, precharge strength, WL
slew, and half-selected cells can shift the value. Treat the CSV read-disturb
values as nominal bitcell-testbench values, not final post-layout array values.

## Row-Level Cell-Local Energy Lower Bounds

For one full-row access, 128 bitcells participate. Multiplying the bitcell
energy by 128 gives an intrinsic lower bound:

```text
E_write_cells_row = 128 * E_write_bitcell
E_read_cells_row  = 128 * E_read_bitcell
```

| corner | cell-local row write (pJ) | cell-local row read (pJ) |
| ------ | ------------------------: | -----------------------: |
| ff     | 0.154                     | 0.024                    |
| ss_08  | 0.098                     | 0.004                    |
| ss_1   | 0.088                     | 0.008                    |
| tt     | 0.135                     | 0.009                    |

These are lower bounds. In a 128-column SRAM, bitline, wordline, write-driver,
precharge, decoder, and sense-amplifier energy can dominate.

## 45 nm Array Energy Best Estimates

Using the assumptions above:

```text
E_WL        ~= C_WL * VDD^2
E_BL_write  ~= 128 * C_BL * VDD^2
E_BL_read   ~= 128 * C_BL * VDD * DeltaV_BL
```

For write energy, the bitline term assumes one full-swing bitline side per
written bit. A conservative two-sided differential write estimate can be up to
about 2x the listed bitline-write term.

### Estimated Parasitic Components

| corner | VDD (V) | WL energy (pJ) | write BL energy (pJ) | read BL energy (pJ) |
| ------ | ------: | -------------: | -------------------: | ------------------: |
| ff     | 1.2     | 0.036-0.115    | 2.765-9.216          | 0.115-1.152         |
| ss_08  | 0.8     | 0.016-0.051    | 1.229-4.096          | 0.077-0.768         |
| ss_1   | 1.0     | 0.025-0.080    | 1.920-6.400          | 0.096-0.960         |
| tt     | 1.0     | 0.025-0.080    | 1.920-6.400          | 0.096-0.960         |

### Estimated Full-Row Access Energy

This combines the cell-local row lower bound with estimated WL and BL energy:

```text
E_write_row_est ~= E_write_cells_row + E_WL + E_BL_write
E_read_row_est  ~= E_read_cells_row  + E_WL + E_BL_read
```

| corner | estimated row write energy (pJ) | estimated row read energy (pJ) |
| ------ | ------------------------------: | -----------------------------: |
| ff     | 2.955-9.485                     | 0.175-1.291                    |
| ss_08  | 1.343-4.245                     | 0.097-0.823                    |
| ss_1   | 2.033-6.568                     | 0.129-1.048                    |
| tt     | 2.080-6.615                     | 0.130-1.049                    |

These totals still exclude decoder internal switching, precharge circuitry,
write-driver internal power, sense-amplifier energy, clock/control buffers, and
short-circuit current. They are better array-level estimates than multiplying
the bitcell energy alone, but they are not a replacement for an array
testbench.

## NVSim 128x128 Array Model

NVSim was also run with dedicated 45 nm configs that force the intended physical
organization:

```text
Capacity      = 2 KB
Word width    = 128 bits
Bank          = 1 x 1
Mat           = 1 x 1
Subarray      = 128 rows x 128 columns
Senseamp mux  = 1
Output muxes  = 1
```

Generated files:

- `nvsim/2kb_128x128_45nm_hp.cfg`
- `nvsim/2kb_128x128_45nm_lstp.cfg`
- `nvsim/2kb_128x128_45nm_lop.cfg`
- `sram_analysis_plots/nvsim_128x128_45nm/summary.md`
- `sram_analysis_plots/nvsim_128x128_45nm/summary.csv`
- `scripts/run_nvsim_128x128.py`

The NVSim roadmaps are not exact replacements for the Cadence PVT corners. HP,
LSTP, and LOP are built-in technology-roadmap models, while the bitcell CSVs are
from the gpdk45 transistor-level testbench. Use NVSim primarily for array
parasitics and peripheral estimates; use the Cadence results for bitcell
stability, read disturb, and corner-specific cell leakage.

| NVSim roadmap | subarray | area (um^2) | area efficiency (%) | read latency (ps) | write latency (ps) | read energy (pJ) | write energy (pJ) | leakage power (uW) |
| ------------- | -------- | ----------: | ------------------: | ----------------: | -----------------: | ---------------: | ----------------: | -----------------: |
| HP            | 128x128  | 6318.533    | 76.662              | 251.740           | 251.740            | 1.492            | 1.131             | 3045.000           |
| LSTP          | 128x128  | 6285.082    | 77.070              | 762.106           | 762.106            | 1.480            | 1.126             | 0.479              |
| LOP           | 128x128  | 6285.082    | 77.070              | 473.169           | 473.169            | 0.738            | 0.558             | 40.081             |

NVSim also reports useful delay contributors:

| NVSim roadmap | row decoder latency (ps) | bitline latency (ps) |
| ------------- | -----------------------: | -------------------: |
| HP            | 89.118                   | 69.191               |
| LSTP          | 248.026                  | 212.989              |
| LOP           | 145.913                  | 158.666              |

These results support the earlier conclusion: full-array access delay is not the
20 ps to 85 ps intrinsic bitcell delay. Once row decoding, bitline development,
predecode, sense amp, and precharge behavior are included, a 128x128 45 nm SRAM
access lands in the hundreds of picoseconds.

The NVSim dynamic energy results are lower than the broad hand-estimated
write-energy range for some cases. That is plausible because the hand estimate
used deliberately conservative bitline capacitance bounds, while NVSim derives
wire and device capacitance from its internal 45 nm model and the default
`SRAM.cell`. The safest reportable statement is:

```text
Cadence bitcell row lower bound: 0.004 pJ to 0.154 pJ
NVSim 128x128 array estimate   : 0.558 pJ to 1.492 pJ per access
Conservative hand estimate     : 0.097 pJ to 9.49 pJ per access
```

For leakage, the LSTP result is close to the Cadence-derived TT/SS cell-leakage
order of magnitude, while the NVSim HP leakage is much larger. Do not replace
the measured Cadence leakage table with the HP value unless the report is
explicitly discussing NVSim's HP roadmap model.

## Cadence-Calibrated NVSim Corner Configs

The Cadence corner table can be used to generate corner-specific NVSim configs.
The implemented calibration keeps the same physical 128x128 organization and
uses a common NVSim LSTP 45 nm base model, then applies:

```text
VddOverride = Cadence corner VDD
IonScale    = TT write_delay_ps / corner write_delay_ps
IoffScale   = corner standby_leakage_nA / TT standby_leakage_nA
```

Using a common LSTP base keeps the corner sweep internally comparable and gives
leakage values near the direct Cadence bitcell-array leakage power. The
calibration is still a compact model: it does not replace extracted wire RC,
device mismatch, or a transistor-level full-array simulation.

Generated files:

- `nvsim/2kb_128x128_45nm_cal_ff.cfg`
- `nvsim/2kb_128x128_45nm_cal_ss_08.cfg`
- `nvsim/2kb_128x128_45nm_cal_ss_1.cfg`
- `nvsim/2kb_128x128_45nm_cal_tt.cfg`
- `sram_analysis_plots/nvsim_128x128_45nm/calibrated_summary.md`
- `sram_analysis_plots/nvsim_128x128_45nm/calibrated_summary.csv`

| corner | base | VDD (V) | Ion scale | Ioff scale | read latency (ps) | write latency (ps) | read energy (pJ) | write energy (pJ) | leakage power (uW) |
| ------ | ---- | ------: | --------: | ---------: | ----------------: | -----------------: | ---------------: | ----------------: | -----------------: |
| ff     | LSTP | 1.2     | 1.394     | 19.656     | 654.194           | 654.194            | 2.119            | 1.615             | 11.301             |
| ss_08  | LSTP | 0.8     | 0.247     | 0.504      | 2217.000          | 2217.000           | 0.953            | 0.725             | 0.193              |
| ss_1   | LSTP | 1.0     | 0.615     | 0.743      | 1120.000          | 1120.000           | 1.480            | 1.126             | 0.356              |
| tt     | LSTP | 1.0     | 1.000     | 1.000      | 762.106           | 762.106            | 1.480            | 1.126             | 0.479              |

This gives a more useful array-level PVT estimate than the built-in NVSim
roadmaps alone:

- FF has the highest VDD, fastest calibrated device current, highest dynamic
  energy, and largest leakage.
- SS at 0.8 V is the slowest corner and has the lowest dynamic energy.
- SS at 1.0 V is slower than TT but has similar switched-capacitance energy in
  NVSim because the calibrated model changes current/leakage, not the physical
  capacitance.
- Read disturb, SNM, and bitcell write margin should still come from the
  Cadence bitcell results.

## Generated Report Figures

Array-level report figures were generated from the CSV summaries by:

```text
UV_CACHE_DIR=.cache/uv MPLCONFIGDIR=.cache/matplotlib uv run --with matplotlib python scripts/plot_array_figures.py
```

Generated files:

- `report_figures/calibrated_latency_by_corner.png`
- `report_figures/calibrated_energy_by_corner.png`
- `report_figures/calibrated_leakage_by_corner_log.png`
- `report_figures/latency_contributors_by_corner.png`
- `report_figures/array_vs_bitcell_energy_leakage.png`
- `report_figures/roadmap_vs_calibrated_summary.png`
- `report_figures/figure_index.md`

## Delay Extrapolation

Write delay from the bitcell CSV is the time from the first rising 50 percent WL
crossing to the first rising 50 percent Q crossing. That is an intrinsic
cell-flip delay under the testbench drive conditions.

It should not be multiplied by 128 or 16,384. A full-array delay estimate is:

```text
t_write_array ~= decoder delay
               + WL driver/wire delay to worst cell
               + BL/write-driver settling
               + bitcell write delay

t_read_array  ~= decoder delay
               + WL driver/wire delay to worst cell
               + BL develop time
               + sense-amplifier delay
```

The bitcell write delays are valid lower bounds:

| corner | bitcell write delay lower bound (ps) |
| ------ | -----------------------------------: |
| ff     | 15.182                               |
| ss_08  | 85.623                               |
| ss_1   | 34.436                               |
| tt     | 21.167                               |

For a 45 nm 128x128 SRAM, a practical pre-layout estimate is that full-array
write and read access delays will be in the hundreds of picoseconds once WL/BL
RC, decoder, write-driver, and sense-amplifier delays are included. The exact
number depends strongly on driver sizing and extracted capacitance.

## Read Delay Proxy

The bitcell transient analysis used this proxy:

```text
time from second WL 50 percent rise to |BL - BLB| = 50 mV
```

It returned `N/A` for all corners because the exported BL and BLB traces did not
separate by 50 mV during the second WL pulse. This means the current bitcell
testbench is not sufficient for array read-delay estimation. A realistic read
delay should be measured with:

- explicit BL/BLB capacitance for 128 rows,
- precharge/equalization circuitry,
- sense amplifier or read threshold,
- worst-case selected column and wordline location.

## Summary

- Leakage scales directly to about 0.186 uA to 7.258 uA of cell leakage across
  the listed corners.
- SNM remains a bitcell metric; use the butterfly plots directly for nominal
  stability, and use Monte Carlo for array yield.
- Read disturb is meaningful as a bitcell-corner comparison, but array
  parasitics can change it.
- Bitcell-only row energy is tiny, about 0.088 pJ to 0.154 pJ for write and
  0.004 pJ to 0.024 pJ for read.
- Estimated 128-bit row energy including first-order WL/BL parasitics is about
  1.34 pJ to 9.49 pJ for write and 0.097 pJ to 1.29 pJ for read.
- NVSim forced to the actual 128x128 array predicts about 0.558 pJ to 1.492 pJ
  per access and about 252 ps to 762 ps read/write latency across HP/LSTP/LOP
  roadmap models.
- Cadence-calibrated NVSim predicts about 0.725 pJ to 2.119 pJ per access,
  about 654 ps to 2.217 ns access latency, and about 0.193 uW to 11.301 uW
  leakage across the extracted corners.
- Full-array delay cannot be obtained by scaling the bitcell delay. The bitcell
  delay is a lower bound; final access delay needs array-level RC and peripheral
  circuitry.
