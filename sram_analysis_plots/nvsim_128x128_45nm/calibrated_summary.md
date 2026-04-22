# Cadence-Calibrated NVSim 128x128 45 nm SRAM Summary

Generated from `sram_analysis_plots/rw_metrics.csv` using TT as the calibration baseline.
All calibrated corners use the same `LSTP` NVSim device roadmap, then apply corner-specific VDD, Ion, and Ioff scaling.

| corner | base | vdd_V | ion_scale | ioff_scale | subarray | area_um2 | area_eff_% | read_lat_ps | write_lat_ps | bitline_lat_ps | rowdec_lat_ps | read_energy_pJ | write_energy_pJ | leakage_uW |
| ------ | ---- | ----- | --------- | ---------- | -------- | -------- | ---------- | ----------- | ------------ | -------------- | ------------- | -------------- | --------------- | ---------- |
| ff     | LSTP | 1.200 | 1.394     | 19.656     | 128x128  | 6285.082 | 77.070     | 654.194     | 654.194      | 162.147        | 223.526       | 2.119          | 1.615           | 11.301     |
| ss_08  | LSTP | 0.800 | 0.247     | 0.504      | 128x128  | 6285.082 | 77.070     | 2217.000    | 2217.000     | 757.491        | 643.754       | 0.953          | 0.725           | 0.193      |
| ss_1   | LSTP | 1.000 | 0.615     | 0.743      | 128x128  | 6285.082 | 77.070     | 1120.000    | 1120.000     | 319.865        | 356.752       | 1.480          | 1.126           | 0.356      |
| tt     | LSTP | 1.000 | 1.000     | 1.000      | 128x128  | 6285.082 | 77.070     | 762.106     | 762.106      | 212.989        | 248.026       | 1.480          | 1.126           | 0.479      |

Generated configs:
- `nvsim/2kb_128x128_45nm_cal_ff.cfg`
- `nvsim/2kb_128x128_45nm_cal_ss_08.cfg`
- `nvsim/2kb_128x128_45nm_cal_ss_1.cfg`
- `nvsim/2kb_128x128_45nm_cal_tt.cfg`

Raw outputs:
- `cal_ff.txt`
- `cal_ss_08.txt`
- `cal_ss_1.txt`
- `cal_tt.txt`

Calibration formulas:
- `IonScale = tt_write_delay_ps / corner_write_delay_ps`
- `IoffScale = corner_standby_leakage_nA / tt_standby_leakage_nA`
- `VddOverride = corner_vdd_V`
