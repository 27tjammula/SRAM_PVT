# Bridge Methodology

Every bridge row is labeled as exactly one of: `Measured in Cadence`, `Estimated by NVSim`, or `Derived proxy`.

## Locked Organization

- Capacity: `2048` bytes
- Physical organization: `128 x 128`
- Word width: `8` bits
- Column selection: `16:1`

## Case Rules

| Case | Rule |
| --- | --- |
| Baseline | All macro columns direct from the fresh NVSim baseline. TT bitcell hold/read/WNM are attached as nominal support. |
| High Vt | Only macro leakage changes, using the TT hold-window energy ratio as a derived proxy. Area, latency, and dynamic energy stay baseline. |
| Negative BL | Only macro write latency changes, using the TT Cadence write-delay ratio as a derived proxy. Other macro columns stay baseline. |
| WL Underdrive | No macro timing or energy change is forced. Baseline macro columns are reused while read SNM and read disturb remain Cadence support, so the result stays in the bitcell section for emphasis. |

## Evidence-Type Inventory

| Evidence type | Row count |
| --- | --- |
| Measured in Cadence | 10 |
| Estimated by NVSim | 22 |
| Derived proxy | 2 |

## Important Caveats

- TT nominal is the report-body bridge anchor. Corner robustness remains in the bitcell section.
- The wordline-underdrive read-disturb metric is explicitly treated as a pulse-window transient metric.
- The write-noise-margin value remains tied to the current project WNM convention used in the butterfly plots.
