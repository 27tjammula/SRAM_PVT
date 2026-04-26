#!/usr/bin/env python3
"""Inspect timing landmarks (WL/BL/BLB/Q/QB) in trans_negBLopt_*.csv."""
from pathlib import Path
import csv
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "sim_data" / "optimized" / "negative_bitline"

FILES = [
    ("trans_negBLopt_tt.csv",     "TT/1.0V/27C", 1.0),
    ("trans_negBLopt_ss_1V.csv",  "SS/1.0V/85C", 1.0),
    ("trans_negBLopt_ss_0.8V.csv","SS/0.8V/85C", 0.8),
    ("trans_negBLopt_ff.csv",     "FF/1.2V/85C", 1.2),
]

SIGS = ["/WL", "/BL", "/BLB", "/Q", "/QB"]


def load(path: Path):
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and r[0].strip()]
    arr = np.array([[float(v) for v in r] for r in rows])
    cols = {}
    for i, h in enumerate(header):
        h = h.strip()
        base = h.split(" (")[0]
        suf = h.rsplit(" ", 1)[-1]
        cols.setdefault(base, {})[suf] = i
    return cols, arr


def report(path: Path, label: str, vdd: float):
    cols, arr = load(path)
    t_ns = arr[:, cols["/Q"]["X"]] * 1e9
    print(f"\n=== {label}  ({path.name})  VDD={vdd}V ===")
    # Print min/max per signal to see polarity
    for s in SIGS:
        if s not in cols:
            continue
        y = arr[:, cols[s]["Y"]]
        print(f"  {s:5s}  min={y.min():+.3f}V  max={y.max():+.3f}V")

    # Sample values at coarse time grid 0..30 ns step 2 ns to see the waveform shape
    print("  t(ns) | " + "  ".join(f"{s:>7s}" for s in SIGS if s in cols))
    for tt in range(0, 31, 2):
        idx = int(np.argmin(np.abs(t_ns - tt)))
        row = [f"{arr[idx, cols[s]['Y']]:+.3f}" for s in SIGS if s in cols]
        print(f"  {t_ns[idx]:5.2f} | " + "  ".join(f"{v:>7s}" for v in row))


for name, label, vdd in FILES:
    report(SRC / name, label, vdd)
