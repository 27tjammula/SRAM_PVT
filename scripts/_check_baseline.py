from pathlib import Path
import csv
import numpy as np
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "sim_data" / "baseline" / "rwtrans"

for name in ["rw_tt.csv", "rw_ss_1.csv", "rw_ss_08.csv", "rw_ff.csv"]:
    print(f"\n--- {name} ---")
    with open(SRC / name) as f:
        r = csv.reader(f); h = next(r)
        rows = [row for row in r if row and row[0].strip()]
    arr = np.array([[float(v) for v in row] for row in rows])
    cols = {}
    for i, hh in enumerate(h):
        hh = hh.strip(); base = hh.split(" (")[0]; suf = hh.rsplit(" ",1)[-1]
        cols.setdefault(base, {})[suf] = i
    t = arr[:, cols["/Q"]["X"]] * 1e9
    print(f"  rows={len(arr)} t={t.min():.2f}-{t.max():.2f} ns")
    for s in ["/WL","/BL","/BLB","/Q","/QB"]:
        y = arr[:, cols[s]["Y"]]
        print(f"  {s:5s} min={y.min():+.3f} max={y.max():+.3f}")
    print(f"  {'t(ns)':>7s} {'WL':>7s} {'BL':>7s} {'BLB':>7s} {'Q':>7s} {'QB':>7s}")
    for tt in range(0, int(t.max())+1, 2):
        idx = int(np.argmin(np.abs(t - tt)))
        print(f"  {t[idx]:7.2f} "
              f"{arr[idx, cols['/WL']['Y']]:7.3f} "
              f"{arr[idx, cols['/BL']['Y']]:7.3f} "
              f"{arr[idx, cols['/BLB']['Y']]:7.3f} "
              f"{arr[idx, cols['/Q']['Y']]:7.3f} "
              f"{arr[idx, cols['/QB']['Y']]:7.3f}")
