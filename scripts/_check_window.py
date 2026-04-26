from pathlib import Path
import csv
import numpy as np
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "sim_data" / "optimized" / "negative_bitline"

for name in ["trans_negBLopt_tt.csv", "trans_negBLopt_ss_0.8V.csv"]:
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
    wl = arr[:, cols["/WL"]["Y"]]
    bl = arr[:, cols["/BL"]["Y"]]
    q = arr[:, cols["/Q"]["Y"]]
    qb = arr[:, cols["/QB"]["Y"]]
    m = (t >= 20.0) & (t <= 22.5)
    print(f" {'t(ns)':>8s} {'WL':>8s} {'BL':>8s} {'Q':>8s} {'QB':>8s}")
    for i in np.where(m)[0]:
        print(f" {t[i]:8.4f} {wl[i]:8.3f} {bl[i]:8.3f} {q[i]:8.3f} {qb[i]:8.3f}")
