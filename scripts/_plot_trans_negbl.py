#!/usr/bin/env python3
"""Quick plot of trans_negBLopt_*.csv transient signals (stacked subplots)."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "sim_data" / "optimized" / "negative_bitline"
DST = REPO / "sram_analysis_plots" / "optimized" / "negative_bitline"
DST.mkdir(parents=True, exist_ok=True)

SIGNAL_COLORS = {
    "/Q":   "#1f77b4",
    "/QB":  "#d62728",
    "/WL":  "#2ca02c",
    "/BL":  "#9467bd",
    "/BLB": "#ff7f0e",
}
SIGNAL_ORDER = ("/Q", "/QB", "/WL", "/BL", "/BLB")

FILES = [
    ("trans_negBLopt_tt.csv",      "trans_negBLopt_tt"),
    ("trans_negBLopt_ss_1V.csv",   "trans_negBLopt_ss_1V"),
    ("trans_negBLopt_ss_0.8V.csv", "trans_negBLopt_ss_0.8V"),
    ("trans_negBLopt_ff.csv",      "trans_negBLopt_ff"),
]


def load(path: Path):
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and r[0].strip()]
    arr = np.array([[float(v) for v in r] for r in rows])
    return header, arr


def col_index(header, signal, axis):
    suf = f" {axis}"
    for i, h in enumerate(header):
        h = h.strip()
        if h.endswith(suf) and h.split(" (")[0] == signal:
            return i
    raise KeyError(f"{signal}{suf} not found")


for name, stem in FILES:
    src = SRC / name
    if not src.exists():
        print(f"missing {src}")
        continue
    header, arr = load(src)
    t_ns = arr[:, col_index(header, "/Q", "X")] * 1e9

    available = []
    for sig in SIGNAL_ORDER:
        try:
            yi = col_index(header, sig, "Y")
        except KeyError:
            continue
        available.append((sig, SIGNAL_COLORS[sig], yi))

    n = len(available)
    fig, axes = plt.subplots(
        n, 1, figsize=(8.8, 1.4 * n + 0.6), sharex=True,
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]
    for ax, (sig, color, yi) in zip(axes, available):
        ax.plot(t_ns, arr[:, yi], color=color, lw=1.8, label=sig[1:])
        ax.set_ylabel(f"{sig[1:]}\n(V)")
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    axes[0].set_title(f"{stem} - Voltages")
    axes[-1].set_xlabel("Time (ns)")
    axes[-1].set_xlim(t_ns.min(), t_ns.max())

    out = DST / (stem + ".png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out.relative_to(REPO)}  rows={len(arr)}  t={t_ns.min():.2f}–{t_ns.max():.2f} ns")
