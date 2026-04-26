#!/usr/bin/env python3
"""Measure & plot negative-bitline write delay (3rd WL pulse, ~21 ns).

Style: matches sram_analysis_plots/baseline/rwtrans/* — white background,
matplotlib tab10 palette, light-grey grid, dpi=200.
"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC_OPT  = REPO / "sim_data" / "optimized" / "negative_bitline"
SRC_BASE = REPO / "sim_data" / "baseline"  / "rwtrans"
DST = REPO / "sram_analysis_plots" / "optimized" / "negative_bitline"
DST.mkdir(parents=True, exist_ok=True)

# tab10 palette to match generate_structured_sram_plots.py / baseline plots
C_Q   = "#1f77b4"
C_QB  = "#d62728"
C_WL  = "#2ca02c"
C_BL  = "#9467bd"
C_BLB = "#ff7f0e"
C_DELAY    = "#d62728"   # red, matches QB
C_BASE_W1  = "#1f77b4"   # blue (baseline)
C_OPT_W0   = "#ff7f0e"   # orange (negative-BL optimization)

WIN_W0_NEGBL = (20.0, 28.0)
WIN_W1       = (2.5,  9.0)
ZOOM_WINDOW  = (20.95, 21.15)

CORNERS = [
    ("TT / 1.0V / 27°C", 1.0, "trans_negBLopt_tt.csv",      "rw_tt.csv",   "tt_1V0"),
    ("SS / 1.0V / 85°C", 1.0, "trans_negBLopt_ss_1V.csv",   "rw_ss_1.csv", "ss_1V0"),
    ("SS / 0.8V / 85°C", 0.8, "trans_negBLopt_ss_0.8V.csv", "rw_ss_08.csv","ss_0V8"),
    ("FF / 1.2V / 85°C", 1.2, "trans_negBLopt_ff.csv",      "rw_ff.csv",   "ff_1V2"),
]


def load(path: Path):
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and r[0].strip()]
    arr = np.array([[float(v) for v in r] for r in rows])
    cols = {}
    for i, h in enumerate(header):
        h = h.strip(); base = h.split(" (")[0]; suf = h.rsplit(" ", 1)[-1]
        cols.setdefault(base, {})[suf] = i
    return cols, arr


def first_crossing(t, y, level, rising: bool):
    if rising:
        idx = np.where((y[:-1] < level) & (y[1:] >= level))[0]
    else:
        idx = np.where((y[:-1] > level) & (y[1:] <= level))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    y0, y1 = y[i], y[i + 1]
    t0, t1 = t[i], t[i + 1]
    if y1 == y0:
        return t0
    return t0 + (level - y0) * (t1 - t0) / (y1 - y0)


def load_traces(path: Path):
    cols, arr = load(path)
    return {
        "t":   arr[:, cols["/Q"]["X"]] * 1e9,
        "wl":  arr[:, cols["/WL"]["Y"]],
        "bl":  arr[:, cols["/BL"]["Y"]],
        "blb": arr[:, cols["/BLB"]["Y"]],
        "q":   arr[:, cols["/Q"]["Y"]],
        "qb":  arr[:, cols["/QB"]["Y"]],
    }


def measure(traces: dict, vdd: float, window, q_rising: bool):
    t = traces["t"]; m = (t >= window[0]) & (t <= window[1])
    half = vdd / 2.0
    t_wl = first_crossing(t[m], traces["wl"][m], half, rising=True)
    t_q  = first_crossing(t[m], traces["q"][m],  half, rising=q_rising)
    if t_wl is None or t_q is None:
        return None, t_wl, t_q
    return (t_q - t_wl) * 1000.0, t_wl, t_q


# ── measure ─────────────────────────────────────────────────────────────────
results = []
for label, vdd, opt_name, base_name, tag in CORNERS:
    opt  = load_traces(SRC_OPT  / opt_name)
    base = load_traces(SRC_BASE / base_name)

    d_base_w1, _, _           = measure(base, vdd, WIN_W1,       q_rising=True)
    d_opt_w0, t_wl_w0, t_q_w0 = measure(opt,  vdd, WIN_W0_NEGBL, q_rising=False)

    print(
        f"{label:18s}  VDD={vdd}V  "
        f"base_w1={d_base_w1:6.1f} ps  opt_w0(negBL)={d_opt_w0:6.1f} ps"
    )
    results.append({
        "label": label, "vdd": vdd, "tag": tag,
        "d_base_w1": d_base_w1, "d_opt_w0": d_opt_w0,
        "t": opt["t"], "wl": opt["wl"], "bl": opt["bl"],
        "q": opt["q"], "qb": opt["qb"],
        "t_wl": t_wl_w0, "t_q": t_q_w0, "delay_ps": d_opt_w0,
    })


# ── per-corner zoom plots (one figure each) ─────────────────────────────────
for r in results:
    fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
    t = r["t"]; m = (t >= ZOOM_WINDOW[0]) & (t <= ZOOM_WINDOW[1])
    ax.plot(t[m], r["wl"][m], color=C_WL, lw=2.0, label="WL")
    ax.plot(t[m], r["bl"][m], color=C_BL, lw=1.8, label="BL")
    ax.plot(t[m], r["q"][m],  color=C_Q,  lw=2.2, label="Q")

    half = r["vdd"] / 2.0
    ax.axhline(half, color="0.6", lw=0.8, ls=":")
    if r["t_wl"] is not None:
        ax.axvline(r["t_wl"], color=C_WL, lw=0.8, ls="--", alpha=0.7)
        ax.plot(r["t_wl"], half, "o", color=C_WL, ms=7)
    if r["t_q"] is not None:
        ax.axvline(r["t_q"], color=C_Q, lw=0.8, ls="--", alpha=0.7)
        ax.plot(r["t_q"], half, "o", color=C_Q, ms=7)

    if r["delay_ps"] is not None:
        y_arrow = r["vdd"] * 1.10  # placed above the VDD rail in the headroom band
        ax.annotate(
            "", xy=(r["t_q"], y_arrow), xytext=(r["t_wl"], y_arrow),
            arrowprops=dict(arrowstyle="<->", color=C_DELAY, lw=1.8),
        )
        ax.text(
            (r["t_wl"] + r["t_q"]) / 2, y_arrow + 0.04 * r["vdd"],
            f"{r['delay_ps']:.1f} ps", color=C_DELAY, fontsize=12,
            ha="center", va="bottom", fontweight="bold",
        )

    ax.set_xlim(*ZOOM_WINDOW)
    ax.set_ylim(-0.15 * r["vdd"], 1.30 * r["vdd"])
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"Negative-Bitline Write Delay - {r['label']}")
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.legend(loc="center right", fontsize=10, framealpha=0.95)

    out = DST / f"write_delay_zoom_{r['tag']}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out.relative_to(REPO)}")


# ── corner bar chart ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.8, 4.6), constrained_layout=True)
labels  = [r["label"].replace(" / ", "\n") for r in results]
base_w1 = [r["d_base_w1"] for r in results]
opt_w0  = [r["d_opt_w0"]  for r in results]
x = np.arange(len(labels)); w = 0.36

b1 = ax.bar(x - w/2, base_w1, w, color=C_BASE_W1,
            label="Baseline (BL=VDD)")
b2 = ax.bar(x + w/2, opt_w0,  w, color=C_OPT_W0,
            label="Negative-Bitline (BL=-50 mV)")

ymax = max(base_w1 + opt_w0) * 1.18
for bars, vals in [(b1, base_w1), (b2, opt_w0)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ymax * 0.015,
                f"{v:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Write Delay (ps)")
ax.set_title("Write Delay by Corner - Baseline vs Negative-Bitline")
ax.set_ylim(0, ymax)
ax.grid(True, axis="y", color="0.9", linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

out_bar = DST / "write_delay_bars.png"
fig.savefig(out_bar, dpi=200)
plt.close(fig)
print(f"saved {out_bar.relative_to(REPO)}")
