import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams['font.family'] = 'DejaVu Sans'

# ── color palette ─────────────────────────────────────────────────────────────
DARK  = '#0D1117'; PANEL = '#161B22'; ACCENT = '#58A6FF'
GREEN = '#3FB950'; AMBER = '#F0A500'; MUTED  = '#8B949E'; WHITE = '#E6EDF3'

# ── helpers ───────────────────────────────────────────────────────────────────
def load_butterfly(path):
    df = pd.read_csv(path)
    return df.iloc[:,0].values, df.iloc[:,1].values, df.iloc[:,2].values, df.iloc[:,3].values

def estimate_snm(x1, y1, x2, y2):
    xg = np.linspace(max(min(x1), min(x2)), min(max(x1), max(x2)), 2000)
    f1 = interp1d(x1, y1, bounds_error=False, fill_value='extrapolate')
    f2 = interp1d(x2, y2, bounds_error=False, fill_value='extrapolate')
    diff = f1(xg) - f2(xg)
    cross = np.where(np.diff(np.sign(diff)))[0]
    if len(cross) >= 2:
        return abs(xg[cross[1]] - xg[cross[0]]) / (2 * np.sqrt(2)) * 1000
    return None

def load_trans(path):
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    get = lambda kw, suf: df[[c for c in cols if kw in c and c.endswith(suf)][0]].values
    t  = get('/QB', ' X') * 1e9
    qb = get('/QB', ' Y')
    wl = get('/WL', ' Y')
    return t, qb, wl

def qb_disturb_baseline(path):
    t, qb, wl = load_trans(path)
    vdd = max(wl)
    hi = (wl > vdd * 0.4).astype(int)
    rises = np.where(np.diff(hi) == 1)[0]
    falls = np.where(np.diff(hi) == -1)[0]
    peaks = []
    for i in range(1, min(len(rises), len(falls))):   # skip write pulse (i=0)
        t0, t1 = t[rises[i]], t[falls[i]]
        mask = (t >= t0) & (t <= t1)
        if mask.sum() > 0:
            peaks.append(max(qb[mask]) * 1000)
    return np.mean(peaks) if peaks else 0.0

def qb_disturb_opt(path, read_windows):
    t, qb, wl = load_trans(path)
    peaks = []
    for t0, t1 in read_windows:
        mask = (t >= t0) & (t <= t1)
        if mask.sum() > 0:
            peaks.append(max(qb[mask]) * 1000)
    return np.mean(peaks) if peaks else 0.0

def style_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color('#30363D')
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(WHITE)

# ── file paths — update these to match your directory ─────────────────────────
BASE_DIR = '/mnt/user-data/uploads/'

base_snm_files = {
    'tt': BASE_DIR + 'ReadSNM_tt.csv',
    'ss': BASE_DIR + 'ReadSNM_ss.csv',
    'ff': BASE_DIR + 'ReadSNM_ff.csv',
}
opt_snm_files = {
    'tt': BASE_DIR + 'read_snm_opt_tt.csv',
    'ss': BASE_DIR + 'read_snm_opt_ss.csv',
    'ff': BASE_DIR + 'read_snm_opt_ff.csv',
}
base_trans_files = {
    'tt':   BASE_DIR + 'rw_tt.csv',
    'ss1':  BASE_DIR + 'rw_ss_1.csv',
    'ss08': BASE_DIR + 'rw_ss_08.csv',
    'ff':   BASE_DIR + 'rw_ff.csv',
}
opt_trans_files = {
    'tt':   BASE_DIR + 'trans_WLunderopt_tt.csv',
    'ss1':  BASE_DIR + 'trans_WLunderopt_ss_1V.csv',
    'ss08': BASE_DIR + 'trans_WLunderopt_ss_0_8V.csv',
    'ff':   BASE_DIR + 'trans_WLunderopt_ff.csv',
}

# Read windows for optimized transient (verified from simulation timing)
opt_read_windows = [(7.0, 10.01), (15.0, 18.01)]

# ── compute values ────────────────────────────────────────────────────────────
corners_snm   = ['TT\n1.0V/27°C', 'SS\n1.0V/85°C', 'FF\n1.2V/85°C']
snm_base = [estimate_snm(*load_butterfly(base_snm_files[t])) for t in ['tt','ss','ff']]
snm_opt  = [estimate_snm(*load_butterfly(opt_snm_files[t]))  for t in ['tt','ss','ff']]

trans_info = [
    ('TT/1.0V/27°C', 'tt'),
    ('SS/1.0V/85°C', 'ss1'),
    ('SS/0.8V/85°C', 'ss08'),
    ('FF/1.2V/85°C', 'ff'),
]
disturb_base = [qb_disturb_baseline(base_trans_files[k]) for _, k in trans_info]
disturb_opt  = [qb_disturb_opt(opt_trans_files[k], opt_read_windows) for _, k in trans_info]
trans_labels = [l for l, _ in trans_info]

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Butterfly overlay
# ═══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor=DARK)
style_ax(ax1)

butterfly_styles = [
    ('TT/1.0V/27°C', 'tt', ACCENT),
    ('SS/1.0V/85°C', 'ss', AMBER),
    ('FF/1.2V/85°C', 'ff', GREEN),
]
for label, tag, color in butterfly_styles:
    x1b, y1b, x2b, y2b = load_butterfly(base_snm_files[tag])
    x1o, y1o, x2o, y2o = load_butterfly(opt_snm_files[tag])
    ax1.plot(x1b, y1b, color=color, lw=1.5, ls='--', alpha=0.5)
    ax1.plot(x2b, y2b, color=color, lw=1.5, ls='--', alpha=0.5, label=f'Baseline — {label}')
    ax1.plot(x1o, y1o, color=color, lw=2.0)
    ax1.plot(x2o, y2o, color=color, lw=2.0, label=f'WL Underdrive — {label}')

ax1.set_xlabel('Storage Node Voltage (V)', fontsize=11)
ax1.set_ylabel('Complementary Node Voltage (V)', fontsize=11)
ax1.set_title('Read SNM Butterfly: Baseline (dashed) vs WL Underdrive 0.8×V$_{DD}$ (solid)',
              fontsize=12, pad=10)
ax1.legend(fontsize=8, ncol=2, framealpha=0.15, labelcolor=WHITE,
           facecolor=PANEL, edgecolor='#30363D')
fig1.tight_layout()
fig1.savefig('/mnt/user-data/outputs/fig1_butterfly.png', dpi=160,
             bbox_inches='tight', facecolor=DARK)
print("Saved fig1_butterfly.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Read SNM bar chart
# ═══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(7, 5), facecolor=DARK)
style_ax(ax2)

x = np.arange(len(corners_snm))
w = 0.35
b1 = ax2.bar(x - w/2, snm_base, w, label='Baseline',      color=ACCENT, alpha=0.75, linewidth=0)
b2 = ax2.bar(x + w/2, snm_opt,  w, label='WL Underdrive', color=GREEN,  alpha=0.85, linewidth=0)
for bar, val in zip(list(b1) + list(b2), snm_base + snm_opt):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.0f}', ha='center', va='bottom', fontsize=9,
             color=WHITE, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(corners_snm, fontsize=9)
ax2.set_ylabel('Read SNM (mV)', fontsize=11)
ax2.set_title('Read SNM by Corner', fontsize=12, pad=10)
ax2.set_ylim(0, max(snm_opt) * 1.22)
ax2.legend(fontsize=9, framealpha=0.15, labelcolor=WHITE,
           facecolor=PANEL, edgecolor='#30363D')
ax2.yaxis.grid(True, alpha=0.15, color=MUTED)
fig2.tight_layout()
fig2.savefig('/mnt/user-data/outputs/fig2_read_snm_bars.png', dpi=160,
             bbox_inches='tight', facecolor=DARK)
print("Saved fig2_read_snm_bars.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — QB read disturb bar chart
# ═══════════════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(8, 5), facecolor=DARK)
style_ax(ax3)

x3 = np.arange(len(trans_labels))
b3 = ax3.bar(x3 - w/2, disturb_base, w, label='Baseline',      color=ACCENT, alpha=0.75, linewidth=0)
b4 = ax3.bar(x3 + w/2, disturb_opt,  w, label='WL Underdrive', color=GREEN,  alpha=0.85, linewidth=0)
for bar, val in zip(list(b3) + list(b4), disturb_base + disturb_opt):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.0f}', ha='center', va='bottom', fontsize=9,
             color=WHITE, fontweight='bold')
tick_labels = [l.replace('/', '\n') for l in trans_labels]
ax3.set_xticks(x3)
ax3.set_xticklabels(tick_labels, fontsize=8.5)
ax3.set_ylabel('QB Read Disturb Peak (mV)', fontsize=11)
ax3.set_title('QB Read Disturb Peak by Corner', fontsize=12, pad=10)
ax3.set_ylim(0, max(disturb_base) * 1.25)
ax3.legend(fontsize=9, framealpha=0.15, labelcolor=WHITE,
           facecolor=PANEL, edgecolor='#30363D')
ax3.yaxis.grid(True, alpha=0.15, color=MUTED)
fig3.tight_layout()
fig3.savefig('/mnt/user-data/outputs/fig3_qb_disturb_bars.png', dpi=160,
             bbox_inches='tight', facecolor=DARK)
print("Saved fig3_qb_disturb_bars.png")
