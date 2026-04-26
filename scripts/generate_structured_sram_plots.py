#!/usr/bin/env python3
"""Generate SRAM plots in a folder tree that mirrors sim_data."""

from __future__ import annotations

import argparse
import csv
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_DATA_ROOT = REPO_ROOT / "sim_data"
PLOT_ROOT = REPO_ROOT / "sram_analysis_plots"

SNM_CURVE_COLORS = ("#1f77b4", "#d62728")
SQUARE_COLOR = "#2ca02c"
CONTACT_COLOR = "#ff7f0e"
SIGNAL_COLORS = {
    "/Q": "#1f77b4",
    "/QB": "#d62728",
    "/WL": "#2ca02c",
    "/BL": "#9467bd",
    "/BLB": "#ff7f0e",
}
SIGNAL_ORDER = ("/Q", "/QB", "/WL", "/BL", "/BLB")
CURRENT_SIGNAL = "/V0/PLUS"

EYE_SCAN_SAMPLES = 6001
FIT_SCAN_SAMPLES = 6001
FIT_REFINE_SAMPLES = 20001
PLACEMENT_SCAN_SAMPLES = 1601
CONTACT_SCAN_SAMPLES = 4001
ROOT_TOL = 1e-8
SIDE_TOL = 1e-6
SLACK_TOL = 2e-6


@dataclass(frozen=True)
class Curve2D:
    x: np.ndarray
    y: np.ndarray
    label: str


@dataclass(frozen=True)
class SquareFit:
    side: float
    x_left: float
    y_bottom: float
    square_xy: np.ndarray
    contacts_xy: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SRAM plots under sram_analysis_plots."
    )
    parser.add_argument(
        "--sim-data-root",
        type=Path,
        default=SIM_DATA_ROOT,
        help="Root directory containing SRAM CSV exports.",
    )
    parser.add_argument(
        "--plot-root",
        type=Path,
        default=PLOT_ROOT,
        help="Output directory for generated plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim_data_root = args.sim_data_root.resolve()
    plot_root = args.plot_root.resolve()

    csv_paths = sorted(path for path in sim_data_root.rglob("*.csv") if path.is_file())
    if not csv_paths:
        raise SystemExit(f"No CSV files found under {sim_data_root}")

    generated: list[tuple[Path, float | None]] = []
    for csv_path in csv_paths:
        rel = csv_path.relative_to(sim_data_root)
        header, data = load_csv(csv_path)
        kind = classify_csv(header, data)

        if kind == "snm":
            output_path = plot_root / rel.with_suffix(".png")
            margin = plot_snm_csv(csv_path, header, data, output_path, sim_data_root)
            generated.append((output_path, margin))
        elif kind == "wnm":
            output_path = plot_root / rel.with_suffix(".png")
            margin = plot_wnm_csv(csv_path, header, data, output_path, sim_data_root)
            generated.append((output_path, margin))
        else:
            voltage_output = plot_root / rel.parent / f"{rel.stem}_voltages.png"
            current_output = plot_root / rel.parent / f"{rel.stem}_supply_current.png"
            plot_voltage_csv(csv_path, voltage_output, sim_data_root)
            plot_current_csv(csv_path, current_output, sim_data_root)
            generated.extend(((voltage_output, None), (current_output, None)))

    print(f"Generated {len(generated)} plot files under {plot_root}")
    for path, margin in generated:
        rel = path.relative_to(REPO_ROOT)
        if margin is None:
            print(f"  {rel}")
        elif margin > 0:
            print(f"  {rel}  (margin = {margin * 1000:.1f} mV)")
        else:
            print(f"  {rel}  (no eye)")


def classify_csv(header: list[str], data: np.ndarray) -> str:
    if data.ndim == 2 and data.shape[1] == 4:
        return "snm" if " vs " in " ".join(header).lower() else "wnm"
    return "waveform"


def plot_snm_csv(
    csv_path: Path,
    header: list[str],
    data: np.ndarray,
    output_path: Path,
    sim_data_root: Path,
) -> float:
    curve_a = Curve2D(data[:, 0], data[:, 1], "curve A")
    curve_b = Curve2D(data[:, 2], data[:, 3], "curve B")
    fit = largest_square_between_curves(curve_a, curve_b, choose_smallest=True)
    plot_butterfly(curve_a, curve_b, fit, "SNM", csv_path, output_path, sim_data_root)
    return 0.0 if fit is None else fit.side


def plot_wnm_csv(
    csv_path: Path,
    header: list[str],
    data: np.ndarray,
    output_path: Path,
    sim_data_root: Path,
) -> float:
    curve_left, curve_right = wnm_curves_from_csv(csv_path, header, data)
    fit = diagonal_square_from_crossings(curve_left, curve_right)
    if fit is None:
        fit = diagonal_square_between_curves(curve_left, curve_right)
    plot_butterfly(curve_left, curve_right, fit, "WNM", csv_path, output_path, sim_data_root)
    return 0.0 if fit is None else fit.side


def wnm_curves_from_csv(
    csv_path: Path,
    header: list[str],
    data: np.ndarray,
) -> tuple[Curve2D, Curve2D]:
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index in range(0, 4, 2):
        signal = signal_from_header(header[index])
        traces[signal] = (data[:, index], data[:, index + 1])

    if "/Q" not in traces or "/QB" not in traces:
        raise ValueError(f"{csv_path} is missing /Q or /QB write traces")

    q_sweep, q_values = traces["/Q"]
    qb_sweep, qb_values = traces["/QB"]
    if not np.allclose(q_sweep, qb_sweep, atol=1e-9, rtol=1e-6):
        raise ValueError(
            f"{csv_path} has mismatched /Q and /QB X-columns, so the write butterfly mapping is ambiguous"
        )

    common_sweep = 0.5 * (q_sweep + qb_sweep)
    sweep_max = float(np.max(common_sweep))
    q_high = float(np.max(q_values))
    scale = 1.0 if sweep_max <= 1e-12 else q_high / sweep_max
    sweep_v = common_sweep * scale

    # The ADE write exports use a shared sweep axis that is often normalized to
    # 0..1 even when the actual cell high level is not 1 V. Convert that sweep
    # back into volts, keep Q in the direct (sweep, Q) plane, and mirror only
    # QB into the same butterfly plane. This preserves the diagonal-corner WNM
    # construction on V1 = V2 while keeping the branch geometry physically
    # aligned in the FF/SS baseline plots.
    curve_left = Curve2D(qb_values, sweep_v, "QB(sweep) mirrored")
    curve_right = Curve2D(sweep_v, q_values, "Q(sweep) direct")
    return curve_left, curve_right


def plot_butterfly(
    curve_a: Curve2D,
    curve_b: Curve2D,
    fit: SquareFit | None,
    margin_label: str,
    csv_path: Path,
    output_path: Path,
    sim_data_root: Path,
) -> None:
    all_values = np.concatenate((curve_a.x, curve_a.y, curve_b.x, curve_b.y))
    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))
    pad = 0.05 * max(vmax - vmin, 1e-3)
    lo = vmin - pad
    hi = vmax + pad

    fig, ax = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
    ax.plot(curve_a.x, curve_a.y, lw=2.0, color=SNM_CURVE_COLORS[0], label=curve_a.label)
    ax.plot(curve_b.x, curve_b.y, lw=2.0, color=SNM_CURVE_COLORS[1], label=curve_b.label)
    ax.plot([lo, hi], [lo, hi], color="0.7", lw=0.8, ls="--", label="V1 = V2")

    if fit is not None:
        square_closed = np.vstack((fit.square_xy, fit.square_xy[:1]))
        ax.plot(
            square_closed[:, 0],
            square_closed[:, 1],
            color=SQUARE_COLOR,
            lw=2.0,
            label=f"{margin_label} = {fit.side * 1000:.1f} mV",
        )
        ax.fill(fit.square_xy[:, 0], fit.square_xy[:, 1], color=SQUARE_COLOR, alpha=0.10)
        if len(fit.contacts_xy) > 0:
            contact_marker_size = 6.0 if margin_label == "WNM" else 7.0
            ax.plot(
                fit.contacts_xy[:, 0],
                fit.contacts_xy[:, 1],
                "o",
                color=CONTACT_COLOR,
                markersize=contact_marker_size,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label="contact points",
                zorder=5,
            )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.88", linewidth=0.8)
    ax.set_xlabel("V1 (V)")
    ax.set_ylabel("V2 (V)")
    ax.set_title(title_from_path(csv_path.relative_to(sim_data_root)))
    ax.legend(loc="best", fontsize=8, framealpha=0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def largest_square_between_curves(
    curve_a: Curve2D,
    curve_b: Curve2D,
    *,
    choose_smallest: bool,
) -> SquareFit | None:
    x_a, _, interp_a = curve_as_y_of_x(curve_a)
    x_b, _, interp_b = curve_as_y_of_x(curve_b)

    x_lo = max(float(x_a[0]), float(x_b[0]))
    x_hi = min(float(x_a[-1]), float(x_b[-1]))
    if x_hi - x_lo <= SIDE_TOL:
        return None

    fits: list[SquareFit] = []
    for interval_lo, interval_hi in eye_intervals(interp_a, interp_b, x_lo, x_hi):
        fit = fit_square_in_interval(interp_a, interp_b, interval_lo, interval_hi)
        if fit is not None and fit.side > SIDE_TOL:
            fits.append(fit)

    if not fits:
        return None
    return min(fits, key=lambda item: item.side) if choose_smallest else max(fits, key=lambda item: item.side)


def curve_as_y_of_x(curve: Curve2D) -> tuple[np.ndarray, np.ndarray, PchipInterpolator]:
    order = np.argsort(curve.x)
    x_sorted = curve.x[order]
    y_sorted = curve.y[order]
    x_unique, y_unique = collapse_duplicate_x(x_sorted, y_sorted)
    if len(x_unique) < 2:
        raise ValueError("Need at least two unique x-points to interpolate the butterfly curve")
    return x_unique, y_unique, PchipInterpolator(x_unique, y_unique, extrapolate=False)


def collapse_duplicate_x(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_x: list[float] = []
    unique_y: list[float] = []
    group_x = float(x[0])
    group_y = [float(y[0])]

    for x_value, y_value in zip(x[1:], y[1:]):
        if abs(float(x_value) - group_x) <= 1e-10:
            group_y.append(float(y_value))
            continue
        unique_x.append(group_x)
        unique_y.append(float(np.mean(group_y)))
        group_x = float(x_value)
        group_y = [float(y_value)]

    unique_x.append(group_x)
    unique_y.append(float(np.mean(group_y)))
    return np.asarray(unique_x), np.asarray(unique_y)


def eye_intervals(
    interp_a: PchipInterpolator,
    interp_b: PchipInterpolator,
    x_lo: float,
    x_hi: float,
) -> list[tuple[float, float]]:
    grid = np.linspace(x_lo, x_hi, EYE_SCAN_SAMPLES)
    diff = interp_b(grid) - interp_a(grid)
    if np.all(np.isnan(diff)):
        return []

    roots = [x_lo]
    for index in range(len(grid) - 1):
        d0 = float(diff[index])
        d1 = float(diff[index + 1])
        if abs(d0) <= ROOT_TOL:
            roots.append(float(grid[index]))
        if abs(d1) <= ROOT_TOL:
            roots.append(float(grid[index + 1]))
        if d0 == 0.0 or d1 == 0.0 or d0 * d1 > 0.0:
            continue
        delta = d1 - d0
        root = float(grid[index]) if abs(delta) <= ROOT_TOL else float(grid[index] - d0 * (grid[index + 1] - grid[index]) / delta)
        roots.append(root)
    roots.append(x_hi)

    deduped = dedupe_sorted(roots, tol=max(ROOT_TOL, (x_hi - x_lo) / (EYE_SCAN_SAMPLES - 1)))
    intervals: list[tuple[float, float]] = []
    for start, stop in zip(deduped[:-1], deduped[1:]):
        if stop - start <= SIDE_TOL:
            continue
        midpoint = 0.5 * (start + stop)
        width = abs(float(interp_b(midpoint) - interp_a(midpoint)))
        if width > SIDE_TOL:
            intervals.append((start, stop))

    if intervals:
        return intervals
    if float(np.nanmax(np.abs(diff))) > SIDE_TOL:
        return [(x_lo, x_hi)]
    return []


def dedupe_sorted(values: list[float], tol: float) -> list[float]:
    ordered = sorted(values)
    deduped = [ordered[0]]
    for value in ordered[1:]:
        if abs(value - deduped[-1]) > tol:
            deduped.append(value)
    return deduped


def fit_square_in_interval(
    interp_a: PchipInterpolator,
    interp_b: PchipInterpolator,
    interval_lo: float,
    interval_hi: float,
) -> SquareFit | None:
    coarse = fit_square_on_grid(interp_a, interp_b, interval_lo, interval_hi, FIT_SCAN_SAMPLES)
    if coarse is None:
        return None

    refined = fit_square_on_grid(
        interp_a,
        interp_b,
        interval_lo,
        interval_hi,
        FIT_REFINE_SAMPLES,
        min_side=coarse["side"],
    )
    if refined is None:
        refined = coarse

    return refine_square_placement(interp_a, interp_b, interval_lo, interval_hi, refined)


def diagonal_square_between_curves(
    curve_left: Curve2D,
    curve_right: Curve2D,
) -> SquareFit | None:
    """Largest axis-aligned square whose lower-left and upper-right corners
    both lie on V1 = V2, fitting inside the eye between curve_left (left
    boundary, x = curve_left(y)) and curve_right (right boundary)."""
    _, _, interp_left_x = curve_as_x_of_y(curve_left)
    _, _, interp_right_x = curve_as_x_of_y(curve_right)

    y_lo = max(float(np.min(curve_left.y)), float(np.min(curve_right.y)))
    y_hi = min(float(np.max(curve_left.y)), float(np.max(curve_right.y)))
    if y_hi - y_lo <= SIDE_TOL:
        return None

    grid_n = 4001
    y_grid = np.linspace(y_lo, y_hi, grid_n)
    x_l = interp_left_x(y_grid)
    x_r = interp_right_x(y_grid)
    if np.any(np.isnan(x_l)) or np.any(np.isnan(x_r)):
        finite = ~(np.isnan(x_l) | np.isnan(x_r))
        if not np.any(finite):
            return None
        y_grid = y_grid[finite]
        x_l = x_l[finite]
        x_r = x_r[finite]

    best_side = 0.0
    best_anchor: float | None = None
    eps = max(SIDE_TOL, 1e-7)

    for i, a in enumerate(y_grid):
        s_max = float(y_grid[-1] - a)
        if s_max <= SIDE_TOL:
            continue
        s_lo = 0.0
        s_hi = s_max
        for _ in range(48):
            s = 0.5 * (s_lo + s_hi)
            j_hi = int(np.searchsorted(y_grid, a + s, side="right") - 1)
            if j_hi <= i:
                s_hi = s
                continue
            max_xl = float(np.max(x_l[i : j_hi + 1]))
            min_xr = float(np.min(x_r[i : j_hi + 1]))
            if max_xl <= a + eps and min_xr >= a + s - eps:
                s_lo = s
            else:
                s_hi = s
        if s_lo > best_side:
            best_side = s_lo
            best_anchor = float(a)

    if best_anchor is None or best_side <= SIDE_TOL:
        return None

    side = best_side
    x_left = best_anchor
    y_bottom = best_anchor
    corners = np.asarray(
        [
            [x_left, y_bottom],
            [x_left + side, y_bottom],
            [x_left + side, y_bottom + side],
            [x_left, y_bottom + side],
        ],
        dtype=float,
    )
    contacts = np.asarray(
        [[x_left, y_bottom], [x_left + side, y_bottom + side]],
        dtype=float,
    )
    return SquareFit(
        side=side,
        x_left=x_left,
        y_bottom=y_bottom,
        square_xy=corners,
        contacts_xy=contacts,
    )


def diagonal_square_from_crossings(
    curve_left: Curve2D,
    curve_right: Curve2D,
) -> SquareFit | None:
    left_crossings = curve_diagonal_crossings(curve_left)
    right_crossings = curve_diagonal_crossings(curve_right)
    if len(left_crossings) == 0 or len(right_crossings) == 0:
        return None

    lower = left_crossings[np.argmin(np.sum(left_crossings, axis=1))]
    upper = right_crossings[np.argmax(np.sum(right_crossings, axis=1))]
    side = float(upper[0] - lower[0])
    if side <= SIDE_TOL:
        return None

    x_left = float(lower[0])
    y_bottom = float(lower[1])
    corners = np.asarray(
        [
            [x_left, y_bottom],
            [x_left + side, y_bottom],
            [x_left + side, y_bottom + side],
            [x_left, y_bottom + side],
        ],
        dtype=float,
    )
    contacts = np.asarray([lower, upper], dtype=float)
    return SquareFit(
        side=side,
        x_left=x_left,
        y_bottom=y_bottom,
        square_xy=corners,
        contacts_xy=contacts,
    )


def curve_as_x_of_y(curve: Curve2D) -> tuple[np.ndarray, np.ndarray, PchipInterpolator]:
    order = np.argsort(curve.y)
    y_sorted = curve.y[order]
    x_sorted = curve.x[order]
    y_unique, x_unique = collapse_duplicate_x(y_sorted, x_sorted)
    if len(y_unique) < 2:
        raise ValueError("Need at least two unique y-points to invert the butterfly curve")
    return y_unique, x_unique, PchipInterpolator(y_unique, x_unique, extrapolate=False)


def lower_half_square_between_curves(curve_a: Curve2D, curve_b: Curve2D) -> SquareFit | None:
    x_a, _, interp_a = curve_as_y_of_x(curve_a)
    x_b, _, interp_b = curve_as_y_of_x(curve_b)

    x_lo = max(float(x_a[0]), float(x_b[0]))
    x_hi = min(float(x_a[-1]), float(x_b[-1]))
    if x_hi - x_lo <= SIDE_TOL:
        return None

    fits: list[SquareFit] = []
    for interval_lo, interval_hi in eye_intervals(interp_a, interp_b, x_lo, x_hi):
        fit = fit_square_in_interval(interp_a, interp_b, interval_lo, interval_hi)
        if fit is not None and fit.side > SIDE_TOL:
            fits.append(fit)

    if not fits:
        return None
    return min(fits, key=lambda fit: (fit.y_bottom, fit.x_left, -fit.side))


def fit_square_on_grid(
    interp_a: PchipInterpolator,
    interp_b: PchipInterpolator,
    interval_lo: float,
    interval_hi: float,
    sample_count: int,
    *,
    min_side: float = 0.0,
) -> dict[str, float | int | np.ndarray] | None:
    x = np.linspace(interval_lo, interval_hi, sample_count)
    y_a = interp_a(x)
    y_b = interp_b(x)
    lower = np.minimum(y_a, y_b)
    upper = np.maximum(y_a, y_b)
    clearance = upper - lower

    max_clearance = float(np.nanmax(clearance))
    max_side = min(interval_hi - interval_lo, max_clearance)
    if max_side <= SIDE_TOL or min_side > max_side:
        return None

    low = max(min_side, 0.0)
    high = max_side
    best_window: dict[str, float | int] | None = None

    for _ in range(32):
        side = 0.5 * (low + high)
        window = feasible_window(x, lower, upper, side)
        if window is None:
            high = side
        else:
            low = side
            best_window = window

    window = feasible_window(x, lower, upper, low) or best_window
    if window is None or low <= SIDE_TOL:
        return None

    return {
        "side": low,
        "start_index": int(window["start_index"]),
        "grid": x,
        "dx": float(x[1] - x[0]),
    }


def feasible_window(
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    side: float,
) -> dict[str, float | int] | None:
    if side <= SIDE_TOL:
        return {"start_index": 0, "gap": float(np.nanmax(upper - lower))}

    dx = float(x[1] - x[0])
    window_size = max(2, int(np.ceil(side / dx)) + 1)
    if window_size > len(x):
        return None

    lower_max = rolling_max(lower, window_size)
    upper_min = rolling_min(upper, window_size)
    gaps = upper_min - lower_max - side
    feasible_indices = np.flatnonzero(gaps >= -SLACK_TOL)
    if feasible_indices.size == 0:
        return None

    best = int(feasible_indices[np.argmin(np.abs(gaps[feasible_indices]))])
    return {"start_index": best, "gap": float(gaps[best])}


def rolling_max(values: np.ndarray, window_size: int) -> np.ndarray:
    queue: deque[int] = deque()
    out = np.empty(len(values) - window_size + 1)

    for index, value in enumerate(values):
        while queue and values[queue[-1]] <= value:
            queue.pop()
        queue.append(index)
        while queue[0] <= index - window_size:
            queue.popleft()
        if index >= window_size - 1:
            out[index - window_size + 1] = values[queue[0]]
    return out


def rolling_min(values: np.ndarray, window_size: int) -> np.ndarray:
    queue: deque[int] = deque()
    out = np.empty(len(values) - window_size + 1)

    for index, value in enumerate(values):
        while queue and values[queue[-1]] >= value:
            queue.pop()
        queue.append(index)
        while queue[0] <= index - window_size:
            queue.popleft()
        if index >= window_size - 1:
            out[index - window_size + 1] = values[queue[0]]
    return out


def refine_square_placement(
    interp_a: PchipInterpolator,
    interp_b: PchipInterpolator,
    interval_lo: float,
    interval_hi: float,
    grid_fit: dict[str, float | int | np.ndarray],
) -> SquareFit | None:
    side = float(grid_fit["side"])
    grid = grid_fit["grid"]
    if not isinstance(grid, np.ndarray):
        return None

    start_index = int(grid_fit["start_index"])
    dx = float(grid_fit["dx"])
    start_guess = float(grid[start_index])

    start_min = max(interval_lo, start_guess - 8.0 * dx)
    start_max = min(interval_hi - side, start_guess + 8.0 * dx)
    if start_max < start_min:
        start_candidates = np.asarray([min(max(start_guess, interval_lo), interval_hi - side)])
    else:
        start_candidates = np.linspace(start_min, start_max, 241)

    best: dict[str, float | np.ndarray] | None = None
    best_contacts = -1
    best_slack = float("inf")

    for start in start_candidates:
        candidate = evaluate_square_candidate(interp_a, interp_b, start, side)
        if candidate is None:
            continue
        contacts = candidate["contacts_xy"]
        contact_count = 0 if not isinstance(contacts, np.ndarray) else len(contacts)
        slack = abs(float(candidate["slack"]))
        if contact_count > best_contacts or (contact_count == best_contacts and slack < best_slack):
            best = candidate
            best_contacts = contact_count
            best_slack = slack

    if best is None:
        return None

    return square_fit_from_candidate(side, best)


def square_fit_from_candidate(side: float, candidate: dict[str, float | np.ndarray]) -> SquareFit:
    x_left = float(candidate["x_left"])
    y_bottom = float(candidate["y_bottom"])
    corners_xy = np.asarray(
        [
            [x_left, y_bottom],
            [x_left + side, y_bottom],
            [x_left + side, y_bottom + side],
            [x_left, y_bottom + side],
        ],
        dtype=float,
    )
    contacts_xy = np.asarray(candidate["contacts_xy"], dtype=float)
    return SquareFit(
        side=side,
        x_left=x_left,
        y_bottom=y_bottom,
        square_xy=corners_xy,
        contacts_xy=contacts_xy if len(contacts_xy) > 0 else np.zeros((0, 2)),
    )


def evaluate_square_candidate(
    interp_a: PchipInterpolator,
    interp_b: PchipInterpolator,
    x_left: float,
    side: float,
) -> dict[str, float | np.ndarray] | None:
    x_right = x_left + side
    probe_x = np.linspace(x_left, x_right, PLACEMENT_SCAN_SAMPLES)
    y_a = interp_a(probe_x)
    y_b = interp_b(probe_x)
    lower = np.minimum(y_a, y_b)
    upper = np.maximum(y_a, y_b)
    max_lower = float(np.max(lower))
    min_upper = float(np.min(upper))
    slack = min_upper - max_lower - side
    if slack < -SLACK_TOL:
        return None

    best: dict[str, float | np.ndarray] | None = None
    best_contacts = -1
    best_error = float("inf")
    y_candidates = dedupe_sorted([max_lower, min_upper - side], tol=1e-9)

    for y_bottom in y_candidates:
        contacts_xy, error = collect_square_contacts(interp_a, interp_b, x_left, y_bottom, side)
        contact_count = len(contacts_xy)
        if contact_count > best_contacts or (contact_count == best_contacts and error < best_error):
            best = {
                "x_left": float(x_left),
                "y_bottom": float(y_bottom),
                "contacts_xy": contacts_xy,
                "slack": float(slack),
            }
            best_contacts = contact_count
            best_error = error

    return best


def collect_square_contacts(
    interp_a: PchipInterpolator,
    interp_b: PchipInterpolator,
    x_left: float,
    y_bottom: float,
    side: float,
) -> tuple[np.ndarray, float]:
    x_right = x_left + side
    y_top = y_bottom + side
    contact_tol = max(5e-6, 0.0025 * max(side, 1e-3))

    x_dense = np.linspace(x_left, x_right, CONTACT_SCAN_SAMPLES)
    curve_a = np.column_stack((x_dense, interp_a(x_dense)))
    curve_b = np.column_stack((x_dense, interp_b(x_dense)))
    all_points = np.vstack((curve_a, curve_b))

    contacts: list[np.ndarray] = []
    total_error = 0.0

    for x_edge in (x_left, x_right):
        for interp in (interp_a, interp_b):
            point = np.asarray([x_edge, float(interp(x_edge))])
            if y_bottom - contact_tol <= point[1] <= y_top + contact_tol:
                contacts.append(point)

    for y_edge in (y_bottom, y_top):
        for curve_points in (curve_a, curve_b):
            horizontal_contact = nearest_horizontal_contact(curve_points, y_edge, x_left, x_right, contact_tol)
            if horizontal_contact is None:
                continue
            point, error = horizontal_contact
            contacts.append(point)
            total_error += error

    unique = unique_points(contacts)
    return unique, total_error


def nearest_horizontal_contact(
    points_xy: np.ndarray,
    target_y: float,
    x_left: float,
    x_right: float,
    tolerance: float,
) -> tuple[np.ndarray, float] | None:
    mask = (points_xy[:, 0] >= x_left - tolerance) & (points_xy[:, 0] <= x_right + tolerance)
    if not np.any(mask):
        return None
    candidates = points_xy[mask]
    distances = np.abs(candidates[:, 1] - target_y)
    index = int(np.argmin(distances))
    error = float(distances[index])
    if error > tolerance:
        return None
    return candidates[index], error


def unique_points(points: list[np.ndarray]) -> np.ndarray:
    if not points:
        return np.zeros((0, 2))
    unique: list[np.ndarray] = []
    for point in points:
        if any(np.allclose(point, other, atol=1e-7, rtol=0.0) for other in unique):
            continue
        unique.append(point)
    return np.asarray(unique, dtype=float)


def curve_diagonal_crossings(curve: Curve2D) -> np.ndarray:
    points: list[np.ndarray] = []
    for index in range(len(curve.x) - 1):
        x0 = float(curve.x[index])
        y0 = float(curve.y[index])
        x1 = float(curve.x[index + 1])
        y1 = float(curve.y[index + 1])
        d0 = y0 - x0
        d1 = y1 - x1

        if abs(d0) <= ROOT_TOL:
            points.append(np.asarray([x0, y0], dtype=float))
        if abs(d1) <= ROOT_TOL:
            points.append(np.asarray([x1, y1], dtype=float))
        if d0 * d1 >= 0.0:
            continue

        denom = d0 - d1
        if abs(denom) <= ROOT_TOL:
            continue
        t = d0 / denom
        point = np.asarray(
            [x0 + t * (x1 - x0), y0 + t * (y1 - y0)],
            dtype=float,
        )
        points.append(point)

    return unique_points(points)


def plot_voltage_csv(csv_path: Path, output_path: Path, sim_data_root: Path) -> None:
    signals = load_waveform_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)

    plotted = False
    for signal_name in SIGNAL_ORDER:
        waveform = signals.get(signal_name)
        if waveform is None:
            continue
        ax.plot(
            waveform["time_ns"],
            waveform["value"],
            lw=1.8,
            label=signal_name[1:],
            color=SIGNAL_COLORS[signal_name],
        )
        plotted = True

    if not plotted:
        raise ValueError(f"{csv_path} has no recognized voltage signals")

    ax.grid(True, color="0.9", linewidth=0.8)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"{title_from_path(csv_path.relative_to(sim_data_root))} - Voltages")
    ax.legend(ncol=min(5, len(ax.lines)), fontsize=8, loc="best", framealpha=0.95)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_current_csv(csv_path: Path, output_path: Path, sim_data_root: Path) -> None:
    signals = load_waveform_csv(csv_path)
    waveform = signals.get(CURRENT_SIGNAL)
    if waveform is None:
        raise ValueError(f"{csv_path} is missing {CURRENT_SIGNAL}")

    values = waveform["value"]
    scale, unit = current_scale(values)

    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    ax.plot(waveform["time_ns"], values * scale, lw=1.8, color="#444444")
    ax.grid(True, color="0.9", linewidth=0.8)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(f"I_VDD ({unit})")
    ax.set_title(f"{title_from_path(csv_path.relative_to(sim_data_root))} - Supply Current")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def load_csv(csv_path: Path) -> tuple[list[str], np.ndarray]:
    with csv_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]
    return header, np.asarray(rows, dtype=float)


def load_waveform_csv(csv_path: Path) -> dict[str, dict[str, np.ndarray]]:
    header, data = load_csv(csv_path)
    signals: dict[str, dict[str, np.ndarray]] = {}
    for index in range(0, len(header), 2):
        signal_name = signal_from_header(header[index])
        signals[signal_name] = {
            "time_ns": data[:, index] * 1e9,
            "value": data[:, index + 1],
        }
    return signals


def signal_from_header(header_cell: str) -> str:
    match = re.search(r"(/[^\s(,)\"]+)", header_cell)
    if not match:
        raise ValueError(f"Could not parse signal name from header cell {header_cell!r}")
    return match.group(1)


def title_from_path(rel_path: Path) -> str:
    parts = [pretty_token(part) for part in rel_path.parts[:-1]]
    stem = pretty_token(rel_path.stem)
    return " / ".join([*parts, stem])


def pretty_token(token: str) -> str:
    token = token.replace(".csv", "")
    token = token.replace("_", " ")
    return token


def current_scale(values: np.ndarray) -> tuple[float, str]:
    peak = float(np.nanmax(np.abs(values)))
    if peak >= 1e-3:
        return 1e3, "mA"
    if peak >= 1e-6:
        return 1e6, "uA"
    if peak >= 1e-9:
        return 1e9, "nA"
    return 1.0, "A"


if __name__ == "__main__":
    main()
