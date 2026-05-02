# viz_pq.py
import csv
import os
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


SEARCH_METHOD_ORDER = ("grid", "powell", "newton")
SEARCH_METHOD_COLORS = {
    "grid": "#1f77b4",
    "powell": "#d62728",
    "newton": "#2ca02c",
}
SEARCH_METHOD_MARKERS = {
    "grid": "o",
    "powell": "s",
    "newton": "^",
}


def _set_latex_style() -> None:
    """
    LaTeX-like rendering WITHOUT requiring a system LaTeX install (matplotlib mathtext).
    """
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",        # Computer Modern
        "axes.unicode_minus": False,
        "axes.grid": False,              # no background grid

        # --- font sizes (adjust as you like) ---
        "font.size": 17,                 # base size (affects many things)
        "axes.labelsize": 17,            # x/y label size
        "xtick.labelsize": 17,           # tick number size (x)
        "ytick.labelsize": 17,           # tick number size (y)
    })



def _latex_num_fixed(x: float, decimals: int = 2) -> str:
    # Fixed decimals for the "normal" range, scientific outside it.
    ax = abs(float(x))
    if ax == 0:
        return rf"${0:.{decimals}f}$"

    if ax < 1e-3 or ax >= 1e4:
        s = f"{x:.2e}"  # 1.23e-04
        mant, exp = s.split("e")
        exp = int(exp)
        return rf"${mant}\times 10^{{{exp}}}$"

    return rf"${x:.{decimals}f}$"


def _latex_num_e(x: float) -> str:
    # compact style: mantissa + 'e' in math, hyphen in TEXT, exponent in math
    if x == 0:
        return r"$0$"
    ax = abs(float(x))
    if ax < 1e-3 or ax >= 1e4:
        s = f"{x:.2e}"  # 1.23e-04
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".")
        exp = int(exp)

        if exp < 0:
            # $2e$-$4$  -> hyphen is text, exponent digits are math
            return rf"${mant}e$-" + rf"${abs(exp)}$"
        else:
            return rf"${mant}e{exp}$"

    return rf"${x:g}$"





def _stack_histories(histories: List[Sequence[float]]) -> np.ndarray:
    """
    Returns array of shape [R, T_max] filled with NaN where runs ended early.
    """
    if len(histories) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    T_max = max(len(h) for h in histories)
    R = len(histories)
    arr = np.full((R, T_max), np.nan, dtype=np.float32)
    for r, h in enumerate(histories):
        if len(h) > 0:
            arr[r, : len(h)] = np.asarray(h, dtype=np.float32)
    return arr


def _finite_mean_std(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Column-wise mean/std over finite entries only, without NumPy runtime warnings
    on all-NaN columns.
    Returns (mean, std, valid_mask).
    """
    values = np.asarray(arr, dtype=np.float64)
    finite_mask = np.isfinite(values)
    counts = finite_mask.sum(axis=0)
    valid_mask = counts > 0

    mean = np.full(values.shape[1], np.nan, dtype=np.float64)
    std = np.full(values.shape[1], np.nan, dtype=np.float64)
    if not valid_mask.any():
        return mean, std, valid_mask

    safe_values = np.where(finite_mask, values, 0.0)
    sums = safe_values.sum(axis=0)
    mean[valid_mask] = sums[valid_mask] / counts[valid_mask]

    centered = np.where(finite_mask, values - mean[np.newaxis, :], 0.0)
    sq_sums = (centered * centered).sum(axis=0)
    std[valid_mask] = np.sqrt(sq_sums[valid_mask] / counts[valid_mask])
    return mean, std, valid_mask


_SCHEDULE_LABELS = {
    "p": r"$p$",
    "q": r"$q$",
    "obj": r"$\mathrm{Objective}$",
    "rho": r"$\rho = q \cdot \frac{|\overline{E}|}{|E|}$",
}


_SCHEDULE_FORMATTERS = {
    "p": "fixed",
    "q": "scientific",
    "obj": "scientific",
    "rho": "scientific",
}


_SCHEDULE_LOG_SCALE = {"obj"}

def _plot_schedule(
    histories: List[Sequence[float]],
    *,
    tag: str,               # e.g., "cora_gcn_rade"
    name: str,              # "p" or "q"
    out_dir: str,
    show_runs: bool = True, # faint per-run lines in background
    dpi: int = 300,
) -> None:
    _set_latex_style()
    os.makedirs(out_dir, exist_ok=True)

    arr = _stack_histories(histories)  # [R, T]
    if arr.size == 0:
        return
    if not np.isfinite(arr).any():
        return

    mean, std, valid_cols = _finite_mean_std(arr)
    if not valid_cols.any():
        return
    epochs = np.arange(1, mean.shape[0] + 1)
    use_log_scale = name in _SCHEDULE_LOG_SCALE

    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    # No grid (explicit)
    ax.grid(False)

    # per-run traces (faint)
    if show_runs:
        for r in range(arr.shape[0]):
            y = arr[r]
            mask = ~np.isnan(y)
            if use_log_scale:
                mask = mask & (y > 0.0)
            if mask.any():
                ax.plot(epochs[mask], y[mask], linewidth=1.0, alpha=0.25)

    # mean + std shading
    mean_mask = valid_cols & ~np.isnan(mean)
    if use_log_scale:
        mean_mask = mean_mask & (mean > 0.0)
    if mean_mask.any():
        ax.plot(epochs[mean_mask], mean[mean_mask], linewidth=2.0)
        if use_log_scale:
            pos = arr[np.isfinite(arr) & (arr > 0.0)]
            lower_floor = float(pos.min()) if pos.size > 0 else 1e-12
            lower = np.maximum(mean - std, lower_floor)
            upper = mean + std
            fill_mask = mean_mask & np.isfinite(lower) & np.isfinite(upper) & (upper > 0.0)
            if fill_mask.any():
                ax.fill_between(epochs[fill_mask], lower[fill_mask], upper[fill_mask], alpha=0.20)
        else:
            fill_mask = mean_mask & np.isfinite(std)
            if fill_mask.any():
                ax.fill_between(epochs[fill_mask], (mean - std)[fill_mask], (mean + std)[fill_mask], alpha=0.20)

    # LaTeX-like labels (mathtext)
    ax.set_xlabel(r"$\mathrm{Epoch}$")
    ax.set_ylabel(_SCHEDULE_LABELS.get(name, rf"${name}$"))

    # Tick formatters:
    #  - x: integers in math mode
    #  - y: p uses ×10^{}, q uses compact e-notation to reduce axis width
    def _x_fmt(v, pos):
        return rf"${int(v)}$" if float(v).is_integer() else rf"${v:g}$"

    def _y_fmt(v, pos):
        formatter_kind = _SCHEDULE_FORMATTERS.get(name, "fixed")
        if formatter_kind == "scientific":
            return _latex_num_e(v)
        if formatter_kind == "fixed":
            if name == "p":
                return _latex_num_fixed(v, 2)  # e.g., 0.50, 0.55
            return _latex_num_fixed(v)
        return _latex_num_fixed(v)

    ax.xaxis.set_major_formatter(FuncFormatter(_x_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_y_fmt))
    if use_log_scale:
        ax.set_yscale("log")

    # No title
    fig.tight_layout()

    # filename format requested: e.g., visualization/cora_gcn_rade_p.png
    out_path = os.path.join(out_dir, f"{tag}_{name}.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def save_pq_plots(
    p_histories: List[Sequence[float]],
    q_histories: List[Sequence[float]],
    obj_histories: List[Sequence[float]],
    rho_histories: List[Sequence[float]],
    *,
    tag: str,  # e.g., "cora_gcn_rade"
    out_dir: str = "visualization",
    show_runs: bool = True,
) -> None:
    _plot_schedule(p_histories, tag=tag, name="p", out_dir=out_dir, show_runs=show_runs)
    _plot_schedule(q_histories, tag=tag, name="q", out_dir=out_dir, show_runs=show_runs)
    _plot_schedule(obj_histories, tag=tag, name="obj", out_dir=out_dir, show_runs=show_runs)
    _plot_schedule(rho_histories, tag=tag, name="rho", out_dir=out_dir, show_runs=show_runs)


def _ordered_method_histories(
    method_histories: Dict[str, List[Sequence[float]]],
) -> Dict[str, List[Sequence[float]]]:
    return {
        method: method_histories[method]
        for method in SEARCH_METHOD_ORDER
        if method in method_histories
    }


def _pad_histories(arr: np.ndarray, t_max: int) -> np.ndarray:
    if arr.shape[1] == t_max:
        return arr
    out = np.full((arr.shape[0], t_max), np.nan, dtype=np.float32)
    out[:, : arr.shape[1]] = arr
    return out


def _save_search_method_raw_data(
    method_histories: Dict[str, List[Sequence[float]]],
    *,
    tag: str,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    method_histories = _ordered_method_histories(method_histories)
    arrays = {
        method: _stack_histories(histories)
        for method, histories in method_histories.items()
    }
    arrays = {method: arr for method, arr in arrays.items() if arr.size > 0}
    if len(arrays) == 0:
        return

    methods = list(arrays.keys())
    t_max = max(arr.shape[1] for arr in arrays.values())
    arrays = {method: _pad_histories(arr, t_max) for method, arr in arrays.items()}
    num_runs = max(arr.shape[0] for arr in arrays.values())

    out_path = os.path.join(out_dir, f"{tag}_pq_search_obj_compare_raw.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["run", "epoch"]
        for method in methods:
            header.extend([f"{method}_obj", f"{method}_has_value"])
        for left_method, right_method in combinations(methods, 2):
            header.append(f"{right_method}_minus_{left_method}")
        writer.writerow(header)

        for run_idx in range(num_runs):
            for epoch_idx in range(t_max):
                row = [run_idx + 1, epoch_idx + 1]
                values = {}
                for method in methods:
                    arr = arrays[method]
                    has_run = run_idx < arr.shape[0]
                    val = float(arr[run_idx, epoch_idx]) if has_run else float("nan")
                    ok = np.isfinite(val)
                    values[method] = val
                    row.extend(["" if not ok else f"{val:.16g}", int(ok)])
                for left_method, right_method in combinations(methods, 2):
                    gap_val = values[right_method] - values[left_method]
                    row.append("" if not np.isfinite(gap_val) else f"{gap_val:.16g}")
                writer.writerow(row)


def _plot_search_method_objectives(
    method_histories: Dict[str, List[Sequence[float]]],
    *,
    tag: str,
    out_dir: str,
    show_runs: bool = True,
    dpi: int = 300,
) -> None:
    _set_latex_style()
    os.makedirs(out_dir, exist_ok=True)

    method_histories = _ordered_method_histories(method_histories)
    arrays = {
        method: _stack_histories(histories)
        for method, histories in method_histories.items()
    }
    arrays = {method: arr for method, arr in arrays.items() if arr.size > 0}
    if len(arrays) == 0:
        return

    t_max = max(arr.shape[1] for arr in arrays.values())
    arrays = {method: _pad_histories(arr, t_max) for method, arr in arrays.items()}

    valid_cols = np.zeros(t_max, dtype=bool)
    for arr in arrays.values():
        valid_cols |= np.isfinite(arr).any(axis=0)
    if not valid_cols.any():
        return

    arrays = {method: arr[:, valid_cols] for method, arr in arrays.items()}
    epochs = np.arange(1, next(iter(arrays.values())).shape[1] + 1)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.grid(False)

    if show_runs:
        for method, arr in arrays.items():
            color = SEARCH_METHOD_COLORS.get(method, None)
            for r in range(arr.shape[0]):
                y = arr[r]
                mask = ~np.isnan(y)
                if mask.any():
                    ax.plot(epochs[mask], y[mask], linewidth=1.0, alpha=0.18, color=color)

    for method, arr in arrays.items():
        mean, std, valid_mask = _finite_mean_std(arr)
        if not valid_mask.any():
            continue
        color = SEARCH_METHOD_COLORS.get(method, None)
        ax.plot(epochs[valid_mask], mean[valid_mask], linewidth=2.2, color=color, label=method)
        ax.fill_between(
            epochs[valid_mask],
            (mean - std)[valid_mask],
            (mean + std)[valid_mask],
            alpha=0.18,
            color=color,
        )

    ax.set_xlabel(r"$\mathrm{Epoch}$")
    ax.set_ylabel(r"$\mathrm{Objective}$")
    ax.set_yscale("log")

    def _x_fmt(v, pos):
        return rf"${int(v)}$" if float(v).is_integer() else rf"${v:g}$"

    ax.xaxis.set_major_formatter(FuncFormatter(_x_fmt))
    ax.legend(frameon=False, loc="lower left", fontsize=12)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{tag}_pq_search_obj_compare.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_search_method_gap(
    method_histories: Dict[str, List[Sequence[float]]],
    *,
    tag: str,
    out_dir: str,
    show_runs: bool = True,
    dpi: int = 300,
) -> None:
    _set_latex_style()
    os.makedirs(out_dir, exist_ok=True)

    method_histories = _ordered_method_histories(method_histories)
    arrays = {
        method: _stack_histories(histories)
        for method, histories in method_histories.items()
    }
    arrays = {method: arr for method, arr in arrays.items() if arr.size > 0}
    methods = list(arrays.keys())
    if len(arrays) < 2:
        return

    t_max = max(arr.shape[1] for arr in arrays.values())
    arrays = {method: _pad_histories(arr, t_max) for method, arr in arrays.items()}
    valid_cols = np.zeros(t_max, dtype=bool)
    for arr in arrays.values():
        valid_cols |= np.isfinite(arr).any(axis=0)
    if not valid_cols.any():
        return
    arrays = {method: arr[:, valid_cols] for method, arr in arrays.items()}
    epochs = np.arange(1, next(iter(arrays.values())).shape[1] + 1)

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.grid(False)

    ax.axhline(0.0, linewidth=1.2, linestyle="--", color="#111111", alpha=0.8)
    pair_colors = ["#4c4c4c", "#9467bd", "#8c564b"]

    for pair_idx, (left_method, right_method) in enumerate(combinations(methods, 2)):
        gap_arr = arrays[right_method] - arrays[left_method]
        gap_mean, gap_std, valid_mask = _finite_mean_std(gap_arr)
        if not valid_mask.any():
            continue
        color = pair_colors[pair_idx % len(pair_colors)]

        if show_runs:
            for r in range(gap_arr.shape[0]):
                y = gap_arr[r]
                mask = ~np.isnan(y)
                if mask.any():
                    ax.plot(epochs[mask], y[mask], linewidth=1.0, alpha=0.14, color=color)

        ax.plot(
            epochs[valid_mask],
            gap_mean[valid_mask],
            linewidth=2.2,
            color=color,
            label=f"{right_method} - {left_method}",
        )
        ax.fill_between(
            epochs[valid_mask],
            (gap_mean - gap_std)[valid_mask],
            (gap_mean + gap_std)[valid_mask],
            alpha=0.16,
            color=color,
        )

    ax.set_xlabel(r"$\mathrm{Epoch}$")
    ax.set_ylabel(r"$\mathrm{Objective\ gap}$", fontsize=16)

    def _x_fmt(v, pos):
        return rf"${int(v)}$" if float(v).is_integer() else rf"${v:g}$"

    ax.xaxis.set_major_formatter(FuncFormatter(_x_fmt))
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{tag}_pq_search_obj_gap.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_pq_search_comparison_plots(
    method_obj_histories: Dict[str, List[Sequence[float]]],
    *,
    tag: str,
    out_dir: str = "visualization",
    show_runs: bool = True,
) -> None:
    _save_search_method_raw_data(
        method_obj_histories,
        tag=tag,
        out_dir=out_dir,
    )
    _plot_search_method_objectives(
        method_obj_histories,
        tag=tag,
        out_dir=out_dir,
        show_runs=show_runs,
    )
    _plot_search_method_gap(
        method_obj_histories,
        tag=tag,
        out_dir=out_dir,
        show_runs=show_runs,
    )


def _save_surface_raw_data(snapshot: Dict[str, object], *, out_path: str) -> None:
    p_values = np.asarray(snapshot["p_values"], dtype=np.float64)
    second_axis_name = str(snapshot.get("second_axis_name", "q")).strip() or "q"
    second_values = np.asarray(snapshot["q_values"], dtype=np.float64)
    objective = np.asarray(snapshot["objective"], dtype=np.float64)
    g_reg = np.asarray(snapshot["G_reg"], dtype=np.float64)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["p", second_axis_name, "objective", "G_reg"])
        for second_idx, second_val in enumerate(second_values):
            for p_idx, p_val in enumerate(p_values):
                writer.writerow(
                    [
                        f"{float(p_val):.16g}",
                        f"{float(second_val):.16g}",
                        f"{float(objective[second_idx, p_idx]):.16g}",
                        f"{float(g_reg[second_idx, p_idx]):.16g}",
                    ]
                )


def _finite_positive(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x) & (x > 0.0)]


def _build_relative_surface(objective: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    """
    Relative objective surface:
        obj_rel = obj / min(obj)
    on finite positive entries.
    Best point is always 1.0.
    """
    z = np.asarray(objective, dtype=np.float64).copy()
    pos = _finite_positive(z)
    if pos.size == 0:
        return np.ones_like(z, dtype=np.float64)

    z_min = max(float(pos.min()), eps)
    z = np.where(np.isfinite(z), np.maximum(z, z_min) / z_min, np.nan)
    return z


def _build_log10_relative_surface(objective: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    """
    Log-relative objective surface:
        log10(obj / min(obj))
    on finite positive entries.
    Best point is always 0.0.
    """
    rel = _build_relative_surface(objective, eps=eps)
    rel = np.where(np.isfinite(rel), np.maximum(rel, 1.0), np.nan)
    return np.log10(rel)


def _compute_shared_log10_relative_vmax(
    surface_snapshots: List[Dict[str, object]],
    *,
    quantile: float = 0.995,
    min_vmax: float = 1e-3,
) -> float:
    """
    Shared upper limit across all frames, based on log10 relative objective values.
    Uses a high quantile to avoid one extreme value dominating all frames.
    """
    vals = []
    for snapshot in surface_snapshots:
        objective = np.asarray(snapshot["objective"], dtype=np.float64)
        rel_log10 = _build_log10_relative_surface(objective)
        finite = rel_log10[np.isfinite(rel_log10)]
        if finite.size > 0:
            vals.append(finite)

    if len(vals) == 0:
        return min_vmax

    vals = np.concatenate(vals, axis=0)
    vmax = float(np.quantile(vals, quantile))
    vmax = max(vmax, min_vmax)
    return vmax


def _plot_surface_snapshot(
    snapshot: Dict[str, object],
    *,
    tag: str,
    run_idx: int,
    out_dir: str,
    dpi: int,
    shared_log10_relative_vmax: float,
) -> str:
    _set_latex_style()
    os.makedirs(out_dir, exist_ok=True)

    epoch = int(snapshot["epoch"])
    p_values = np.asarray(snapshot["p_values"], dtype=np.float64)
    second_axis_name = str(snapshot.get("second_axis_name", "q")).strip() or "q"
    second_values = np.asarray(snapshot["q_values"], dtype=np.float64)
    objective = np.asarray(snapshot["objective"], dtype=np.float64)
    search_results = {
        str(method): dict(result)
        for method, result in dict(snapshot["search_results"]).items()
    }

    frame_base = f"{tag}_run{run_idx:02d}_epoch{epoch:03d}"
    csv_path = os.path.join(out_dir, f"{frame_base}.csv")
    _save_surface_raw_data(snapshot, out_path=csv_path)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    ax.grid(False)

    if p_values.size > 1 and second_values.size > 1:
        p_mesh, second_mesh = np.meshgrid(p_values, second_values)

        # Log-relative surface: best value = 0
        z_plot = _build_log10_relative_surface(objective)
        finite = z_plot[np.isfinite(z_plot)]

        if finite.size > 0 and float(finite.max()) > 0.0:
            vmax = max(shared_log10_relative_vmax, 1e-3)
            levels = np.linspace(0.0, vmax, num=16)

            contourf = ax.contourf(
                p_mesh,
                second_mesh,
                z_plot,
                levels=levels,
                cmap="viridis",
                extend="max",
            )
            ax.contour(
                p_mesh,
                second_mesh,
                z_plot,
                levels=levels,
                colors="white",
                linewidths=0.55,
                alpha=0.75,
            )
        else:
            contourf = ax.contourf(
                p_mesh,
                q_mesh,
                np.zeros_like(z_plot),
                levels=8,
                cmap="viridis",
            )

        cbar = fig.colorbar(contourf, ax=ax)
        cbar.set_label(r"$\log_{10}(\mathrm{obj} / \mathrm{obj}_{\min})$")

        ax.set_xlabel(r"$p$")
        ax.set_ylabel(r"$\rho$" if second_axis_name == "rho" else r"$q$")

        # small padding so boundary points at the search limits are fully visible
        p_pad = 0.02 * max(float(p_values.max() - p_values.min()), 1e-12)
        second_pad = 0.05 * max(float(second_values.max() - second_values.min()), 1e-12)
        ax.set_xlim(float(p_values.min()) - p_pad, float(p_values.max()) + p_pad)
        ax.set_ylim(float(second_values.min()) - second_pad, float(second_values.max()) + second_pad)

        for method in SEARCH_METHOD_ORDER:
            if method not in search_results:
                continue
            result = search_results[method]
            ax.scatter(
                [float(result["p_best"])],
                [float(result.get(f"{second_axis_name}_best", result["q_best"]))],
                s=82,
                color=SEARCH_METHOD_COLORS.get(method, "#111111"),
                marker=SEARCH_METHOD_MARKERS.get(method, "o"),
                edgecolors="white",
                linewidths=0.9,
                label=f"{method}: {float(result['obj']):.2e}",
                zorder=3,
                clip_on=False,
            )
    else:
        # 1D fallback
        if p_values.size > 1:
            x_vals = p_values
            y_vals = objective[0, :]
            x_label = r"$p$"
            key_name = "p_best"
        else:
            x_vals = second_values
            y_vals = objective[:, 0]
            x_label = r"$\rho$" if second_axis_name == "rho" else r"$q$"
            key_name = f"{second_axis_name}_best" if second_axis_name == "rho" else "q_best"

        pos = _finite_positive(y_vals)
        if pos.size > 0:
            y_floor = float(pos.min())
            y_vals = np.maximum(y_vals, y_floor)

        ax.plot(x_vals, y_vals, linewidth=2.2, color="#1f77b4")
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\mathrm{Objective}$")
        ax.set_yscale("log")

        for method in SEARCH_METHOD_ORDER:
            if method not in search_results:
                continue
            result = search_results[method]
            ax.scatter(
                [float(result[key_name])],
                [float(result["obj"])],
                s=76,
                color=SEARCH_METHOD_COLORS.get(method, "#111111"),
                marker=SEARCH_METHOD_MARKERS.get(method, "o"),
                edgecolors="white",
                linewidths=0.8,
                label=method,
                zorder=3,
                clip_on=False,
            )

    ax.set_title(rf"$\mathrm{{Objective\ surface,\ epoch}}\ {epoch}$", pad=10)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    frame_path = os.path.join(out_dir, f"{frame_base}.png")
    fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return frame_path


def _save_surface_animation(
    frame_paths: List[str],
    *,
    gif_path: str,
    duration_ms: int = 600,
) -> bool:
    if len(frame_paths) <= 1 or Image is None:
        return False

    raw_frames = [Image.open(path).convert("RGBA") for path in frame_paths]
    try:
        max_width = max(frame.width for frame in raw_frames)
        max_height = max(frame.height for frame in raw_frames)

        frames = []
        for frame in raw_frames:
            canvas = Image.new("RGBA", (max_width, max_height), (255, 255, 255, 255))
            offset = (
                (max_width - frame.width) // 2,
                (max_height - frame.height) // 2,
            )
            canvas.paste(frame, offset)
            frames.append(canvas.convert("P", palette=Image.ADAPTIVE))

        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(duration_ms),
            loop=0,
            optimize=False,
            disposal=2,
        )
    finally:
        for frame in raw_frames:
            frame.close()
    return True


def save_pq_objective_surface_plots(
    surface_snapshots: List[Dict[str, object]],
    *,
    tag: str,
    run_idx: int,
    out_dir: str = "visualization",
    create_animation: bool = False,
    dpi: int = 260,
) -> None:
    if len(surface_snapshots) == 0:
        return

    surface_dir = os.path.join(out_dir, f"{tag}_run{run_idx:02d}_pq_surface")
    os.makedirs(surface_dir, exist_ok=True)

    ordered_snapshots = sorted(surface_snapshots, key=lambda snapshot: int(snapshot["epoch"]))
    shared_log10_relative_vmax = _compute_shared_log10_relative_vmax(ordered_snapshots)

    frame_paths = [
        _plot_surface_snapshot(
            snapshot,
            tag=tag,
            run_idx=run_idx,
            out_dir=surface_dir,
            dpi=dpi,
            shared_log10_relative_vmax=shared_log10_relative_vmax,
        )
        for snapshot in ordered_snapshots
    ]

    if create_animation:
        gif_path = os.path.join(surface_dir, f"{tag}_run{run_idx:02d}_pq_surface.gif")
        _save_surface_animation(frame_paths, gif_path=gif_path)
