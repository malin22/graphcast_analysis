"""Publication-ready AR perturbation figures for the GraphCast report.

Produces two core figures:
  Figure 1: spatial ΔIVT maps for negative/positive perturbations at early,
            middle, and late lead times, with ClimateNet contours where available.
  Figure 2: immediate (+6 h) AR-mask dose response plus the absolute global
            mean IVT trajectories, with gamma=0 shown as the baseline.

Each composite figure is also exported as separate panel PNGs for flexible
assembly in LaTeX.

The script intentionally keeps the report figures selective. Diagnostic plots,
videos, precipitation metrics, etc. remain in evaluate_ar.py.
"""

import os
import glob
from contextlib import closing

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from evaluation_helpers import (
    area_weighted_mean,
    discover_files,
    format_lead_time,
    get_lat_name,
    get_lon_name,
    get_valid_time,
    load_mask_on_grid,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WEATHER_FEATURE = "AR"
ACTIVATION_TYPE = "raw_activations_new_normalization_no_threshold"
CENTER_STR = "2021-02-12T18"
NODE_HIERARCHY_LEVEL = 6
CONTROL_GAMMA = 0.0
MAX_MASK_TIME_DIFFERENCE_HOURS = 3

# Moderate, symmetric perturbations for the spatial figure and time summary.
REPORT_GAMMAS = (-0.5, 0.5)

# Requested representative lead times. The nearest available model step is used.
MAP_LEAD_HOURS = (24, 72, 120)

# If None, "outside" means the full forecast domain outside the ClimateNet mask.
# To use a local analysis box, set e.g. (lon_min, lon_max, lat_min, lat_max).
REPORT_DOMAIN = None

Q_VAR = "specific_humidity"
U_VAR = "u_component_of_wind"
V_VAR = "v_component_of_wind"
G = 9.80665

BASE_DIR = os.path.join(
    "results", "perturbation", WEATHER_FEATURE,
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}",
    ACTIVATION_TYPE,
    CENTER_STR,
)
INPUT_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "evaluation", "report_figures")
PANEL_DIR = os.path.join(OUT_DIR, "panels")
MASK_DIR = f"/share/prj-4d/graphcast_shared/data/ClimateNetLarge/{WEATHER_FEATURE}_labels_cleaned"

# Publication defaults. Keep typography restrained and let the data dominate.
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": 400,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# -----------------------------------------------------------------------------
# Core calculations
# -----------------------------------------------------------------------------
def compute_ivt(ds):
    """Compute IVT magnitude integrated over 200--1000 hPa."""
    q = ds[Q_VAR]
    u = ds[U_VAR]
    v = ds[V_VAR]

    if "level" not in q.dims:
        raise ValueError("Expected pressure-level variables with dimension 'level'.")

    levels_hpa = ds["level"].values.astype(float)
    keep = (levels_hpa >= 200.0) & (levels_hpa <= 1000.0)
    q = q.isel(level=keep)
    u = u.isel(level=keep)
    v = v.isel(level=keep)

    p = q["level"].values.astype(float)
    if np.nanmax(p) < 2000:
        p = p * 100.0

    order = np.argsort(p)
    p = p[order]
    q = q.isel(level=order)
    u = u.isel(level=order)
    v = v.isel(level=order)
    axis = q.get_axis_num("level")

    ivt_u = np.trapezoid((q * u).values, x=p, axis=axis) / G
    ivt_v = np.trapezoid((q * v).values, x=p, axis=axis) / G

    dims = [d for d in q.dims if d != "level"]
    coords = {d: q.coords[d] for d in dims if d in q.coords}
    return xr.DataArray(
        np.sqrt(ivt_u ** 2 + ivt_v ** 2),
        dims=dims,
        coords=coords,
        name="ivt",
    )


def open_forecast(path):
    """Open one forecast and remove the singleton batch dimension."""
    ds = xr.open_dataset(path)
    if "batch" in ds.dims:
        ds = ds.isel(batch=0)
    return ds


def maybe_subset_domain(da):
    """Optionally restrict a field to REPORT_DOMAIN."""
    if REPORT_DOMAIN is None:
        return da

    lon_min, lon_max, lat_min, lat_max = REPORT_DOMAIN
    lat = get_lat_name(da)
    lon = get_lon_name(da)

    # Latitude is normally sorted ascending by the GraphCast output.
    da = da.sel({lat: slice(lat_min, lat_max)})

    # This simple branch assumes the requested longitude interval does not cross
    # the coordinate seam. For a dateline-crossing box, pre-wrap coordinates.
    da = da.sel({lon: slice(lon_min, lon_max)})
    return da


def lead_hours_for_step(ds, t_idx):
    valid = get_valid_time(ds, t_idx, CENTER_STR)
    return float((valid - pd.Timestamp(CENTER_STR)) / pd.Timedelta(hours=1))


def nearest_step_for_lead(ds, requested_lead_h):
    leads = np.array([lead_hours_for_step(ds, i) for i in range(ds.sizes["time"])])
    return int(np.argmin(np.abs(leads - requested_lead_h)))


def get_gamma_path(file_table, gamma):
    hit = file_table[np.isclose(file_table["gamma"].astype(float), gamma)]
    if hit.empty:
        raise ValueError(
            f"Requested gamma={gamma:g} not found. Available: "
            f"{sorted(file_table['gamma'].unique())}"
        )
    return hit.iloc[0]["file"]


def build_ivt_cache(file_table):
    """Load IVT trajectories once for all gamma values."""
    cache = {}
    valid_times = None
    lead_hours = None

    for _, row in file_table.iterrows():
        gamma = float(row["gamma"])
        ds = open_forecast(row["file"])
        try:
            ivt = compute_ivt(ds).load()
            cache[gamma] = ivt
            if valid_times is None:
                valid_times = [get_valid_time(ds, i, CENTER_STR) for i in range(ds.sizes["time"])]
                lead_hours = np.array([
                    (t - pd.Timestamp(CENTER_STR)) / pd.Timedelta(hours=1)
                    for t in valid_times
                ], dtype=float)
        finally:
            ds.close()

    return cache, valid_times, lead_hours


def collect_mask_matches(control_ivt, valid_times, lead_hours):
    """Find all distinct ClimateNet masks matched to forecast steps.

    If two forecast steps happen to map to the same ClimateNet file, retain the
    forecast step with the smaller time mismatch. This prevents duplicate masks
    from appearing as separate verification points.
    """
    candidates = []
    for t_idx, (valid_time, lead_h) in enumerate(zip(valid_times, lead_hours)):
        target = maybe_subset_domain(control_ivt.isel(time=t_idx))
        try:
            mask, mask_path, mask_time, diff_h = load_mask_on_grid(
                valid_time,
                target,
                MASK_DIR,
                MAX_MASK_TIME_DIFFERENCE_HOURS,
            )
            candidates.append({
                "time_index": t_idx,
                "valid_time": pd.Timestamp(valid_time),
                "lead_hours": float(lead_h),
                "mask": mask.load(),
                "mask_path": mask_path,
                "mask_time": pd.Timestamp(mask_time),
                "mask_time_diff_h": float(diff_h),
            })
        except Exception:
            continue

    # One forecast verification per distinct external mask.
    best_by_mask = {}
    for rec in candidates:
        key = os.path.abspath(rec["mask_path"])
        if key not in best_by_mask or rec["mask_time_diff_h"] < best_by_mask[key]["mask_time_diff_h"]:
            best_by_mask[key] = rec

    matches = sorted(best_by_mask.values(), key=lambda r: r["lead_hours"])
    if not matches:
        raise RuntimeError(
            f"No ClimateNet masks matched any forecast step within "
            f"{MAX_MASK_TIME_DIFFERENCE_HOURS} h."
        )
    return matches


def compute_mask_metrics(ivt_cache, matches):
    """Compute inside/outside ΔIVT dose responses at every matched mask time."""
    control = ivt_cache[CONTROL_GAMMA]
    records = []

    for match in matches:
        t_idx = match["time_index"]
        mask = match["mask"]
        control_t = maybe_subset_domain(control.isel(time=t_idx))

        for gamma in sorted(ivt_cache):
            ivt_t = maybe_subset_domain(ivt_cache[gamma].isel(time=t_idx))
            delta = ivt_t - control_t

            records.append({
                "gamma": gamma,
                "time_index": t_idx,
                "lead_hours": match["lead_hours"],
                "forecast_valid_time": match["valid_time"],
                "mask_time": match["mask_time"],
                "mask_time_diff_h": match["mask_time_diff_h"],
                "mask_file": match["mask_path"],
                "delta_ivt_inside_mask_mean": area_weighted_mean(delta, mask),
                "delta_ivt_outside_mask_mean": area_weighted_mean(delta, ~mask),
            })

    return pd.DataFrame(records).sort_values(["lead_hours", "gamma"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Global trajectory metric
# -----------------------------------------------------------------------------
def compute_global_ivt_trajectory(ivt_cache, lead_hours):
    """Area-weighted absolute global mean IVT for every gamma and lead time."""
    records = []
    for gamma in sorted(ivt_cache):
        for t_idx, lead_h in enumerate(lead_hours):
            ivt = ivt_cache[gamma].isel(time=t_idx)
            records.append({
                "gamma": float(gamma),
                "lead_hours": float(lead_h),
                "global_mean_ivt": area_weighted_mean(ivt),
            })
    return pd.DataFrame(records)


def save_panel(fig, filename):
    """Save a standalone panel PNG."""
    path = os.path.join(PANEL_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


# -----------------------------------------------------------------------------
# Figure 1: spatial response
# -----------------------------------------------------------------------------
def add_mask_contour_if_available(ax, field, valid_time):
    try:
        mask, _, mask_time, diff_h = load_mask_on_grid(
            valid_time,
            field,
            MASK_DIR,
            MAX_MASK_TIME_DIFFERENCE_HOURS,
        )
    except Exception:
        return None

    lat = get_lat_name(field)
    lon = get_lon_name(field)
    ax.contour(
        field[lon].values,
        field[lat].values,
        mask.values.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=0.8,
    )
    return mask_time, diff_h


def draw_spatial_panel(ax, ivt_cache, gamma, idx, valid_times, vmax, show_mask=True):
    control = ivt_cache[CONTROL_GAMMA]
    delta = maybe_subset_domain(ivt_cache[gamma].isel(time=idx) - control.isel(time=idx))
    lat = get_lat_name(delta)
    lon = get_lon_name(delta)
    mesh = ax.pcolormesh(
        delta[lon].values, delta[lat].values, delta.values,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto", rasterized=True,
    )
    #if show_mask:
        #add_mask_contour_if_available(ax, delta, valid_times[idx])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return mesh


def make_figure1(ivt_cache, valid_times, lead_hours):
    neg_gamma, pos_gamma = REPORT_GAMMAS
    for g in REPORT_GAMMAS:
        if g not in ivt_cache:
            raise ValueError(f"Figure 1 requires gamma={g:g}.")

    control = ivt_cache[CONTROL_GAMMA]
    step_indices = [int(np.argmin(np.abs(lead_hours - h))) for h in MAP_LEAD_HOURS]

    displayed = []
    for gamma in (neg_gamma, pos_gamma):
        for idx in step_indices:
            delta = maybe_subset_domain(ivt_cache[gamma].isel(time=idx) - control.isel(time=idx))
            displayed.append(np.asarray(delta.values).ravel())
    all_values = np.concatenate(displayed)
    vmax = np.nanpercentile(np.abs(all_values), 99)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = np.nanmax(np.abs(all_values))

    # Composite figure.
    fig, axes = plt.subplots(
        2, len(step_indices), figsize=(10.5, 5.2),
        sharex=True, sharey=True, constrained_layout=True,
    )
    last_mesh = None
    for r, gamma in enumerate((neg_gamma, pos_gamma)):
        for c, idx in enumerate(step_indices):
            ax = axes[r, c]
            last_mesh = draw_spatial_panel(ax, ivt_cache, gamma, idx, valid_times, vmax)
            if r == 0:
                ax.set_title(f"+{format_lead_time(round(lead_hours[idx]))}")
            if c == 0:
                ax.set_ylabel(rf"$\gamma={gamma:+g}$" + "\nLatitude")
            else:
                ax.set_ylabel("")
            if r != 1:
                ax.set_xlabel("")

    cbar = fig.colorbar(last_mesh, ax=axes, location="right", shrink=0.90, pad=0.02)
    cbar.set_label(r"$\Delta$IVT [kg m$^{-1}$ s$^{-1}$]")
    stem = os.path.join(OUT_DIR, "figure1_spatial_delta_ivt")
    fig.savefig(stem + ".png", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", stem + ".png")

    # Standalone panels, all with the same color normalization as the composite.
    for gamma in (neg_gamma, pos_gamma):
        for idx in step_indices:
            panel_fig, panel_ax = plt.subplots(figsize=(5.0, 3.2), constrained_layout=True)
            mesh = draw_spatial_panel(panel_ax, ivt_cache, gamma, idx, valid_times, vmax)
            panel_ax.set_title(rf"$\gamma={gamma:+g}$, +{format_lead_time(round(lead_hours[idx]))}")
            cb = panel_fig.colorbar(mesh, ax=panel_ax, pad=0.02)
            cb.set_label(r"$\Delta$IVT [kg m$^{-1}$ s$^{-1}$]")
            save_panel(
                panel_fig,
                f"figure1_gamma_{gamma:+g}_lead_{int(round(lead_hours[idx])):03d}h.png".replace("+", "p").replace("-", "m"),
            )


# -----------------------------------------------------------------------------
# Figure 2: quantitative AR-specific response
# -----------------------------------------------------------------------------
def draw_dose_panel(ax, metrics, lead_h, ylims=None, title=None, show_legend=True):
    group = metrics[np.isclose(metrics["lead_hours"], lead_h)].sort_values("gamma")
    ax.plot(
        group["gamma"], group["delta_ivt_inside_mask_mean"],
        marker="o", linewidth=2.0, label="Inside AR mask",
    )
    ax.plot(
        group["gamma"], group["delta_ivt_outside_mask_mean"],
        marker="s", linewidth=1.8, linestyle="--", label="Outside AR mask",
    )
    ax.axhline(0, linewidth=0.8, color="0.45")
    ax.axvline(0, linewidth=0.8, color="0.45")
    if ylims is not None:
        ax.set_ylim(*ylims)
    ax.set_xlabel(r"Perturbation strength $\gamma$")
    ax.set_ylabel(r"Mean $\Delta$IVT [kg m$^{-1}$ s$^{-1}$]")
    if title:
        ax.set_title(title, loc="left")
    ax.grid(alpha=0.2)
    if show_legend:
        ax.legend(frameon=False)


def draw_global_trajectory_panel(ax, global_metrics, title=None, show_legend=True):
    gammas = sorted(global_metrics["gamma"].unique())
    nonzero = [g for g in gammas if not np.isclose(g, CONTROL_GAMMA)]
    max_abs_gamma = max(abs(g) for g in nonzero) if nonzero else 1.0
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(-max_abs_gamma, max_abs_gamma)

    # Draw perturbed trajectories first, then place the control on top so that
    # the counterfactual baseline is visually explicit.
    for gamma in nonzero:
        group = global_metrics[np.isclose(global_metrics["gamma"], gamma)].sort_values("lead_hours")
        ax.plot(
            group["lead_hours"], group["global_mean_ivt"],
            marker="o", markersize=3.2, linewidth=1.6,
            color=cmap(norm(gamma)), label=rf"$\gamma={gamma:g}$",
            alpha=0.9,
        )

    if CONTROL_GAMMA in gammas:
        group = global_metrics[np.isclose(global_metrics["gamma"], CONTROL_GAMMA)].sort_values("lead_hours")
        ax.plot(
            group["lead_hours"], group["global_mean_ivt"],
            color="black", linewidth=2.4, linestyle="--",
            label=rf"$\gamma=0$ (baseline)", zorder=10,
        )

    ax.set_xlabel("Forecast lead time [h]")
    ax.set_ylabel(r"Global mean IVT [kg m$^{-1}$ s$^{-1}$]")
    if title:
        ax.set_title(title, loc="left")
    ax.grid(alpha=0.2)
    if show_legend:
        ax.legend(frameon=False, ncol=2, fontsize=7)


def make_figure2(metrics, global_metrics):
    """Immediate AR-specific dose response + full absolute-IVT trajectory."""
    available_leads = np.sort(metrics["lead_hours"].unique())
    early_lead = float(available_leads[0])

    early_values = metrics[np.isclose(metrics["lead_hours"], early_lead)][
        ["delta_ivt_inside_mask_mean", "delta_ivt_outside_mask_mean"]
    ].to_numpy()
    ymax = np.nanmax(np.abs(early_values))
    margin = 0.08 * ymax if ymax > 0 else 1.0
    ylims = (-ymax - margin, ymax + margin)

    # Composite report figure: only the clean first-intervention ClimateNet
    # comparison, followed by the complete autoregressive IVT trajectories.
    fig, axes = plt.subplots(
        1, 2, figsize=(9.0, 3.7),
        gridspec_kw={"width_ratios": [1, 1.35]},
        constrained_layout=True,
    )
    draw_dose_panel(
        axes[0], metrics, early_lead, ylims,
        title=f"(a) Immediate AR response (+{format_lead_time(round(early_lead))})",
        show_legend=True,
    )
    draw_global_trajectory_panel(
        axes[1], global_metrics,
        title="(b) Global IVT evolution",
        show_legend=True,
    )

    stem = os.path.join(OUT_DIR, "figure2_ar_response")
    fig.savefig(stem + ".png", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", stem + ".png")

    # Standalone panels for direct LaTeX assembly.
    panel_fig, panel_ax = plt.subplots(figsize=(4.7, 3.5), constrained_layout=True)
    draw_dose_panel(
        panel_ax, metrics, early_lead, ylims,
        #title=f"Immediate AR response (+{format_lead_time(round(early_lead))})",
        show_legend=True,
    )
    save_panel(panel_fig, "figure2a_early_dose_response.png")

    panel_fig, panel_ax = plt.subplots(figsize=(5.4, 3.5), constrained_layout=True)
    draw_global_trajectory_panel(
        panel_ax, global_metrics,
        #title="Global IVT evolution",
        show_legend=True,
    )
    save_panel(panel_fig, "figure2b_global_ivt_trajectory.png")


# -----------------------------------------------------------------------------
# Reproducibility table
# -----------------------------------------------------------------------------
def save_mask_match_table(matches):
    rows = [{
        "time_index": m["time_index"],
        "lead_hours": m["lead_hours"],
        "forecast_valid_time": m["valid_time"],
        "climatenet_mask_time": m["mask_time"],
        "time_difference_hours": m["mask_time_diff_h"],
        "mask_file": m["mask_path"],
    } for m in matches]
    path = os.path.join(OUT_DIR, "climatenet_mask_matches.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Saved:", path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PANEL_DIR, exist_ok=True)
    file_table = discover_files(INPUT_DIR, CENTER_STR)

    if CONTROL_GAMMA not in file_table["gamma"].values:
        raise ValueError(f"No control gamma={CONTROL_GAMMA:g} found.")

    print("Loading IVT trajectories...")
    ivt_cache, valid_times, lead_hours = build_ivt_cache(file_table)

    print("Matching forecast steps to ClimateNet masks...")
    matches = collect_mask_matches(ivt_cache[CONTROL_GAMMA], valid_times, lead_hours)
    print(f"Found {len(matches)} distinct ClimateNet verification masks:")
    for m in matches:
        print(
            f"  +{m['lead_hours']:.0f} h | forecast {m['valid_time']} | "
            f"mask {m['mask_time']} | Δt={m['mask_time_diff_h']:.1f} h"
        )

    metrics = compute_mask_metrics(ivt_cache, matches)
    metrics_path = os.path.join(OUT_DIR, "ar_mask_dose_response_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print("Saved:", metrics_path)
    save_mask_match_table(matches)

    global_metrics = compute_global_ivt_trajectory(ivt_cache, lead_hours)
    global_metrics_path = os.path.join(OUT_DIR, "global_ivt_trajectory.csv")
    global_metrics.to_csv(global_metrics_path, index=False)
    print("Saved:", global_metrics_path)

    print("Making Figure 1...")
    make_figure1(ivt_cache, valid_times, lead_hours)

    print("Making Figure 2...")
    make_figure2(metrics, global_metrics)

    print("[DONE]")


if __name__ == "__main__":
    main()
