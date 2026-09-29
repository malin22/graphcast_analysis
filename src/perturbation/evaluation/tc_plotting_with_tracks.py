import os
from contextlib import closing

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.ndimage import label

from evaluation_helpers import (
    area_weighted_mean,
    discover_files,
    format_lead_time,
    gamma_colors,
    get_lat_name,
    get_lon_name,
    get_valid_time,
    load_mask_on_grid,
    load_prediction,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
WEATHER_FEATURE = "TC"
ACTIVATION_TYPE = "raw_activations"
CENTER_STR = "2021-03-10T18"
NODE_HIERARCHY_LEVEL = 6
CONTROL_GAMMA = 0.0
MAX_MASK_TIME_DIFFERENCE_HOURS = 3

# Spatial comparison: symmetric perturbations shown as two rows.
REPORT_GAMMAS = (-0.5, 0.5)

# Spatial transition through time. The nearest available model step is used.
MAP_LEAD_HOURS = (6, 24, 72, 120)

# Immediate-response lead time used for the dose-response figures.
DOSE_LEAD_HOURS = 6.0

# TC tracking / intensity settings.
TC_RADIUS_KM = 200.0
TRACK_SEARCH_RADIUS_KM = 300.0
TRACK_DISTANCE_WEIGHT_HPA = 2.0
EARTH_RADIUS_KM = 6371.0

# Local spatial figure radius around the unperturbed TC centre.
#LOCAL_MAP_RADIUS_KM = 700.0

# ClimateNet inside/outside comparison. If None, "outside" means the full
# forecast domain outside the selected TC mask, matching the AR script logic.
# Set e.g. 700.0 to compare the TC mask against a local environmental annulus.
MASK_OUTSIDE_RADIUS_KM = None


BASE_DIR = os.path.join(
    "results", "perturbation", WEATHER_FEATURE,
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}",
    ACTIVATION_TYPE,
    CENTER_STR,
)
INPUT_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(
    "plots", "perturbation", WEATHER_FEATURE,
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}",
    ACTIVATION_TYPE,
    CENTER_STR, "for_report"
)
os.makedirs(OUT_DIR, exist_ok=True)
MASK_DIR = (
    f"/share/prj-4d/graphcast_shared/data/ClimateNetLarge/"
    f"{WEATHER_FEATURE}_labels_cleaned"
)

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
# Basic fields and geometry
# -----------------------------------------------------------------------------
def get_mslp(ds):
    """Mean sea-level pressure in hPa."""
    mslp = ds["mean_sea_level_pressure"]
    if float(mslp.max(skipna=True)) > 2000:
        mslp = mslp / 100.0
    mslp.name = "mslp"
    return mslp


def compute_10m_wind(ds):
    """10 m wind speed in m s^-1."""
    u10 = ds["10m_u_component_of_wind"]
    v10 = ds["10m_v_component_of_wind"]
    wind10 = np.hypot(u10, v10)
    wind10.name = "wind10"
    return wind10


def open_forecast(path):
    """Open a GraphCast perturbation forecast and drop singleton batch."""
    ds = load_prediction(path, time_selection=None)
    if "batch" in ds.dims:
        ds = ds.isel(batch=0)
    return ds


def great_circle_distance(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    dlon = (lon2 - lon1 + 180.0) % 360.0 - 180.0
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlat = lat2r - lat1r
    dlon = np.deg2rad(dlon)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    )
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def distance_field(da, center_lat, center_lon):
    lat_name = get_lat_name(da)
    lon_name = get_lon_name(da)
    lats = da[lat_name].values
    lons = da[lon_name].values
    lon2d, lat2d = np.meshgrid(lons, lats)
    dist = great_circle_distance(center_lat, center_lon, lat2d, lon2d)
    return xr.DataArray(
        dist,
        coords={lat_name: da[lat_name], lon_name: da[lon_name]},
        dims=(lat_name, lon_name),
        name="distance_km",
    )


def radius_mask(da, center_lat, center_lon, radius_km):
    return distance_field(da, center_lat, center_lon) <= radius_km


def nearest_step_for_lead(ds, requested_lead_h):
    leads = np.array([
        pd.to_timedelta(ds.time.values[i]).total_seconds() / 3600.0
        for i in range(ds.sizes["time"])
    ])
    return int(np.argmin(np.abs(leads - requested_lead_h)))


def lead_hours_for_step(ds, t_idx):
    return float(pd.to_timedelta(ds.time.values[t_idx]).total_seconds() / 3600.0)


def get_gamma_path(file_table, gamma):
    hit = file_table[np.isclose(file_table["gamma"].astype(float), gamma)]
    if hit.empty:
        raise ValueError(
            f"Requested gamma={gamma:g} not found. Available: "
            f"{sorted(file_table['gamma'].unique())}"
        )
    return hit.iloc[0]["file"]


# -----------------------------------------------------------------------------
# TC identification and tracking
# -----------------------------------------------------------------------------
def find_min_mslp_center(mslp, mask=None):
    x = mslp.where(mask) if mask is not None else mslp
    idx = np.unravel_index(np.nanargmin(x.values), x.shape)
    lat_name = get_lat_name(mslp)
    lon_name = get_lon_name(mslp)
    return {
        "lat": float(mslp[lat_name].values[idx[0]]),
        "lon": float(mslp[lon_name].values[idx[1]]),
        "mslp": float(mslp.values[idx]),
    }


def find_tc_center_with_cost(
    mslp,
    prev_lat,
    prev_lon,
    search_radius_km=TRACK_SEARCH_RADIUS_KM,
    distance_weight_hpa=TRACK_DISTANCE_WEIGHT_HPA,
):
    """Track the local MSLP minimum while discouraging unrealistic jumps."""
    search_mask = radius_mask(mslp, prev_lat, prev_lon, search_radius_km)
    dist = distance_field(mslp, prev_lat, prev_lon)
    score = mslp + distance_weight_hpa * (dist / search_radius_km)
    score = score.where(search_mask)

    idx = np.unravel_index(np.nanargmin(score.values), score.shape)
    lat_name = get_lat_name(mslp)
    lon_name = get_lon_name(mslp)
    return {
        "lat": float(mslp[lat_name].values[idx[0]]),
        "lon": float(mslp[lon_name].values[idx[1]]),
        "mslp": float(mslp.values[idx]),
    }


def get_tc_components(tc_mask, min_pixels=5):
    labeled, n_components = label(tc_mask.values.astype(bool))
    components = []
    for component_id in range(1, n_components + 1):
        component = labeled == component_id
        if component.sum() < min_pixels:
            continue
        component_mask = xr.DataArray(
            component,
            coords=tc_mask.coords,
            dims=tc_mask.dims,
        )
        components.append({
            "tc_id": component_id,
            "mask": component_mask,
            "n_pixels": int(component.sum()),
        })
    return components


def select_mask_component_near_center(mask, center_lat, center_lon, min_pixels=1):
    """Select the connected ClimateNet TC component closest to a track centre."""
    components = get_tc_components(mask, min_pixels=min_pixels)
    if not components:
        return mask.astype(bool)

    dist = distance_field(mask, center_lat, center_lon)
    best = None
    best_dist = np.inf
    for comp in components:
        dmin = float(dist.where(comp["mask"]).min(skipna=True).values)
        if dmin < best_dist:
            best_dist = dmin
            best = comp["mask"]
    return best.astype(bool)


def tc_metrics_at_center(ds_step, center_lat, center_lon, radius_km=TC_RADIUS_KM):
    mslp = get_mslp(ds_step)
    wind10 = compute_10m_wind(ds_step)
    mask = radius_mask(mslp, center_lat, center_lon, radius_km)
    return {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "min_mslp_hpa": float(mslp.where(mask).min(skipna=True).values),
        "max_10m_wind": float(wind10.where(mask).max(skipna=True).values),
    }


def track_tc(ds, gamma, initial_center):
    records = []
    prev_lat = initial_center["lat"]
    prev_lon = initial_center["lon"]

    for t_idx in range(ds.sizes["time"]):
        ds_step = ds.isel(time=t_idx)
        mslp = get_mslp(ds_step)

        if t_idx == 0:
            center_lat = prev_lat
            center_lon = prev_lon
        else:
            center = find_tc_center_with_cost(mslp, prev_lat, prev_lon)
            center_lat = center["lat"]
            center_lon = center["lon"]

        metrics = tc_metrics_at_center(ds_step, center_lat, center_lon)
        metrics.update({
            "gamma": float(gamma),
            "time_index": t_idx,
            "lead_hours": lead_hours_for_step(ds, t_idx),
            "forecast_valid_time": str(get_valid_time(ds, t_idx, CENTER_STR)),
        })
        records.append(metrics)
        prev_lat, prev_lon = center_lat, center_lon

    return pd.DataFrame(records)


def build_tracks(file_table):
    """Track each ClimateNet-identified TC for every perturbation strength."""
    control_path = get_gamma_path(file_table, CONTROL_GAMMA)
    control_ds = open_forecast(control_path)
    try:
        valid_time_0 = get_valid_time(control_ds, 0, CENTER_STR)
        mslp0 = get_mslp(control_ds.isel(time=0))
        tc_mask, mask_path, mask_time, mask_diff_h = load_mask_on_grid(
            valid_time_0,
            mslp0,
            MASK_DIR,
            MAX_MASK_TIME_DIFFERENCE_HOURS,
        )
        components = get_tc_components(tc_mask, min_pixels=5)
        if not components:
            raise RuntimeError("No ClimateNet TC components found at initial time.")

        initial_centres = []
        for comp in components:
            center = find_min_mslp_center(mslp0, comp["mask"])
            initial_centres.append({
                "tc_id": comp["tc_id"],
                "n_pixels": comp["n_pixels"],
                **center,
            })
    finally:
        control_ds.close()

    all_tracks = []
    for init in initial_centres:
        for _, row in file_table.sort_values("gamma").iterrows():
            ds = open_forecast(row["file"])
            try:
                track = track_tc(ds, float(row["gamma"]), init)
            finally:
                ds.close()
            track["tc_id"] = init["tc_id"]
            track["initial_component_n_pixels"] = init["n_pixels"]
            all_tracks.append(track)

    tracks = pd.concat(all_tracks, ignore_index=True)

    control = tracks[np.isclose(tracks["gamma"], CONTROL_GAMMA)][[
        "tc_id", "lead_hours", "center_lat", "center_lon",
        "min_mslp_hpa", "max_10m_wind",
    ]].rename(columns={
        "center_lat": "control_center_lat",
        "center_lon": "control_center_lon",
        "min_mslp_hpa": "control_min_mslp_hpa",
        "max_10m_wind": "control_max_10m_wind",
    })

    tracks = tracks.merge(control, on=["tc_id", "lead_hours"], how="left")
    tracks["track_displacement_km"] = great_circle_distance(
        tracks["center_lat"], tracks["center_lon"],
        tracks["control_center_lat"], tracks["control_center_lon"],
    )
    tracks["delta_min_mslp_hpa"] = (
        tracks["min_mslp_hpa"] - tracks["control_min_mslp_hpa"]
    )
    tracks["delta_max_10m_wind"] = (
        tracks["max_10m_wind"] - tracks["control_max_10m_wind"]
    )

    metadata = {
        "initial_mask_file": mask_path,
        "initial_mask_time": str(mask_time),
        "initial_mask_time_diff_h": float(mask_diff_h),
    }
    return tracks, metadata


# -----------------------------------------------------------------------------
# ClimateNet mask response at +6 h
# -----------------------------------------------------------------------------
def compute_mask_localisation_metrics(file_table, tracks, tc_id, requested_lead_h):
    """Mean ΔMSLP and Δwind inside/outside the matched ClimateNet TC mask."""
    control_path = get_gamma_path(file_table, CONTROL_GAMMA)
    control_ds = open_forecast(control_path)
    try:
        idx = nearest_step_for_lead(control_ds, requested_lead_h)
        lead_h = lead_hours_for_step(control_ds, idx)
        valid_time = get_valid_time(control_ds, idx, CENTER_STR)
        control_step = control_ds.isel(time=idx)
        control_mslp = get_mslp(control_step).load()
        control_wind = compute_10m_wind(control_step).load()
    finally:
        control_ds.close()

    control_track = tracks[
        (tracks["tc_id"] == tc_id)
        & np.isclose(tracks["gamma"], CONTROL_GAMMA)
        & np.isclose(tracks["lead_hours"], lead_h)
    ]
    if control_track.empty:
        raise RuntimeError(f"No control track row at +{lead_h:g} h for TC {tc_id}.")
    center_lat = float(control_track.iloc[0]["center_lat"])
    center_lon = float(control_track.iloc[0]["center_lon"])

    mask, mask_path, mask_time, diff_h = load_mask_on_grid(
        valid_time,
        control_mslp,
        MASK_DIR,
        MAX_MASK_TIME_DIFFERENCE_HOURS,
    )
    tc_mask = select_mask_component_near_center(mask, center_lat, center_lon)

    if MASK_OUTSIDE_RADIUS_KM is None:
        outside_mask = ~tc_mask
    else:
        local = radius_mask(control_mslp, center_lat, center_lon, MASK_OUTSIDE_RADIUS_KM)
        outside_mask = local & (~tc_mask)

    records = []
    for _, row in file_table.sort_values("gamma").iterrows():
        gamma = float(row["gamma"])
        ds = open_forecast(row["file"])
        try:
            step = ds.isel(time=idx)
            mslp = get_mslp(step).load()
            wind = compute_10m_wind(step).load()
        finally:
            ds.close()

        dmslp = mslp - control_mslp
        dwind = wind - control_wind
        records.append({
            "tc_id": tc_id,
            "gamma": gamma,
            "lead_hours": lead_h,
            "forecast_valid_time": str(valid_time),
            "mask_time": str(mask_time),
            "mask_time_diff_h": float(diff_h),
            "mask_file": mask_path,
            "delta_mslp_inside_mean": area_weighted_mean(dmslp, tc_mask),
            "delta_mslp_outside_mean": area_weighted_mean(dmslp, outside_mask),
            "delta_wind_inside_mean": area_weighted_mean(dwind, tc_mask),
            "delta_wind_outside_mean": area_weighted_mean(dwind, outside_mask),
        })

    return pd.DataFrame(records).sort_values("gamma").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def save_panel(fig, filename):
    path = os.path.join(PANEL_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def gamma_plot_style(gammas):
    colors, _, _ = gamma_colors(gammas)
    colors[CONTROL_GAMMA] = "black"
    return colors


def subset_local(da, center_lat, center_lon, radius_km):
    """Mask a field to a circular local region while preserving lat/lon grid."""
    local = radius_mask(da, center_lat, center_lon, radius_km)
    return da.where(local)


def add_local_tc_mask_contour(ax, field, valid_time, center_lat, center_lon):
    try:
        mask, _, mask_time, diff_h = load_mask_on_grid(
            valid_time,
            field,
            MASK_DIR,
            MAX_MASK_TIME_DIFFERENCE_HOURS,
        )
    except Exception:
        return None

    mask = select_mask_component_near_center(mask, center_lat, center_lon)
    lat = get_lat_name(field)
    lon = get_lon_name(field)
    ax.contour(
        field[lon].values,
        field[lat].values,
        mask.values.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=0.9,
        transform=ccrs.PlateCarree(),
    )
    return mask_time, diff_h


# -----------------------------------------------------------------------------
# Figure 1: TC tracks across perturbation strengths
# -----------------------------------------------------------------------------
def make_figure1(file_table, tracks, tc_id):
    """Plot the tracked TC centre for every perturbation strength.

    The same initial ClimateNet-identified centre is used for every gamma, and
    each forecast is then tracked independently from its own MSLP field.
    """
    tc_tracks = tracks[tracks["tc_id"] == tc_id].copy()
    gammas = sorted(tc_tracks["gamma"].unique())
    colors = gamma_plot_style(gammas)

    # Geographic extent from all tracks, with a small padding.
    lat_min = float(tc_tracks["center_lat"].min())
    lat_max = float(tc_tracks["center_lat"].max())
    lon_min = float(tc_tracks["center_lon"].min())
    lon_max = float(tc_tracks["center_lon"].max())
    lat_pad = max(2.0, 0.15 * max(lat_max - lat_min, 1.0))
    lon_pad = max(2.0, 0.15 * max(lon_max - lon_min, 1.0))

    fig = plt.figure(figsize=(6.8, 5.6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(
        [lon_min - lon_pad, lon_max + lon_pad,
         lat_min - lat_pad, lat_max + lat_pad],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="50m", linewidth=0.55, color="0.35")
    ax.add_feature(cfeature.LAND, facecolor="0.94", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True,
        linewidth=0.35, color="0.65", alpha=0.5, linestyle=":",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    # Draw perturbed tracks first, then control on top for readability.
    plot_order = [g for g in gammas if not np.isclose(g, CONTROL_GAMMA)]
    if any(np.isclose(g, CONTROL_GAMMA) for g in gammas):
        plot_order.append(CONTROL_GAMMA)

    for gamma in plot_order:
        group = tc_tracks[np.isclose(tc_tracks["gamma"], gamma)].sort_values("lead_hours")
        is_control = np.isclose(gamma, CONTROL_GAMMA)
        ax.plot(
            group["center_lon"], group["center_lat"],
            color=colors[gamma],
            linewidth=2.4 if is_control else 1.6,
            linestyle="--" if is_control else "-",
            marker="o", markersize=3.2 if is_control else 2.6,
            transform=ccrs.PlateCarree(),
            label=rf"$\gamma={gamma:g}$",
            zorder=4 if is_control else 3,
        )

    # Common initial centre.
    first = tc_tracks.sort_values("lead_hours").iloc[0]
    ax.scatter(
        first["center_lon"], first["center_lat"],
        marker="*", s=85, facecolor="white", edgecolor="black",
        linewidth=0.8, transform=ccrs.PlateCarree(), zorder=6,
        label="Initial centre",
    )

    ax.set_title("Predicted tropical-cyclone tracks", pad=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(
        frameon=False, loc="upper left", ncol=2,
        handlelength=2.3, columnspacing=1.0,
    )

    stem = os.path.join(OUT_DIR, f"figure1_tc{tc_id}_tracks")
    fig.savefig(stem + ".png", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", stem + ".png")


# -----------------------------------------------------------------------------
# Figure 2: immediate TC dose response (+6 h)
# -----------------------------------------------------------------------------
def make_figure2(mask_metrics, tc_id):
    """Dose response at +6 h: mean response inside vs outside ClimateNet TC mask."""
    lead_h = float(mask_metrics["lead_hours"].iloc[0])
    gamma_ticks = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), constrained_layout=True)

    axes[0].plot(
        mask_metrics["gamma"], mask_metrics["delta_wind_inside_mean"],
        marker="o", linewidth=2.0, color="forestgreen", label="Inside TC mask",
    )
    axes[0].plot(
        mask_metrics["gamma"], mask_metrics["delta_wind_outside_mean"],
        marker="s", linewidth=1.7, linestyle="--", color="0.5", label="Outside TC mask",
    )
    axes[0].set_ylabel(r"Mean $\Delta$10 m wind [m s$^{-1}$]")
    axes[0].set_title(
        f"(a) Wind response (+{format_lead_time(round(lead_h))})", loc="left"
    )

    axes[1].plot(
        mask_metrics["gamma"], mask_metrics["delta_mslp_inside_mean"],
        marker="o", linewidth=2.0, color="forestgreen", label="Inside TC mask",
    )
    axes[1].plot(
        mask_metrics["gamma"], mask_metrics["delta_mslp_outside_mean"],
        marker="s", linewidth=1.7, linestyle="--", color="0.5", label="Outside TC mask",
    )
    axes[1].set_ylabel(r"Mean $\Delta$MSLP [hPa]")
    axes[1].set_title(
        f"(b) Pressure response (+{format_lead_time(round(lead_h))})", loc="left"
    )

    for ax in axes:
        ax.axhline(0, color="0.45", linewidth=0.8)
        ax.axvline(0, color="0.45", linewidth=0.8)
        ax.set_xticks(gamma_ticks, labels=["-1", "-0.5", "-0.2", "0", "0.2", "0.5", "1"])
        ax.set_xlabel(r"Perturbation strength $\gamma$")
        ax.grid(alpha=0.2)

    axes[0].legend(frameon=False)

    stem = os.path.join(OUT_DIR, f"figure2_tc{tc_id}_climatenet_dose_response")
    fig.savefig(stem + ".png", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", stem + ".png")


# -----------------------------------------------------------------------------
# Figure 3: persistence through forecast
# -----------------------------------------------------------------------------
def make_figure3(tracks, tc_id):
    tc_tracks = tracks[tracks["tc_id"] == tc_id]
    gammas = sorted(tc_tracks["gamma"].unique())
    colors = gamma_plot_style(gammas)
    nonzero = [g for g in gammas if not np.isclose(g, CONTROL_GAMMA)]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=False)

    for gamma in nonzero:
        group = tc_tracks[np.isclose(tc_tracks["gamma"], gamma)].sort_values("lead_hours")
        axes[0].plot(
            group["lead_hours"], group["delta_max_10m_wind"],
            marker="o", markersize=3.0, linewidth=1.7,
            color=colors[gamma], label=rf"$\gamma={gamma:g}$",
        )
        axes[1].plot(
            group["lead_hours"], group["delta_min_mslp_hpa"],
            marker="o", markersize=3.0, linewidth=1.7,
            color=colors[gamma], label=rf"$\gamma={gamma:g}$",
        )

    for ax in axes:
        ax.axhline(0, color="black", linewidth=1.1, linestyle="--")
        ax.set_xlabel("Forecast lead time [h]")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel(r"$\Delta V_{\max}$ [m s$^{-1}$]")
    axes[0].set_title("(a) Maximum wind response", loc="left", pad=8)
    axes[1].set_ylabel(r"$\Delta p_{\min}$ [hPa]")
    axes[1].set_title("(b) Minimum pressure response", loc="left", pad=8)

    # Shared legend above both panels, clear of the panel titles.
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 0.995), ncol=6,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.82, wspace=0.20)

    stem = os.path.join(OUT_DIR, f"figure3_tc{tc_id}_response_over_time")
    fig.savefig(stem + ".png", bbox_inches="tight")
    plt.close(fig)
    print("Saved:", stem + ".png")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    file_table = discover_files(INPUT_DIR, CENTER_STR)
    if not np.any(np.isclose(file_table["gamma"].astype(float), CONTROL_GAMMA)):
        raise ValueError(f"No control gamma={CONTROL_GAMMA:g} found.")

    for gamma in REPORT_GAMMAS:
        if not np.any(np.isclose(file_table["gamma"].astype(float), gamma)):
            raise ValueError(
                f"Main report figure requires gamma={gamma:g}. Available: "
                f"{sorted(file_table['gamma'].astype(float).unique())}"
            )

    print("Tracking ClimateNet-identified tropical cyclones...")
    tracks, metadata = build_tracks(file_table)
    tracks_path = os.path.join(OUT_DIR, "tracked_tc_metrics_by_gamma.csv")
    tracks.to_csv(tracks_path, index=False)
    print("Saved:", tracks_path)

    pd.DataFrame([metadata]).to_csv(
        os.path.join(OUT_DIR, "tracking_metadata.csv"), index=False
    )

    tc_ids = sorted(tracks["tc_id"].unique())
    print(f"Found TC components: {tc_ids}")

    for tc_id in tc_ids:
        print(f"\n[TC {tc_id}] ClimateNet mask localisation at +{DOSE_LEAD_HOURS:g} h...")
        mask_metrics = compute_mask_localisation_metrics(
            file_table, tracks, tc_id, DOSE_LEAD_HOURS,
        )
        mask_path = os.path.join(OUT_DIR, f"tc{tc_id}_climatenet_mask_response.csv")
        mask_metrics.to_csv(mask_path, index=False)
        print("Saved:", mask_path)

        print(f"[TC {tc_id}] Making Figure 1: TC tracks...")
        make_figure1(file_table, tracks, tc_id)

        print(f"[TC {tc_id}] Making Figure 2: immediate dose response...")
        make_figure2(mask_metrics, tc_id)

        print(f"[TC {tc_id}] Making Figure 3: persistence through forecast...")
        make_figure3(tracks, tc_id)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
