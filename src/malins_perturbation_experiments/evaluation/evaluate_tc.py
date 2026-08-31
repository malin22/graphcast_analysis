"""Tropical-cyclone tracking and intensity evaluation."""

import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import label

from evaluation_helpers import (
    area_weighted_mean, discover_files, format_lead_time, gamma_colors,
    get_lat_name, get_lon_name, get_valid_time, load_era5_at_time,
    load_mask_on_grid, load_prediction, max_value,
)

WEATHER_FEATURE = "TC"
THRESHOLD = 0.8
CENTER_STR = "2021-02-12T18"
NODE_HIERARCHY_LEVEL = 6
CONTROL_GAMMA = 0.0
MAX_MASK_TIME_DIFFERENCE_HOURS = 3
TC_RADIUS_KM = 200.0
TRACK_SEARCH_RADIUS_KM = 300.0
EARTH_RADIUS_KM = 6371.0
TP_VAR = "total_precipitation_6hr"
ERA5_DAILY_DIR = "/share/prj-4d/graphcast_shared/data/era5_daily_nc"

BASE_DIR = os.path.join(
    "results", "perturbation", WEATHER_FEATURE,
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}", ACTIVATION_TYPE, CENTER_STR,
)
INPUT_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "evaluation")
MASK_DIR = f"/share/prj-4d/graphcast_shared/data/ClimateNetLarge/{WEATHER_FEATURE}_labels_cleaned"

def get_first_existing_variable(ds, names):
    for name in names:
        if name in ds:
            return ds[name]

    raise KeyError(
        f"None of these variables were found: {names}. "
        f"Available variables: {list(ds.data_vars)}"
    )

def get_era5_mslp(ds):
    mslp = get_first_existing_variable(
        ds,
        [
            "mean_sea_level_pressure",
            "msl",
        ],
    )

    # Convert Pa to hPa.
    if float(mslp.max(skipna=True)) > 2000:
        mslp = mslp / 100.0

    mslp.name = "mslp"
    return mslp

def compute_era5_10m_wind(ds):
    u10 = get_first_existing_variable(
        ds,
        [
            "10m_u_component_of_wind",
            "u10",
        ],
    )

    v10 = get_first_existing_variable(
        ds,
        [
            "10m_v_component_of_wind",
            "v10",
        ],
    )

    wind10 = np.hypot(u10, v10)
    wind10.name = "wind10"
    return wind10

def great_circle_distance(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    # Wrap longitude difference to [-180, 180]
    dlon = (lon2 - lon1 + 180.0) % 360.0 - 180.0

    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    dlat = lat2 - lat1
    dlon = np.deg2rad(dlon)

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    # Fix both floating point overshoot and NaNs
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
    a = np.clip(a, 0.0, 1.0)

    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

def radius_mask(da, center_lat, center_lon, radius_km):
    lat_name = get_lat_name(da)
    lon_name = get_lon_name(da)

    lats = da[lat_name].values
    lons = da[lon_name].values

    lon2d, lat2d = np.meshgrid(lons, lats)

    dist = great_circle_distance(center_lat, center_lon, lat2d, lon2d)

    return xr.DataArray(
        dist <= radius_km,
        coords={lat_name: da[lat_name], lon_name: da[lon_name]},
        dims=(lat_name, lon_name),
    )

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

def compute_10m_wind(ds):
    u = ds["10m_u_component_of_wind"]
    v = ds["10m_v_component_of_wind"]
    wind = np.hypot(u, v)
    wind.name = "wind10"
    return wind

def get_mslp(ds):
    mslp = ds["mean_sea_level_pressure"]

    # Pa -> hPa
    if float(mslp.max()) > 2000:
        mslp = mslp / 100.0

    mslp.name = "mslp"
    return mslp

def tc_metrics_at_center(ds_step, center_lat, center_lon, radius_km=300.0):
    mslp = get_mslp(ds_step)
    wind10 = compute_10m_wind(ds_step)

    mask = radius_mask(mslp, center_lat, center_lon, radius_km)

    rec = {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "min_mslp_hpa": float(mslp.where(mask).min(skipna=True).values),
        "max_10m_wind": float(wind10.where(mask).max(skipna=True).values),
    }

    if TP_VAR in ds_step:
        tp = ds_step[TP_VAR]
        rec["mean_precip"] = area_weighted_mean(tp, mask)
        rec["max_precip"] = max_value(tp, mask)

    return rec

def find_tc_center_with_cost(
    mslp,
    prev_lat,
    prev_lon,
    search_radius_km=600.0,
    distance_weight_hpa=2.0,
):
    """
    Find TC center by minimizing:
        score = MSLP + distance_weight_hpa * (distance / search_radius_km)

    distance_weight_hpa controls how much you penalize jumping away
    from the previous center.

    Example:
        distance_weight_hpa = 2.0 means a point at the edge of the
        search radius pays a +2 hPa penalty.
    """
    search_mask = radius_mask(
        mslp,
        prev_lat,
        prev_lon,
        search_radius_km,
    )

    lat_name = get_lat_name(mslp)
    lon_name = get_lon_name(mslp)

    lats = mslp[lat_name].values
    lons = mslp[lon_name].values
    lon2d, lat2d = np.meshgrid(lons, lats)

    dist_km = great_circle_distance(
        prev_lat,
        prev_lon,
        lat2d,
        lon2d,
    )

    dist_da = xr.DataArray(
        dist_km,
        coords={lat_name: mslp[lat_name], lon_name: mslp[lon_name]},
        dims=(lat_name, lon_name),
    )

    score = mslp + distance_weight_hpa * (dist_da / search_radius_km)
    score = score.where(search_mask)

    idx = np.unravel_index(
        np.nanargmin(score.values),
        score.shape,
    )

    return {
        "lat": float(mslp[lat_name].values[idx[0]]),
        "lon": float(mslp[lon_name].values[idx[1]]),
        "mslp": float(mslp.values[idx]),
        "distance_from_prev_km": float(dist_da.values[idx]),
        "score": float(score.values[idx]),
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

def era5_tc_metrics_at_center(
    era5_step,
    center_lat,
    center_lon,
    radius_km=TC_RADIUS_KM,
):
    mslp = get_era5_mslp(era5_step)
    wind10 = compute_era5_10m_wind(era5_step)

    mask = radius_mask(
        mslp,
        center_lat,
        center_lon,
        radius_km,
    )

    return {
        "center_lat": center_lat,
        "center_lon": center_lon,
        "min_mslp_hpa": float(
            mslp.where(mask).min(skipna=True).values
        ),
        "max_10m_wind": float(
            wind10.where(mask).max(skipna=True).values
        ),
    }

def extract_tracked_tc_from_era5(
    forecast_ds,
    initial_center,
):
    """
    Track the TC independently in ERA5 at every GraphCast forecast
    valid time.

    forecast_ds is used only for its lead times and valid times.
    """
    records = []

    prev_lat = initial_center["lat"]
    prev_lon = initial_center["lon"]

    for t_idx in range(forecast_ds.sizes["time"]):
        valid_time = get_valid_time(forecast_ds, t_idx, CENTER_STR)

        era5_step, era5_file = load_era5_at_time(valid_time, ERA5_DAILY_DIR)
        era5_mslp = get_era5_mslp(era5_step)

        center = find_tc_center_with_cost(
            era5_mslp,
            prev_lat=prev_lat,
            prev_lon=prev_lon,
            search_radius_km=TRACK_SEARCH_RADIUS_KM,
            distance_weight_hpa=2.0,
        )

        center_lat = center["lat"]
        center_lon = center["lon"]

        metrics = era5_tc_metrics_at_center(
            era5_step,
            center_lat,
            center_lon,
            radius_km=TC_RADIUS_KM,
        )

        lead_h = (
            pd.to_timedelta(
                forecast_ds.time.values[t_idx]
            ).total_seconds()
            / 3600.0
        )

        metrics.update({
            "source": "ERA5",
            "lead_hours": lead_h,
            "lead_label": format_lead_time(lead_h),
            "forecast_valid_time": str(valid_time),
            "era5_time": str(valid_time),
            "era5_file": era5_file,
        })

        records.append(metrics)

        prev_lat = center_lat
        prev_lon = center_lon

    return pd.DataFrame(records)

def extract_tracked_tc_for_gamma(ds, gamma, initial_center):
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
            #search_mask = radius_mask(
            #    mslp,
            #    prev_lat,
            #    prev_lon,
            #    TRACK_SEARCH_RADIUS_KM,
            #)

            #center = find_min_mslp_center(mslp, mask=search_mask)
            
            center = find_tc_center_with_cost(
                mslp,
                prev_lat=prev_lat,
                prev_lon=prev_lon,
                search_radius_km=TRACK_SEARCH_RADIUS_KM,
                distance_weight_hpa=2.0,
            )

            
            center_lat = center["lat"]
            center_lon = center["lon"]

        metrics = tc_metrics_at_center(
            ds_step,
            center_lat,
            center_lon,
            radius_km=TC_RADIUS_KM,
        )

        lead_h = pd.to_timedelta(ds.time.values[t_idx]).total_seconds() / 3600.0

        metrics.update({
            "gamma": gamma,
            "lead_hours": lead_h,
            "lead_label": format_lead_time(lead_h),
            "forecast_valid_time": str(get_valid_time(ds, t_idx, CENTER_STR)),
        })

        records.append(metrics)

        prev_lat = center_lat
        prev_lon = center_lon

    return pd.DataFrame(records)

def evaluate_tc_tracks():
    out_dir = os.path.join(OUT_DIR, "tc_tracks")
    os.makedirs(out_dir, exist_ok=True)

    file_table = discover_files(INPUT_DIR, CENTER_STR)

    if CONTROL_GAMMA not in file_table["gamma"].values:
        raise ValueError(f"No control gamma={CONTROL_GAMMA} found.")

    control_file = file_table[file_table["gamma"] == CONTROL_GAMMA]["file"].iloc[0]
    control_ds = load_prediction(control_file, time_selection=None)

    valid_time_0 = get_valid_time(control_ds, 0, CENTER_STR)
    control_step0 = control_ds.isel(time=0)
    mslp0 = get_mslp(control_step0)

    # Load TC mask at the initial forecast time
    tc_mask, mask_path, mask_time, mask_diff_h = load_mask_on_grid(
        valid_time_0, mslp0, MASK_DIR, MAX_MASK_TIME_DIFFERENCE_HOURS
    )

    # Split initial TC mask into separate connected objects
    tc_components = get_tc_components(tc_mask, min_pixels=5)

    print(f"Initial mask: {mask_path}")
    print(f"Mask time: {mask_time}, diff={mask_diff_h:.1f} h")
    print(f"Found {len(tc_components)} TC components")

    all_tracks = []
    all_era5_tracks = []

    for comp in tc_components:
        tc_id = comp["tc_id"]
        component_mask = comp["mask"]

        initial_center = find_min_mslp_center(
            mslp0,
            mask=component_mask,
        )

        era5_track = extract_tracked_tc_from_era5(
            forecast_ds=control_ds,
            initial_center=initial_center,
        )

        era5_track["tc_id"] = tc_id
        era5_track["component_n_pixels"] = comp["n_pixels"]
        era5_track["initial_center_lat"] = initial_center["lat"]
        era5_track["initial_center_lon"] = initial_center["lon"]
        era5_track["mask_file"] = mask_path
        era5_track["mask_time"] = str(mask_time)
        era5_track["mask_time_diff_h"] = mask_diff_h

        all_era5_tracks.append(era5_track)

        print(
            f"TC {tc_id}: pixels={comp['n_pixels']}, "
            f"initial center=({initial_center['lat']:.2f}, "
            f"{initial_center['lon']:.2f}), "
            f"MSLP={initial_center['mslp']:.2f}"
        )

        for _, row in file_table.sort_values("gamma").iterrows():
            ds = load_prediction(row["file"], time_selection=None)

            track = extract_tracked_tc_for_gamma(
                ds,
                gamma=row["gamma"],
                initial_center=initial_center,
            )

            track["tc_id"] = tc_id
            track["component_n_pixels"] = comp["n_pixels"]
            track["initial_center_lat"] = initial_center["lat"]
            track["initial_center_lon"] = initial_center["lon"]
            track["initial_center_mslp_hpa"] = initial_center["mslp"]
            track["mask_file"] = mask_path
            track["mask_time"] = str(mask_time)
            track["mask_time_diff_h"] = mask_diff_h
            track["file"] = row["file"]

            all_tracks.append(track)

    if not all_tracks:
        raise ValueError("No TC components found in the initial mask.")

    tracks = pd.concat(all_tracks, ignore_index=True)

    era5_tracks = pd.concat(all_era5_tracks, ignore_index=True)

    era5_tracks_path = os.path.join(
        out_dir,
        "tracked_tc_metrics_era5.csv",
    )
    era5_tracks.to_csv(era5_tracks_path, index=False)
    print("Saved:", era5_tracks_path)

    # Compare every gamma to its own TC's control trajectory
    control = tracks[tracks["gamma"] == CONTROL_GAMMA][
        [
            "tc_id",
            "lead_hours",
            "center_lat",
            "center_lon",
            "min_mslp_hpa",
            "max_10m_wind",
        ]
    ].rename(columns={
        "center_lat": "control_center_lat",
        "center_lon": "control_center_lon",
        "min_mslp_hpa": "control_min_mslp_hpa",
        "max_10m_wind": "control_max_10m_wind",
    })

    tracks = tracks.merge(
        control,
        on=["tc_id", "lead_hours"],
        how="left",
    )

    tracks["track_error_km"] = great_circle_distance(
        tracks["center_lat"],
        tracks["center_lon"],
        tracks["control_center_lat"],
        tracks["control_center_lon"],
    )

    tracks["delta_min_mslp_hpa"] = (
        tracks["min_mslp_hpa"] - tracks["control_min_mslp_hpa"]
    )

    tracks["delta_max_10m_wind"] = (
        tracks["max_10m_wind"] - tracks["control_max_10m_wind"]
    )

    tracks_path = os.path.join(out_dir, "tracked_tc_metrics_by_gamma.csv")
    tracks.to_csv(tracks_path, index=False)
    print("Saved:", tracks_path)

    # Optional: one summary row per TC and gamma
    summary = (
        tracks.groupby(["tc_id", "gamma"])
        .agg(
            n_steps=("lead_hours", "count"),
            max_track_error_km=("track_error_km", "max"),
            mean_track_error_km=("track_error_km", "mean"),
            min_mslp_hpa=("min_mslp_hpa", "min"),
            max_10m_wind=("max_10m_wind", "max"),
            mean_delta_min_mslp_hpa=("delta_min_mslp_hpa", "mean"),
            mean_delta_max_10m_wind=("delta_max_10m_wind", "mean"),
            max_delta_max_10m_wind=("delta_max_10m_wind", "max"),
            min_delta_min_mslp_hpa=("delta_min_mslp_hpa", "min"),
        )
        .reset_index()
    )

    summary_path = os.path.join(out_dir, "tracked_tc_summary_by_gamma.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    # Existing plot functions can still work, but now include all tc_id values.
    # Better: update them later to either facet by tc_id or save one plot per tc_id.
    plot_tc_tracks(tracks, out_dir, era5_tracks=era5_tracks,)
    plot_tc_intensity_from_tracks(tracks, out_dir, era5_tracks=era5_tracks,)
    #plot_tc_track_error(tracks, out_dir)

    return tracks

def plot_tc_intensity_from_tracks(
    tracks,
    out_dir,
    era5_tracks=None,
):
    gammas = sorted(tracks["gamma"].unique())
    colors, cmap, norm = gamma_colors(gammas)

    for tc_id, tc_tracks in tracks.groupby("tc_id"):

        if era5_tracks is not None and not era5_tracks.empty:
            tc_era5 = (
                era5_tracks[
                    era5_tracks["tc_id"] == tc_id
                ]
                .sort_values("lead_hours")
            )
        else:
            tc_era5 = pd.DataFrame()

        # =====================
        # WIND PLOT
        # =====================

        plt.figure(figsize=(8, 5))

        for gamma, group in tc_tracks.groupby("gamma"):
            group = group.sort_values("lead_hours")

            plt.plot(
                group["lead_hours"],
                group["max_10m_wind"],
                marker="o",
                linewidth=2,
                color=colors[gamma],
                label=f"γ={gamma:g}",
            )

        # ERA5 on the same axes
        if not tc_era5.empty:
            plt.plot(
                tc_era5["lead_hours"],
                tc_era5["max_10m_wind"],
                color="black",
                linestyle="--",
                marker="o",
                markersize=4,
                linewidth=3,
                label="ERA5",
                zorder=10,
            )

        plt.xlabel("Forecast lead time [h]")
        plt.ylabel("Max 10 m wind within TC radius [m/s]")
        plt.title(
            f"Tracked TC wind intensity, TC {tc_id} "
            f"({CENTER_STR})"
        )
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        path = os.path.join(
            out_dir,
            f"tracked_tc_{tc_id}_max_wind_by_gamma.png",
        )
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved:", path)

        # =====================
        # PRESSURE PLOT
        # =====================

        plt.figure(figsize=(8, 5))

        for gamma, group in tc_tracks.groupby("gamma"):
            group = group.sort_values("lead_hours")

            plt.plot(
                group["lead_hours"],
                group["min_mslp_hpa"],
                marker="o",
                linewidth=2,
                color=colors[gamma],
                label=f"γ={gamma:g}",
            )

        # ERA5 on the same axes
        if not tc_era5.empty:
            plt.plot(
                tc_era5["lead_hours"],
                tc_era5["min_mslp_hpa"],
                color="black",
                linestyle="--",
                marker="o",
                markersize=4,
                linewidth=3,
                label="ERA5",
                zorder=10,
            )

        plt.xlabel("Forecast lead time [h]")
        plt.ylabel("Min MSLP within TC radius [hPa]")
        plt.title(
            f"Tracked TC pressure intensity, TC {tc_id} "
            f"({CENTER_STR})"
        )
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        path = os.path.join(
            out_dir,
            f"tracked_tc_{tc_id}_min_mslp_by_gamma.png",
        )
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved:", path)

def plot_tc_tracks(
    tracks,
    out_dir,
    era5_tracks=None,
):
    gammas = sorted(tracks["gamma"].unique())
    colors, cmap, norm = gamma_colors(gammas)

    for tc_id, tc_tracks in tracks.groupby("tc_id"):

        plt.figure(figsize=(7, 6))

        for gamma, group in tc_tracks.groupby("gamma"):
            group = group.sort_values("lead_hours")

            plt.plot(
                group["center_lon"],
                group["center_lat"],
                marker="o",
                markersize=4,
                linewidth=2,
                color=colors[gamma],
                label=f"γ={gamma:g}",
            )

        if era5_tracks is not None and not era5_tracks.empty:
            tc_era5 = (
                era5_tracks[
                    era5_tracks["tc_id"] == tc_id
                ]
                .sort_values("lead_hours")
            )

            # ERA5 on the same axes as all gamma trajectories
            if not tc_era5.empty:
                plt.plot(
                    tc_era5["center_lon"],
                    tc_era5["center_lat"],
                    marker="o",
                    markersize=5,
                    color="black",
                    linestyle="--",
                    linewidth=3,
                    label="ERA5",
                    zorder=10,
                )

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(
            f"Tracked TC path, TC {tc_id} ({CENTER_STR})"
        )
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        path = os.path.join(
            out_dir,
            f"tc_{tc_id}_tracks_by_gamma.png",
        )
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print("Saved:", path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("\n[MAKING TC INTENSITY TRAJECTORY PLOTS]\n")
    evaluate_tc_tracks()
    print("[DONE]")


if __name__ == "__main__":
    main()
