import os
import re
from glob import glob
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

from graphcast import icosahedral_mesh


# =============================
# Config
# =============================

PC_SCORES_PATH = "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/pc_scores_2021_per_timestep.npy"
TIMESTEP_FILES_TXT = "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/pc_scores_2021_per_timestep_files.txt"

ERA5_DAILY_DIR = "/share/prj-4d/graphcast_shared/data/era5_daily_nc"

OUT_DIR = "plots/malins_experiments/2021_pc_correlations_per_timestep"
os.makedirs(OUT_DIR, exist_ok=True)

N_PCS = 50
SPLITS = 6

# Weather variables to correlate spatially with PC maps.
# Adapt names/level coordinate names if your ERA5 files differ.
WEATHER_TARGETS = [
    {"name": "z500", "var": "geopotential", "level": 500},
    {"name": "t850", "var": "temperature", "level": 850},
    {"name": "q850", "var": "specific_humidity", "level": 850},
    {"name": "u850", "var": "u_component_of_wind", "level": 850},
    {"name": "v850", "var": "v_component_of_wind", "level": 850},
    {"name": "msl", "var": "mean_sea_level_pressure", "level": None},
    {"name": "2t", "var": "2m_temperature", "level": None},
]


# =============================
# Helpers
# =============================

def vertices_to_latlon(vertices):
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    lon = lon % 360
    return lat, lon


def get_mesh_latlon(splits=6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=splits
    )
    vertices = meshes[splits].vertices
    return vertices_to_latlon(vertices)


def corr_1d(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan

    a = a[mask]
    b = b[mask]

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan

    return np.corrcoef(a, b)[0, 1]


def parse_timestamp_from_path(path):
    """
    Tries to parse timestamps like:
    2021-01-01
    2021-01-01_06
    20210101_06
    """
    fname = os.path.basename(path)

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})[_T-]?(\d{2})?", fname)
    if m:
        year, month, day, hour = m.groups()
        hour = int(hour) if hour is not None else 0
        return pd.Timestamp(int(year), int(month), int(day), hour)

    m = re.search(r"(\d{4})(\d{2})(\d{2})[_T-]?(\d{2})?", fname)
    if m:
        year, month, day, hour = m.groups()
        hour = int(hour) if hour is not None else 0
        return pd.Timestamp(int(year), int(month), int(day), hour)

    raise ValueError(f"Could not parse timestamp from filename: {fname}")


def load_timestamps(files_txt):
    with open(files_txt, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    timestamps = [parse_timestamp_from_path(p) for p in files]

    ##delete me:
    #timestamps = timestamps[:10]
    #T = len(timestamps)
    #---
    return files, pd.to_datetime(timestamps)


@lru_cache(maxsize=16)
def open_era5_day(date_str):
    path = os.path.join(ERA5_DAILY_DIR, f"era5_{date_str}.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERA5 file not found: {path}")
    return xr.open_dataset(path)


def select_level_if_needed(da, level):
    if level is None:
        return da

    for coord in ["level", "pressure_level", "isobaricInhPa"]:
        if coord in da.coords:
            return da.sel({coord: level}, method="nearest")

    raise ValueError(
        f"Could not find pressure-level coordinate for {da.name}. "
        f"Available coords: {list(da.coords)}"
    )


def sample_era5_variable_at_mesh(timestamp, target, node_lat, node_lon):
    date_str = pd.Timestamp(timestamp).strftime("%Y-%m-%d")
    ds = open_era5_day(date_str)

    var = target["var"]
    if var not in ds:
        raise KeyError(f"Variable {var} not found in {date_str}. Available: {list(ds.data_vars)}")

    da = ds[var]
    da = select_level_if_needed(da, target["level"])

    if "time" in da.dims or "time" in da.coords:
        da = da.sel(time=np.datetime64(timestamp), method="nearest")

    sampled = da.interp(
        lat=node_lat,
        lon=node_lon,
        method="nearest",
    )

    return sampled.values.astype(np.float32)


def make_static_geography_features(example_era5_file, lat, lon):
    ds = xr.open_dataset(example_era5_file)

    node_lat = xr.DataArray(lat, dims="node")
    node_lon = xr.DataArray(lon, dims="node")

    features = {
        "latitude": lat.astype(np.float32),
        "sin_longitude": np.sin(np.deg2rad(lon)).astype(np.float32),
        "cos_longitude": np.cos(np.deg2rad(lon)).astype(np.float32),
    }

    if "land_sea_mask" in ds:
        lsm = ds["land_sea_mask"].interp(lat=node_lat, lon=node_lon, method="nearest").values
        features["fractional_lsm"] = lsm.astype(np.float32)
        features["binary_land"] = (lsm > 0.5).astype(np.float32)

    if "geopotential_at_surface" in ds:
        z_surf = ds["geopotential_at_surface"].interp(lat=node_lat, lon=node_lon, method="nearest").values
        features["elevation"] = (z_surf / 9.80665).astype(np.float32)

    return features


def seasonal_features(timestamps):
    doy = timestamps.dayofyear.to_numpy()
    hour = timestamps.hour.to_numpy()

    year_angle = 2 * np.pi * (doy - 1) / 365.25
    day_angle = 2 * np.pi * hour / 24.0

    return pd.DataFrame({
        "timestamp": timestamps,
        "sin_year": np.sin(year_angle),
        "cos_year": np.cos(year_angle),
        "sin_day": np.sin(day_angle),
        "cos_day": np.cos(day_angle),
        "month": timestamps.month,
        "dayofyear": doy,
        "hour": hour,
    })


# =============================
# Main analysis
# =============================

def main():
    pc_scores = np.load(PC_SCORES_PATH, mmap_mode="r")  # [time, nodes, pcs]
    #delete me:
    #pc_scores = pc_scores[:10]
    #----
    T, N, K = pc_scores.shape
    K = min(K, N_PCS)

    print("Loaded PC scores:", pc_scores.shape)

    timestep_files, timestamps = load_timestamps(TIMESTEP_FILES_TXT)

    if len(timestamps) != T:
        raise ValueError(f"Timestamp count {len(timestamps)} does not match PC score time dimension {T}")

    lat, lon = get_mesh_latlon(SPLITS)

    if len(lat) != N:
        raise ValueError(f"Mesh node count {len(lat)} does not match PC node dimension {N}")

    node_lat = xr.DataArray(lat, dims="node")
    node_lon = xr.DataArray(lon, dims="node")

    example_era5_file = os.path.join(
        ERA5_DAILY_DIR,
        f"era5_{pd.Timestamp(timestamps[0]).strftime('%Y-%m-%d')}.nc"
    )

    geography = make_static_geography_features(example_era5_file, lat, lon)

    # -----------------------------
    # 1. Spatial correlations
    # -----------------------------
    spatial_rows = []

    for t_idx, timestamp in enumerate(timestamps):
        print(f"[{t_idx + 1}/{T}] spatial correlations for {timestamp}")

        pc_t = np.asarray(pc_scores[t_idx, :, :K])  # [nodes, pcs]

        node_features = {}

        # Static geography
        node_features.update(geography)

        # Node-varying seasonal feature: local solar time
        local_hour = (timestamp.hour + lon / 15.0) % 24
        local_angle = 2 * np.pi * local_hour / 24.0
        node_features["sin_local_time"] = np.sin(local_angle).astype(np.float32)
        node_features["cos_local_time"] = np.cos(local_angle).astype(np.float32)

        # Weather fields
        for target in WEATHER_TARGETS:
            try:
                node_features[target["name"]] = sample_era5_variable_at_mesh(
                    timestamp,
                    target,
                    node_lat,
                    node_lon,
                )
            except Exception as e:
                print(f"WARNING: skipping weather target {target['name']} at {timestamp}: {e}")

        for pc_idx in range(K):
            pc = pc_t[:, pc_idx]

            for feature_name, feature_values in node_features.items():
                r = corr_1d(pc, feature_values)

                spatial_rows.append({
                    "timestamp": timestamp,
                    "PC": pc_idx + 1,
                    "feature": feature_name,
                    "correlation": r,
                    "abs_correlation": abs(r) if np.isfinite(r) else np.nan,
                })

    spatial_df = pd.DataFrame(spatial_rows)

    spatial_csv = os.path.join(OUT_DIR, "pc_spatial_correlations_per_timestep.csv")
    spatial_df.to_csv(spatial_csv, index=False)

    spatial_summary = (
        spatial_df
        .groupby(["PC", "feature"], as_index=False)
        .agg(
            mean_corr=("correlation", "mean"),
            mean_abs_corr=("abs_correlation", "mean"),
            max_abs_corr=("abs_correlation", "max"),
        )
        .sort_values("mean_abs_corr", ascending=False)
    )

    spatial_summary_csv = os.path.join(OUT_DIR, "pc_spatial_correlation_summary.csv")
    spatial_summary.to_csv(spatial_summary_csv, index=False)

    # -----------------------------
    # 2. Temporal correlations
    # -----------------------------
    # Seasonal variables are scalar per timestep, so we correlate them with
    # PC time series summaries, not with node maps within a timestep.
    season_df = seasonal_features(timestamps)

    temporal_rows = []

    for pc_idx in range(K):
        pc_all = np.asarray(pc_scores[:, :, pc_idx])  # [time, nodes]

        pc_timeseries = {
            "global_mean_score": np.nanmean(pc_all, axis=1),
            "global_absmean_score": np.nanmean(np.abs(pc_all), axis=1),
            "global_std_score": np.nanstd(pc_all, axis=1),
        }

        for pc_summary_name, y in pc_timeseries.items():
            for seasonal_name in [
                "sin_year",
                "cos_year",
                "sin_day",
                "cos_day",
                "month",
                "dayofyear",
                "hour",
            ]:
                r = corr_1d(y, season_df[seasonal_name].to_numpy())

                temporal_rows.append({
                    "PC": pc_idx + 1,
                    "pc_summary": pc_summary_name,
                    "seasonal_feature": seasonal_name,
                    "correlation": r,
                    "abs_correlation": abs(r) if np.isfinite(r) else np.nan,
                })

    temporal_df = pd.DataFrame(temporal_rows)

    temporal_csv = os.path.join(OUT_DIR, "pc_temporal_seasonal_correlations.csv")
    temporal_df.to_csv(temporal_csv, index=False)

    # -----------------------------
    # 3. Top hits
    # -----------------------------
    top_spatial = (
        spatial_summary
        .sort_values("mean_abs_corr", ascending=False)
        .head(100)
    )

    top_temporal = (
        temporal_df
        .sort_values("abs_correlation", ascending=False)
        .head(100)
    )

    top_spatial.to_csv(os.path.join(OUT_DIR, "top100_spatial_pc_feature_correlations.csv"), index=False)
    top_temporal.to_csv(os.path.join(OUT_DIR, "top100_temporal_pc_season_correlations.csv"), index=False)

    print("\nDone.")
    print(f"Saved spatial correlations: {spatial_csv}")
    print(f"Saved spatial summary: {spatial_summary_csv}")
    print(f"Saved temporal seasonal correlations: {temporal_csv}")


if __name__ == "__main__":
    main()