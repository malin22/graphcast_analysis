#!/usr/bin/env python3
import gc
import json
import os
from pathlib import Path

import numpy as np
import xarray as xr

INPUT_DIR = Path("/share/prj-4d/graphcast_shared/data/era5_daily_nc")
OUTPUT_DIR = Path("/share/prj-4d/graphcast_shared/data/era5_daily_course_grid")
YEAR = 2021
RES_DEG = 2.5

DEFAULT_VARS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
    "toa_incident_solar_radiation",
    "geopotential_at_surface",
    "land_sea_mask",
]


def list_year_files(input_dir: Path, year: int) -> list[Path]:
    return sorted(input_dir.glob(f"era5_{year:04d}-*.nc"))


def open_era5(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)

    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})

    if ds["lat"][0] < ds["lat"][-1]:
        ds = ds.sortby("lat")

    return ds


def to_float32(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == np.dtype("|V2"):
        arr = arr.view(np.float16)
    return np.asarray(arr, dtype=np.float32)


def build_grid_edges(res_deg: float):
    lat_edges = np.arange(-90.0, 90.0 + res_deg, res_deg)
    lon_edges = np.arange(0.0, 360.0 + res_deg, res_deg)

    lat_edges[-1] = np.nextafter(lat_edges[-1], np.inf)
    lon_edges[-1] = np.nextafter(lon_edges[-1], np.inf)

    return lat_edges, lon_edges


def build_cell_index(lat_1d, lon_1d, lat_edges, lon_edges):
    lat2d, lon2d = np.meshgrid(lat_1d, lon_1d, indexing="ij")
    flat_lat = lat2d.ravel()
    flat_lon = np.mod(lon2d.ravel(), 360.0)

    lat_idx = np.digitize(flat_lat, lat_edges) - 1
    lon_idx = np.digitize(flat_lon, lon_edges) - 1

    n_lat = len(lat_edges) - 1
    n_lon = len(lon_edges) - 1

    valid = (
        (lat_idx >= 0) & (lat_idx < n_lat) &
        (lon_idx >= 0) & (lon_idx < n_lon)
    )
    cell_idx = lat_idx * n_lon + lon_idx
    return cell_idx, valid, n_lat, n_lon


def coarsen_2d(field_2d, cell_idx, valid, n_lat, n_lon):
    values = to_float32(field_2d).ravel()
    values = values[valid]
    idx = cell_idx[valid]

    finite = np.isfinite(values)
    values = values[finite]
    idx = idx[finite]

    sums = np.zeros(n_lat * n_lon, dtype=np.float64)
    counts = np.zeros(n_lat * n_lon, dtype=np.int32)

    np.add.at(sums, idx, values)
    np.add.at(counts, idx, 1)

    out = np.full(n_lat * n_lon, np.nan, dtype=np.float32)
    mask = counts > 0
    out[mask] = (sums[mask] / counts[mask]).astype(np.float32)
    return out.reshape(n_lat, n_lon)


def feature_specs(ds: xr.Dataset):
    time_features = []
    static_features = []

    for var in DEFAULT_VARS:
        if var not in ds.data_vars:
            continue

        dims = ds[var].dims
        if "time" in dims:
            if "level" in dims:
                nlev = ds[var].sizes["level"]
                for lev in range(nlev):
                    time_features.append((f"{var}_lev{lev:02d}", var, lev))
            else:
                time_features.append((var, var, None))
        else:
            static_features.append((var, var, None))

    return time_features, static_features


def main():
    year_dir = OUTPUT_DIR / f"{YEAR}"
    time_dir = year_dir / "time_series"
    static_dir = year_dir / "static"
    time_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    files = list_year_files(INPUT_DIR, YEAR)
    if not files:
        raise FileNotFoundError(f"No ERA5 files found for {YEAR} in {INPUT_DIR}")

    print(f"Found {len(files)} daily files")

    first = open_era5(files[0])
    lat = to_float32(first["lat"].values)
    lon = to_float32(first["lon"].values)

    lat_edges, lon_edges = build_grid_edges(RES_DEG)
    cell_idx, valid, n_lat, n_lon = build_cell_index(lat, lon, lat_edges, lon_edges)

    time_features, static_features = feature_specs(first)
    print("Time features:")
    for name, _, _ in time_features:
        print(" ", name)
    print("Static features:")
    for name, _, _ in static_features:
        print(" ", name)

    total_times = 0
    time_lengths = []
    for path in files:
        ds = open_era5(path)
        tlen = int(ds.sizes.get("time", 0))
        if tlen > 0:
            total_times += tlen
            time_lengths.append((path.name, tlen))
        ds.close()

    print(f"Total time steps in year: {total_times}")

    # Time-series outputs: one file per variable or variable-level
    time_series = {}
    for name, _, _ in time_features:
        out_path = time_dir / f"{name}.npy"
        time_series[name] = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_times, n_lat, n_lon),
        )

    # Static outputs: one file per variable
    static_outputs = {}
    for name, _, _ in static_features:
        out_path = static_dir / f"{name}.npy"
        static_outputs[name] = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_lat, n_lon),
        )

    time_values = []
    cursor = 0

    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path.name}")
        ds = open_era5(path)

        # record times
        if "time" in ds.coords:
            time_values.extend([np.datetime64(t) for t in ds["time"].values])

        # time-dependent variables
        for out_name, var_name, lev_idx in time_features:
            arr = ds[var_name].values
            arr = to_float32(arr)

            if arr.ndim == 3:
                # (time, lat, lon)
                for t in range(arr.shape[0]):
                    coarse = coarsen_2d(arr[t], cell_idx, valid, n_lat, n_lon)
                    time_series[out_name][cursor + t] = coarse

            elif arr.ndim == 4:
                # (time, level, lat, lon)
                for t in range(arr.shape[0]):
                    coarse = coarsen_2d(arr[t, lev_idx], cell_idx, valid, n_lat, n_lon)
                    time_series[out_name][cursor + t] = coarse
            else:
                raise ValueError(f"Unexpected shape for {var_name}: {arr.shape}")

        # static variables only once
        if cursor == 0:
            for out_name, var_name, _ in static_features:
                arr = ds[var_name].values
                arr = to_float32(arr)
                static_outputs[out_name][:] = coarsen_2d(arr, cell_idx, valid, n_lat, n_lon)

        cursor += int(ds.sizes.get("time", 0))

        ds.close()
        del ds
        gc.collect()

    # Flush memmaps
    for mm in time_series.values():
        mm.flush()
    for mm in static_outputs.values():
        mm.flush()

    metadata = {
        "year": YEAR,
        "input_dir": str(INPUT_DIR),
        "output_dir": str(year_dir),
        "resolution_degrees": RES_DEG,
        "grid_shape": [n_lat, n_lon],
        "lat_edges": lat_edges.tolist(),
        "lon_edges": lon_edges.tolist(),
        "time_features": [name for name, _, _ in time_features],
        "static_features": [name for name, _, _ in static_features],
        "time_values_count": len(time_values),
    }

    with open(year_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    np.save(year_dir / "time_values.npy", np.array(time_values, dtype="datetime64[ns]"))
    print(f"Saved coarse grids to: {year_dir}")


if __name__ == "__main__":
    main()