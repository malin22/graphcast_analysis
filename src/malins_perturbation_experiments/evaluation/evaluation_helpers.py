import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def open_era5_day(date_str, era5_daily_dir):
    path = os.path.join(era5_daily_dir, f"era5_{date_str}.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERA5 file not found: {path}")
    return xr.open_dataset(path)


def standardize_era5_coordinates(ds):
    rename = {}
    if "latitude" in ds.coords and "lat" not in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords and "lon" not in ds.coords:
        rename["longitude"] = "lon"
    if "valid_time" in ds.coords and "time" not in ds.coords:
        rename["valid_time"] = "time"
    if rename:
        ds = ds.rename(rename)
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    return ds


def load_era5_at_time(valid_time, era5_daily_dir):
    valid_time = pd.Timestamp(valid_time)
    date_str = valid_time.strftime("%Y-%m-%d")
    era5_path = os.path.join(era5_daily_dir, f"era5_{date_str}.nc")
    ds = standardize_era5_coordinates(open_era5_day(date_str, era5_daily_dir))
    if "time" not in ds.coords:
        ds.close()
        raise ValueError(f"No time coordinate found in ERA5 file: {era5_path}")
    try:
        step = ds.sel(time=np.datetime64(valid_time.to_datetime64())).load()
    except KeyError:
        available = pd.to_datetime(ds["time"].values)
        ds.close()
        raise KeyError(
            f"ERA5 time {valid_time} was not found in {era5_path}.\n"
            f"Available times: {list(available)}"
        )
    ds.close()
    return step, era5_path


def get_valid_time(ds, time_index, center_str):
    if "datetime" in ds.coords:
        dt = ds["datetime"]
        if "batch" in dt.dims:
            dt = dt.isel(batch=0)
        return pd.Timestamp(dt.isel(time=time_index).values)
    return pd.Timestamp(center_str) + pd.to_timedelta(ds.time.values[time_index])


def discover_files(input_dir, center_str):
    files = sorted(glob.glob(os.path.join(input_dir, "gamma_*.nc")))
    if not files:
        raise FileNotFoundError(f"No gamma_*.nc files found in {input_dir}")
    rows = []
    center_time = pd.Timestamp(center_str)
    for path in files:
        match = re.match(r"gamma_([+-]?\d+(?:\.\d+)?)\.nc$", os.path.basename(path))
        if match:
            rows.append({"file": path, "gamma": float(match.group(1)), "center_time": center_time})
    df = pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)
    print(f"Found {len(df)} forecast files in {input_dir}")
    print("Gammas:", sorted(df["gamma"].unique()))
    print("Center time:", center_str)
    return df


def load_prediction(path, time_selection=None):
    ds = xr.open_dataset(path)
    if "batch" in ds.dims:
        ds = ds.isel(batch=0)
    if time_selection is not None and "time" in ds.dims:
        if time_selection == "first":
            ds = ds.isel(time=0)
        elif time_selection == "last":
            ds = ds.isel(time=-1)
        else:
            raise ValueError(f"Unknown time_selection: {time_selection}")
    return ds


def gamma_colors(gammas):
    cmap = plt.get_cmap("coolwarm")
    norm = TwoSlopeNorm(vmin=min(gammas), vcenter=0.0, vmax=max(gammas))
    return {g: cmap(norm(g)) for g in gammas}, cmap, norm


def get_lat_name(da):
    if "lat" in da.coords:
        return "lat"
    if "latitude" in da.coords:
        return "latitude"
    raise ValueError("Could not find latitude coordinate.")


def get_lon_name(da):
    if "lon" in da.coords:
        return "lon"
    if "longitude" in da.coords:
        return "longitude"
    raise ValueError("Could not find longitude coordinate.")


def format_lead_time(hours):
    hours = int(hours)
    days, rem_hours = divmod(hours, 24)
    if days == 0:
        return f"{rem_hours}h"
    if rem_hours == 0:
        return f"{days}d"
    return f"{days}d {rem_hours}h"


def area_weighted_mean(da, mask=None):
    lat_name = get_lat_name(da)
    weights = np.cos(np.deg2rad(da[lat_name])).broadcast_like(da)
    x, w = da, weights
    if mask is not None:
        x = x.where(mask)
        w = w.where(mask)
    valid = np.isfinite(x)
    numerator = (x.where(valid) * w.where(valid)).sum(skipna=True)
    denominator = w.where(valid).sum(skipna=True)
    return float(numerator / denominator)


def max_value(da, mask=None):
    x = da.where(mask) if mask is not None else da
    return float(x.max(skipna=True).values)


def nearest_mask_file(center_time, mask_dir, max_time_difference_hours=3):
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.nc")))
    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {mask_dir}")
    best_file = min(
        mask_files,
        key=lambda f: abs(pd.Timestamp(os.path.basename(f).replace(".nc", "")) - pd.Timestamp(center_time)),
    )
    best_diff = abs(pd.Timestamp(os.path.basename(best_file).replace(".nc", "")) - pd.Timestamp(center_time))
    if best_diff > pd.Timedelta(hours=max_time_difference_hours):
        raise ValueError(
            f"No mask within {max_time_difference_hours}h for {center_time}. Best diff was {best_diff}."
        )
    return best_file


def load_mask_on_grid(target_time, target_da, mask_dir, max_time_difference_hours=3):
    mask_path = nearest_mask_file(target_time, mask_dir, max_time_difference_hours)
    mask_time = pd.Timestamp(os.path.basename(mask_path).replace(".nc", ""))
    diff_hours = abs(mask_time - pd.Timestamp(target_time)) / pd.Timedelta(hours=1)
    with xr.open_dataset(mask_path) as ds:
        mask = (ds["label"].min("annotator") > 0).astype(float).load()
    lat_name = get_lat_name(target_da)
    lon_name = get_lon_name(target_da)
    mask_interp = mask.interp(
        latitude=target_da[lat_name], longitude=target_da[lon_name], method="nearest"
    )
    rename = {}
    if lat_name != "latitude" and "latitude" in mask_interp.dims:
        rename["latitude"] = lat_name
    if lon_name != "longitude" and "longitude" in mask_interp.dims:
        rename["longitude"] = lon_name
    if rename:
        mask_interp = mask_interp.rename(rename)
    return mask_interp.astype(bool), mask_path, mask_time, diff_hours


def find_next_timestep_with_mask(
    ds_full, start_time_index, center_str, mask_dir, max_time_difference_hours=3, direction=1
):
    n_times = ds_full.sizes["time"]
    if start_time_index < 0:
        start_time_index += n_times
    indices = range(start_time_index, n_times) if direction == 1 else range(start_time_index, -1, -1)
    for t_idx in indices:
        valid_time = get_valid_time(ds_full, t_idx, center_str)
        try:
            mask_path = nearest_mask_file(valid_time, mask_dir, max_time_difference_hours)
            mask_time = pd.Timestamp(os.path.basename(mask_path).replace(".nc", ""))
            mask_diff_h = abs(mask_time - valid_time) / pd.Timedelta(hours=1)
            return t_idx, valid_time, mask_path, mask_time, mask_diff_h
        except Exception as exc:
            print(f"[NO MASK] forecast step {t_idx}, valid_time={valid_time}: {exc}")
    return None, None, None, None, None
