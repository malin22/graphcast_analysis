#!/usr/bin/env python3
"""
Evaluate AR-direction perturbation forecasts.

What this script does:
1. Reads all available gamma runs from INPUT_DIR.
2. Automatically parses gamma values and center times from filenames.
3. Uses gamma=0 as the control forecast.
4. Computes IVT from q, u, v.
5. Optionally matches a ClimateNet AR mask.
6. Saves a compact summary CSV.
7. Saves only the most useful plots:
   - dose-response curves
   - delta-IVT maps relative to gamma=0

Expected filenames:
    AR_gamma_-0.05_2021-01-03T06.nc
    AR_gamma_0.0_2021-01-03T06.nc
    AR_gamma_0.05_2021-01-03T06.nc

If your filenames use +0.05 instead of 0.05, that also works.
"""

import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# =====================
# CONFIG
# =====================

THRESHOLD = None
INPUT_DIR = f"plots/malins_experiments/pertubation_experiments/AR/pertubation_threshold_{THRESHOLD}/"


OUT_DIR = os.path.join(INPUT_DIR, "evaluation")

MASK_DIR = "/share/prj-4d/graphcast_shared/data/ClimateNetLarge/AR_labels_cleaned"
USE_MASK = True

CONTROL_GAMMA = 0.0
MAX_MASK_TIME_DIFFERENCE_HOURS = 3

# Standard AR-like IVT threshold; used only as an auxiliary metric.
IVT_THRESHOLD = 250.0

# Make only the essential maps.
MAKE_DELTA_IVT_MAPS = True

# Evaluate first forecast lead by default.
# If your output contains multiple forecast lead times and you want the last one,
# change this to "last".
TIME_SELECTION = "first"  # "first" or "last"



# Variable names in GraphCast output
Q_VAR = "specific_humidity"
U_VAR = "u_component_of_wind"
V_VAR = "v_component_of_wind"
TP_VAR = "total_precipitation_6hr"

G = 9.80665


# =====================
# FILE HANDLING
# =====================

def parse_gamma_and_time(path):
    """
    Parse gamma and center time from filenames such as:
    AR_gamma_-0.05_2021-01-03T06.nc
    AR_gamma_+0.05_2021-01-03T06.nc
    AR_gamma_0.0_2021-01-03T06.nc
    """
    name = os.path.basename(path)

    pattern = r"AR_gamma_([+-]?\d+(?:\.\d+)?)_(\d{4}-\d{2}-\d{2}T\d{2})\.nc$"
    m = re.search(pattern, name)

    if not m:
        raise ValueError(f"Could not parse gamma/time from filename: {name}")

    gamma = float(m.group(1))
    center_time = pd.Timestamp(m.group(2))

    return gamma, center_time


def discover_files(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "AR_gamma_*.nc")))

    if not files:
        raise FileNotFoundError(f"No AR_gamma_*.nc files found in {input_dir}")

    rows = []
    for f in files:
        gamma, center_time = parse_gamma_and_time(f)
        rows.append({
            "file": f,
            "gamma": gamma,
            "center_time": center_time,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["center_time", "gamma"]).reset_index(drop=True)

    print(f"Found {len(df)} forecast files.")
    print("Gammas:", sorted(df["gamma"].unique()))
    print("Center times:", [str(t) for t in sorted(df["center_time"].unique())])

    return df


def load_prediction(path):
    ds = xr.open_dataset(path)

    if "batch" in ds.dims:
        ds = ds.isel(batch=0)

    if "time" in ds.dims:
        if TIME_SELECTION == "first":
            ds = ds.isel(time=0)
        elif TIME_SELECTION == "last":
            ds = ds.isel(time=-1)
        else:
            raise ValueError(f"Unknown TIME_SELECTION: {TIME_SELECTION}")

    return ds


# =====================
# METEOROLOGICAL METRICS
# =====================

def compute_ivt(ds):
    """
    Compute integrated vapor transport:

        IVT = sqrt((1/g int q*u dp)^2 + (1/g int q*v dp)^2)

    q: specific humidity
    u/v: horizontal wind
    p: pressure levels
    """
    q = ds[Q_VAR]
    u = ds[U_VAR]
    v = ds[V_VAR]

    if "level" not in q.dims:
        raise ValueError("Expected pressure-level variable with dimension 'level'.")

    p = ds["level"].values.astype(float)

    # Convert hPa to Pa if needed.
    if np.nanmax(p) < 2000:
        p = p * 100.0

    # Integrate in increasing pressure order.
    order = np.argsort(p)
    p_sorted = p[order]

    q = q.isel(level=order)
    u = u.isel(level=order)
    v = v.isel(level=order)

    level_axis = q.get_axis_num("level")

    ivt_u = np.trapezoid((q * u).values, x=p_sorted, axis=level_axis) / G
    ivt_v = np.trapezoid((q * v).values, x=p_sorted, axis=level_axis) / G

    dims = [d for d in q.dims if d != "level"]
    coords = {d: q.coords[d] for d in dims if d in q.coords}

    ivt = np.sqrt(ivt_u**2 + ivt_v**2)

    return xr.DataArray(ivt, dims=dims, coords=coords, name="ivt")


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


def area_weighted_mean(da, mask=None):
    lat_name = get_lat_name(da)
    weights = np.cos(np.deg2rad(da[lat_name])).broadcast_like(da)

    x = da
    w = weights

    if mask is not None:
        x = x.where(mask)
        w = w.where(mask)

    valid = np.isfinite(x)
    numerator = (x.where(valid) * w.where(valid)).sum(skipna=True)
    denominator = w.where(valid).sum(skipna=True)

    return float(numerator / denominator)


def area_fraction(mask):
    return float(mask.mean().values)


def max_value(da, mask=None):
    x = da.where(mask) if mask is not None else da
    return float(x.max(skipna=True).values)


# =====================
# MASK HANDLING
# =====================

def nearest_mask_file(center_time):
    mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*.nc")))

    if not mask_files:
        raise FileNotFoundError(f"No mask files found in {MASK_DIR}")

    best_file = None
    best_diff = None

    for f in mask_files:
        t = pd.Timestamp(os.path.basename(f).replace(".nc", ""))
        diff = abs(t - center_time)

        if best_diff is None or diff < best_diff:
            best_file = f
            best_diff = diff

    if best_diff > pd.Timedelta(hours=MAX_MASK_TIME_DIFFERENCE_HOURS):
        raise ValueError(
            f"No mask within {MAX_MASK_TIME_DIFFERENCE_HOURS}h for {center_time}. "
            f"Best diff was {best_diff}."
        )

    return best_file


def load_mask_on_grid(center_time, target_da):
    """
    Load ClimateNet mask and interpolate to forecast lat/lon grid.
    Uses annotator intersection.
    """
    mask_path = nearest_mask_file(center_time)

    ds = xr.open_dataset(mask_path)
    label = ds["label"]
    mask = (label.min("annotator") > 0).astype(float)

    lat_name = get_lat_name(target_da)
    lon_name = get_lon_name(target_da)

    mask_interp = mask.interp(
        latitude=target_da[lat_name],
        longitude=target_da[lon_name],
        method="nearest",
    )

    rename = {}
    if lat_name != "latitude" and "latitude" in mask_interp.dims:
        rename["latitude"] = lat_name
    if lon_name != "longitude" and "longitude" in mask_interp.dims:
        rename["longitude"] = lon_name

    if rename:
        mask_interp = mask_interp.rename(rename)

    return mask_interp.astype(bool), mask_path


# =====================
# PLOTTING
# =====================

def plot_dose_response(summary, metric, ylabel, out_name):
    plt.figure(figsize=(6, 4))

    for center_time, group in summary.groupby("center_time"):
        group = group.sort_values("gamma")
        plt.plot(group["gamma"], group[metric], marker="o", linewidth=2, label=str(center_time))

    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel("Gamma")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs gamma")
    plt.grid(True, alpha=0.3)

    if summary["center_time"].nunique() <= 6:
        plt.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def plot_delta_map(delta_da, gamma, center_time, out_name):
    lat_name = get_lat_name(delta_da)
    lon_name = get_lon_name(delta_da)

    values = delta_da.values
    vmax = np.nanpercentile(np.abs(values), 99)

    if not np.isfinite(vmax) or vmax == 0:
        vmax = np.nanmax(np.abs(values))

    plt.figure(figsize=(10, 4.8))

    delta_da.plot(
        x=lon_name,
        y=lat_name,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        cbar_kwargs={"label": "ΔIVT"},
    )

    plt.title(f"ΔIVT: gamma={gamma:+.2f} minus gamma=0, {center_time}")
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


# =====================
# MAIN
# =====================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    file_table = discover_files(INPUT_DIR)
    records = []

    for center_time in sorted(file_table["center_time"].unique()):
        center_time = pd.Timestamp(center_time)

        group = file_table[file_table["center_time"] == center_time]
        available_gammas = sorted(group["gamma"].unique())

        if CONTROL_GAMMA not in available_gammas:
            print(f"[SKIP] {center_time}: no control gamma={CONTROL_GAMMA}")
            continue

        control_file = group.loc[group["gamma"] == CONTROL_GAMMA, "file"].iloc[0]

        # Load only control once per center_time
        with load_prediction(control_file) as control_ds:
            control_ivt = compute_ivt(control_ds).load()

            ar_mask = None
            mask_path = None

            if USE_MASK:
                try:
                    ar_mask, mask_path = load_mask_on_grid(center_time, control_ivt)
                    ar_mask = ar_mask.load()
                    print(f"Using mask for {center_time}: {mask_path}")
                except Exception as e:
                    print(f"[WARN] Could not load mask for {center_time}: {e}")

            control_ar_like = control_ivt >= IVT_THRESHOLD
            control_ar_like_fraction = area_fraction(control_ar_like)

            control_tp = None
            if TP_VAR in control_ds:
                control_tp = control_ds[TP_VAR].load()

            for _, row in group.sort_values("gamma").iterrows():
                gamma = row["gamma"]
                path = row["file"]

                with load_prediction(path) as ds:
                    ivt = compute_ivt(ds).load()
                    delta_ivt = ivt - control_ivt

                    ar_like = ivt >= IVT_THRESHOLD

                    rec = {
                        "center_time": str(center_time),
                        "gamma": gamma,
                        "file": path,
                        "mask_file": mask_path,
                        "ivt_global_mean": area_weighted_mean(ivt),
                        "ivt_global_max": max_value(ivt),
                        "delta_ivt_global_mean": area_weighted_mean(delta_ivt),
                        "delta_ivt_global_max_abs": max_value(abs(delta_ivt)),
                        "ar_like_area_fraction": area_fraction(ar_like),
                        "delta_ar_like_area_fraction": area_fraction(ar_like) - control_ar_like_fraction,
                    }

                    if TP_VAR in ds and control_tp is not None:
                        tp = ds[TP_VAR].load()
                        delta_tp = tp - control_tp

                        rec.update({
                            "precip_global_mean": area_weighted_mean(tp),
                            "delta_precip_global_mean": area_weighted_mean(delta_tp),
                            "precip_global_max": max_value(tp),
                        })

                    if ar_mask is not None:
                        rec.update({
                            "ivt_inside_mask_mean": area_weighted_mean(ivt, ar_mask),
                            "ivt_outside_mask_mean": area_weighted_mean(ivt, ~ar_mask),
                            "ivt_inside_mask_max": max_value(ivt, ar_mask),
                            "delta_ivt_inside_mask_mean": area_weighted_mean(delta_ivt, ar_mask),
                            "delta_ivt_outside_mask_mean": area_weighted_mean(delta_ivt, ~ar_mask),
                            "delta_ivt_inside_mask_max_abs": max_value(abs(delta_ivt), ar_mask),
                        })

                        if TP_VAR in ds and control_tp is not None:
                            rec.update({
                                "precip_inside_mask_mean": area_weighted_mean(tp, ar_mask),
                                "delta_precip_inside_mask_mean": area_weighted_mean(delta_tp, ar_mask),
                            })

                    records.append(rec)

                    if MAKE_DELTA_IVT_MAPS and gamma != CONTROL_GAMMA:
                        safe_time = str(center_time).replace(":", "").replace(" ", "T")
                        out_name = f"delta_ivt_gamma_{gamma:+.2f}_{safe_time}.png"
                        plot_delta_map(delta_ivt, gamma, center_time, out_name)

                    del ivt, delta_ivt, ar_like

        del control_ivt

    summary = pd.DataFrame(records).sort_values(["center_time", "gamma"])

    summary_path = os.path.join(OUT_DIR, "gamma_summary_metrics.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)
    print(summary)

    plot_dose_response(
        summary,
        metric="delta_ivt_global_mean",
        ylabel="Global mean ΔIVT",
        out_name="dose_response_delta_ivt_global_mean.png",
    )

    if "delta_ivt_inside_mask_mean" in summary.columns:
        plot_dose_response(
            summary,
            metric="delta_ivt_inside_mask_mean",
            ylabel="Mean ΔIVT inside AR mask",
            out_name="dose_response_delta_ivt_inside_mask_mean.png",
        )

        plot_dose_response(
            summary,
            metric="delta_ivt_outside_mask_mean",
            ylabel="Mean ΔIVT outside AR mask",
            out_name="dose_response_delta_ivt_outside_mask_mean.png",
        )

    if "delta_precip_inside_mask_mean" in summary.columns:
        plot_dose_response(
            summary,
            metric="delta_precip_inside_mask_mean",
            ylabel="Mean Δprecipitation inside AR mask",
            out_name="dose_response_delta_precip_inside_mask_mean.png",
        )

    print("[DONE]")


if __name__ == "__main__":
    main()
