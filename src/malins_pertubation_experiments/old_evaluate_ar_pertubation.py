#!/usr/bin/env python3
"""
Evaluate multi-day AR-direction perturbation forecasts.

This version keeps the original evaluation style, but runs it twice:
  1. first rollout step
  2. last rollout step

It also saves MP4 videos showing the full delta-IVT trajectory over all rollout steps.

Expected filenames:
    AR_gamma_-0.05_2021-01-03T06.nc
    AR_gamma_0.0_2021-01-03T06.nc
    AR_gamma_0.05_2021-01-03T06.nc
"""

import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


# =====================
# CONFIG
# =====================

THRESHOLD = None
BASE_DIR = f"plots/malins_experiments/pertubation_experiments/pertubation_threshold_{THRESHOLD}"
OUT_DIR = os.path.join(BASE_DIR, "evaluation")
INPUT_DIR = os.path.join(BASE_DIR, "data")
MASK_DIR = "/share/prj-4d/graphcast_shared/data/ClimateNetLarge/AR_labels_cleaned"
USE_MASK = True

CONTROL_GAMMA = 0.0
MAX_MASK_TIME_DIFFERENCE_HOURS = 3

# Standard AR-like IVT threshold; used only as an auxiliary metric.
IVT_THRESHOLD = 250.0

# Keep the same plots as before, but make them for first and last rollout step.
TIME_SELECTIONS = ["first", "last"]
MAKE_DELTA_IVT_MAPS = True

# Additional animation over the full rollout trajectory.
MAKE_TRAJECTORY_VIDEO = True
VIDEO_FPS = 2
VIDEO_FORMAT = "mp4"  # "mp4" needs ffmpeg; use "gif" if ffmpeg is unavailable.

# Optional: reduce video resolution / file size by plotting every nth frame.
VIDEO_FRAME_STRIDE = 1

# Only generate videos for the lowest, highest, and control gamma values.
VIDEO_GAMMA_SELECTION = "extremes_and_control"

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
    path = Path(path)

    run_folder = path.parent.parent.name
    timestep_folder = path.parent.name

    pattern = r"AR_gamma_([+-]?\d+(?:\.\d+)?)_(\d{4}-\d{2}-\d{2}T\d{2})$"
    m = re.search(pattern, run_folder)

    if not m:
        raise ValueError(f"Could not parse gamma/time from folder: {run_folder}")

    gamma = float(m.group(1))
    center_time = pd.Timestamp(m.group(2))

    timestep = timestep_folder.replace("timestep_", "")

    return gamma, center_time

def lead_hours_from_time(time_values):
    return time_values.astype("timedelta64[h]").astype(int)


def plot_delta_ivt_trajectory_all_gammas(
    center_time,
    available_gammas,
    datasets,
    control_key,
    ar_mask,
    out_dir,
):
    control_ds = datasets[control_key]
    control_ivt = compute_ivt(control_ds)

    plt.figure(figsize=(7, 4.5))

    for gamma in available_gammas:
        if gamma == CONTROL_GAMMA:
            continue

        key = (center_time, gamma)
        if key not in datasets:
            continue

        perturbed_ds = datasets[key]
        perturbed_ivt = compute_ivt(perturbed_ds)

        delta = perturbed_ivt - control_ivt

        means = []
        for t in range(delta.sizes["time"]):
            means.append(
                area_weighted_mean(
                    delta.isel(time=t),
                    ar_mask,
                )
            )

        lead_hours = lead_hours_from_time(delta.time.values)

        plt.plot(
            lead_hours,
            means,
            marker="o",
            linewidth=2,
            label=f"γ={gamma:+.2f}",
        )

    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Forecast lead time [h]")
    plt.ylabel("ΔIVT inside AR mask")
    plt.title(f"ΔIVT trajectory inside AR mask\n{center_time}")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Perturbation")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)

    safe_time = str(center_time).replace(":", "").replace(" ", "T")
    out_path = os.path.join(
        out_dir,
        f"delta_ivt_inside_mask_trajectory_all_gammas_{safe_time}.png",
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out_path)

def make_all_delta_ivt_trajectory_plots():
    out_dir = os.path.join(OUT_DIR, "trajectory_plots")
    os.makedirs(out_dir, exist_ok=True)

    file_table = discover_files(INPUT_DIR)

    datasets = {}

    for _, row in file_table.iterrows():
        key = (row["center_time"], row["gamma"])
        datasets[key] = load_prediction(row["file"], time_selection=None)

    for center_time in sorted(file_table["center_time"].unique()):
        center_time = pd.Timestamp(center_time)

        available_gammas = sorted(
            file_table.loc[file_table["center_time"] == center_time, "gamma"].unique()
        )

        if CONTROL_GAMMA not in available_gammas:
            print(f"[SKIP TRAJECTORY PLOT] {center_time}: no control")
            continue

        control_key = (center_time, CONTROL_GAMMA)
        control_ivt = compute_ivt(datasets[control_key])

        try:
            ar_mask, mask_path = load_mask_on_grid(center_time, control_ivt.isel(time=0))
            print(f"Using mask for trajectory plot {center_time}: {mask_path}")
        except Exception as e:
            print(f"[SKIP TRAJECTORY PLOT] {center_time}: could not load mask: {e}")
            continue

        plot_delta_ivt_trajectory_all_gammas(
            center_time=center_time,
            available_gammas=available_gammas,
            datasets=datasets,
            control_key=control_key,
            ar_mask=ar_mask,
            out_dir=out_dir,
        )


def discover_files(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "AR_gamma_*", "timestep_*", "prediction.nc")))

    if not files:
        raise FileNotFoundError(f"No AR_gamma_*.nc files found in {input_dir}")

    rows = []
    for f in files:
        gamma, center_time = parse_gamma_and_time(f)
        rows.append({"file": f, "gamma": gamma, "center_time": center_time})

    df = pd.DataFrame(rows).sort_values(["center_time", "gamma"]).reset_index(drop=True)

    print(f"Found {len(df)} forecast files.")
    print("Gammas:", sorted(df["gamma"].unique()))
    print("Center times:", [str(t) for t in sorted(df["center_time"].unique())])
    return df


def load_prediction(path, time_selection=None):
    """
    time_selection:
      None    -> keep all rollout lead times
      'first' -> select first forecast lead
      'last'  -> select last forecast lead
    """
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


# =====================
# METEOROLOGICAL METRICS
# =====================

def compute_ivt(ds):
    """
    Compute integrated vapor transport:
        IVT = sqrt((1/g int q*u dp)^2 + (1/g int q*v dp)^2)
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

def plot_dose_response(summary, metric, ylabel, out_name, out_dir):
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
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def plot_delta_map(delta_da, gamma, center_time, out_name, out_dir, lead_label=None):
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

    title = f"ΔIVT: gamma={gamma:+.2f} minus gamma=0, {center_time}"
    if lead_label is not None:
        title += f", {lead_label}"
    plt.title(title)
    plt.tight_layout()

    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

def plot_ivt_map(ivt, gamma, center_time, out_name, out_dir):
    plt.figure(figsize=(10, 4.8))

    ivt.plot(
        cmap="viridis",
        cbar_kwargs={"label": "IVT"}
    )

    plt.title(f"IVT gamma={gamma:+.2f}, {center_time}")
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, out_name), dpi=300)
    plt.close()


def make_delta_ivt_video(center_time, gamma, control_file, perturbed_file):
    control = load_prediction(control_file, time_selection=None)
    perturbed = load_prediction(perturbed_file, time_selection=None)

    control_ivt = compute_ivt(control)
    perturbed_ivt = compute_ivt(perturbed)
    delta = perturbed_ivt - control_ivt

    if "time" not in delta.dims:
        print(f"[SKIP VIDEO] {center_time}, gamma={gamma}: no time dimension")
        return

    delta = delta.isel(time=slice(None, None, VIDEO_FRAME_STRIDE))

    lat_name = get_lat_name(delta)
    lon_name = get_lon_name(delta)

    vmax = np.nanpercentile(np.abs(delta.values), 99)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = np.nanmax(np.abs(delta.values))

    fig, ax = plt.subplots(figsize=(10, 4.8))

    first = delta.isel(time=0)
    first.plot(
        ax=ax,
        x=lon_name,
        y=lat_name,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        cbar_kwargs={"label": "ΔIVT"},
    )

    def update(i):
        ax.clear()
        frame = delta.isel(time=i)
        frame.plot(
            ax=ax,
            x=lon_name,
            y=lat_name,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            add_colorbar=False,
        )
        lead = str(delta.time.values[i])
        ax.set_title(f"ΔIVT trajectory: gamma={gamma:+.2f}, {center_time}, lead={lead}")
        return []

    anim = FuncAnimation(fig, update, frames=delta.sizes["time"], interval=500)

    video_dir = os.path.join(OUT_DIR, "delta_videos")
    os.makedirs(video_dir, exist_ok=True)

    safe_time = str(center_time).replace(":", "").replace(" ", "T")

    if VIDEO_FORMAT == "mp4":
        out_path = os.path.join(video_dir, f"delta_ivt_gamma_{gamma:+.2f}_{safe_time}.mp4")
        writer = FFMpegWriter(fps=VIDEO_FPS)
    elif VIDEO_FORMAT == "gif":
        out_path = os.path.join(video_dir, f"delta_ivt_gamma_{gamma:+.2f}_{safe_time}.gif")
        writer = PillowWriter(fps=VIDEO_FPS)
    else:
        raise ValueError("VIDEO_FORMAT must be 'mp4' or 'gif'.")

    anim.save(out_path, writer=writer, dpi=150)
    plt.close(fig)
    print("Saved video:", out_path)


# =====================
# EVALUATION
# =====================

def evaluate_one_step(time_selection):
    out_dir = os.path.join(OUT_DIR, f"lead_{time_selection}")
    os.makedirs(out_dir, exist_ok=True)

    trajectory_dir = os.path.join(out_dir, "trajectory_plots")
    os.makedirs(trajectory_dir, exist_ok=True)

    file_table = discover_files(INPUT_DIR)

    datasets = {}
    ivts = {}

    for _, row in file_table.iterrows():
        key = (row["center_time"], row["gamma"])
        ds = load_prediction(row["file"], time_selection=time_selection)
        datasets[key] = ds
        ivts[key] = compute_ivt(ds)

    records = []

    for center_time in sorted(file_table["center_time"].unique()):
        center_time = pd.Timestamp(center_time)

        available_gammas = sorted(
            file_table.loc[file_table["center_time"] == center_time, "gamma"].unique()
        )

        if CONTROL_GAMMA not in available_gammas:
            print(f"[SKIP] {center_time}: no control gamma={CONTROL_GAMMA}")
            continue

        control_key = (center_time, CONTROL_GAMMA)
        control_ivt = ivts[control_key]

        ar_mask = None
        mask_path = None

        if USE_MASK:
            try:
                ar_mask, mask_path = load_mask_on_grid(center_time, control_ivt)
                print(f"Using mask for {center_time}: {mask_path}")
            except Exception as e:
                print(f"[WARN] Could not load mask for {center_time}: {e}")

        for gamma in available_gammas:
            key = (center_time, gamma)
            ds = datasets[key]
            ivt = ivts[key]
            delta_ivt = ivt - control_ivt

            ar_like = ivt >= IVT_THRESHOLD
            control_ar_like = control_ivt >= IVT_THRESHOLD

            rec = {
                "time_selection": time_selection,
                "center_time": str(center_time),
                "gamma": gamma,
                "file": file_table[
                    (file_table["center_time"] == center_time)
                    & (file_table["gamma"] == gamma)
                ]["file"].iloc[0],
                "mask_file": mask_path,
                "ivt_global_mean": area_weighted_mean(ivt),
                "ivt_global_max": max_value(ivt),
                "delta_ivt_global_mean": area_weighted_mean(delta_ivt),
                "delta_ivt_global_max_abs": max_value(abs(delta_ivt)),
                "ar_like_area_fraction": area_fraction(ar_like),
                "delta_ar_like_area_fraction": area_fraction(ar_like) - area_fraction(control_ar_like),
            }

            if TP_VAR in ds:
                tp = ds[TP_VAR]
                control_tp = datasets[control_key][TP_VAR]
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

                if TP_VAR in ds:
                    rec.update({
                        "precip_inside_mask_mean": area_weighted_mean(tp, ar_mask),
                        "delta_precip_inside_mask_mean": area_weighted_mean(delta_tp, ar_mask),
                    })


            records.append(rec)

            if MAKE_DELTA_IVT_MAPS and gamma != CONTROL_GAMMA:
                safe_time = str(center_time).replace(":", "").replace(" ", "T")
                out_name = f"delta_ivt_{time_selection}_gamma_{gamma:+.2f}_{safe_time}.png"
                plot_delta_map(
                    delta_ivt,
                    gamma,
                    center_time,
                    out_name,
                    out_dir,
                    lead_label=f"{time_selection} lead",
                )
        plot_ivt_map(ivt, 0, center_time, out_name, out_dir)


    summary = pd.DataFrame(records).sort_values(["center_time", "gamma"])

    summary_path = os.path.join(out_dir, f"gamma_summary_metrics_{time_selection}.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)
    print(summary)

    plot_dose_response(
        summary,
        metric="delta_ivt_global_mean",
        ylabel=f"Global mean ΔIVT ({time_selection} lead)",
        out_name=f"dose_response_delta_ivt_global_mean_{time_selection}.png",
        out_dir=out_dir,
    )

    if "delta_ivt_inside_mask_mean" in summary.columns:
        plot_dose_response(
            summary,
            metric="delta_ivt_inside_mask_mean",
            ylabel=f"Mean ΔIVT inside AR mask ({time_selection} lead)",
            out_name=f"dose_response_delta_ivt_inside_mask_mean_{time_selection}.png",
            out_dir=out_dir,
        )

        plot_dose_response(
            summary,
            metric="delta_ivt_outside_mask_mean",
            ylabel=f"Mean ΔIVT outside AR mask ({time_selection} lead)",
            out_name=f"dose_response_delta_ivt_outside_mask_mean_{time_selection}.png",
            out_dir=out_dir,
        )

    if "delta_precip_inside_mask_mean" in summary.columns:
        plot_dose_response(
            summary,
            metric="delta_precip_inside_mask_mean",
            ylabel=f"Mean Δprecipitation inside AR mask ({time_selection} lead)",
            out_name=f"dose_response_delta_precip_inside_mask_mean_{time_selection}.png",
            out_dir=out_dir,
        )


def select_video_gammas(group):
    if VIDEO_GAMMA_SELECTION != "extremes_and_control":
        return sorted(group["gamma"].unique())

    available = sorted(group["gamma"].unique())
    selected = []

    if CONTROL_GAMMA in available:
        selected.append(CONTROL_GAMMA)

    if available:
        selected.append(available[0])  # lowest
        selected.append(available[-1])  # highest

    return sorted(set(selected))


def make_all_videos():
    file_table = discover_files(INPUT_DIR)

    for center_time in sorted(file_table["center_time"].unique()):
        center_time = pd.Timestamp(center_time)
        group = file_table[file_table["center_time"] == center_time]

        if CONTROL_GAMMA not in group["gamma"].values:
            print(f"[SKIP VIDEO] {center_time}: no control")
            continue

        control_file = group[group["gamma"] == CONTROL_GAMMA]["file"].iloc[0]
        selected_gammas = select_video_gammas(group)

        for gamma in selected_gammas:
            if gamma == CONTROL_GAMMA:
                continue

            perturbed_file = group[group["gamma"] == gamma]["file"].iloc[0]
            make_delta_ivt_video(center_time, gamma, control_file, perturbed_file)


def make_ivt_video(center_time, gamma, forecast_file):
    ds = load_prediction(forecast_file, time_selection=None)
    ivt = compute_ivt(ds)

    if "time" not in ivt.dims:
        print(f"[SKIP IVT VIDEO] {center_time}, gamma={gamma}: no time dimension")
        return

    ivt = ivt.isel(time=slice(None, None, VIDEO_FRAME_STRIDE))

    lat_name = get_lat_name(ivt)
    lon_name = get_lon_name(ivt)

    vmax = np.nanpercentile(ivt.values, 99)

    fig, ax = plt.subplots(figsize=(10, 4.8))

    first = ivt.isel(time=0)
    im = first.plot(
        ax=ax,
        x=lon_name,
        y=lat_name,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        add_colorbar=False,
    )

    fig.colorbar(im, ax=ax, label="IVT")

    def update(i):
        ax.clear()
        frame = ivt.isel(time=i)

        frame.plot(
            ax=ax,
            x=lon_name,
            y=lat_name,
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            add_colorbar=False,
        )

        lead = str(ivt.time.values[i])
        ax.set_title(f"IVT trajectory: gamma={gamma:+.2f}, {center_time}, lead={lead}")
        return []

    anim = FuncAnimation(fig, update, frames=ivt.sizes["time"], interval=500)

    video_dir = os.path.join(OUT_DIR, "ivt_videos")
    os.makedirs(video_dir, exist_ok=True)

    safe_time = str(center_time).replace(":", "").replace(" ", "T")

    if VIDEO_FORMAT == "mp4":
        out_path = os.path.join(video_dir, f"ivt_gamma_{gamma:+.2f}_{safe_time}.mp4")
        writer = FFMpegWriter(fps=VIDEO_FPS)
    elif VIDEO_FORMAT == "gif":
        out_path = os.path.join(video_dir, f"ivt_gamma_{gamma:+.2f}_{safe_time}.gif")
        writer = PillowWriter(fps=VIDEO_FPS)
    else:
        raise ValueError("VIDEO_FORMAT must be 'mp4' or 'gif'.")

    anim.save(out_path, writer=writer, dpi=150)
    plt.close(fig)
    print("Saved IVT video:", out_path)

def make_all_ivt_videos():
    file_table = discover_files(INPUT_DIR)

    for center_time in sorted(file_table["center_time"].unique()):
        center_time = pd.Timestamp(center_time)
        group = file_table[file_table["center_time"] == center_time]
        selected_gammas = select_video_gammas(group)

        for _, row in group[group["gamma"].isin(selected_gammas)].iterrows():
            make_ivt_video(
                center_time=center_time,
                gamma=row["gamma"],
                forecast_file=row["file"],
            )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for time_selection in TIME_SELECTIONS:
        print(f"\n[EVALUATING {time_selection.upper()} LEAD]\n")
        evaluate_one_step(time_selection)


    print("\n[MAKING ΔIVT TRAJECTORY PLOTS]\n")
    make_all_delta_ivt_trajectory_plots()

    if MAKE_TRAJECTORY_VIDEO:
        print("\n[MAKING TRAJECTORY VIDEOS]\n")
        make_all_videos()

        print("\n[MAKING IVT TRAJECTORY VIDEOS]\n")
        make_all_ivt_videos()

    print("[DONE]")


if __name__ == "__main__":
    main()
