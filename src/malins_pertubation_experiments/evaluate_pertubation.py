
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

WEATHER_FEATURE = "AR"
THRESHOLD = 0.9
CENTER_STR = "2021-02-12T18"
NODE_HIERARCHY_LEVEL = 6

# Only generate videos for the lowest, highest, and control gamma values.
VIDEO_GAMMA_SELECTION = [-0.5, 0.5]

BASE_DIR = os.path.join(
    "plots",
    "malins_experiments",
    "pertubation_experiments",
    WEATHER_FEATURE,
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}",
    f"pertubation_threshold_{THRESHOLD}",
    CENTER_STR,
)

INPUT_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "evaluation")
MASK_DIR = "/share/prj-4d/graphcast_shared/data/ClimateNetLarge/AR_labels_cleaned"

CONTROL_GAMMA = 0.0
MAX_MASK_TIME_DIFFERENCE_HOURS = 3

# Standard AR-like IVT threshold; used only as an auxiliary metric.
#IVT_THRESHOLD = 250.0

# Keep the same plots as before, but make them for first and last rollout step.
TIME_SELECTIONS = ["first", "last"]
MAKE_DELTA_IVT_MAPS = True

# Additional animation over the full rollout trajectory.
MAKE_TRAJECTORY_VIDEO = True
VIDEO_FPS = 2
VIDEO_FORMAT = "mp4"  # "mp4" needs ffmpeg; use "gif" if ffmpeg is unavailable.

# Optional: reduce video resolution / file size by plotting every nth frame.
VIDEO_FRAME_STRIDE = 1



# Variable names in GraphCast output
Q_VAR = "specific_humidity"
U_VAR = "u_component_of_wind"
V_VAR = "v_component_of_wind"
TP_VAR = "total_precipitation_6hr"

G = 9.80665


# =====================
# FILE HANDLING
# =====================

def get_valid_time(ds, time_index):
    if "datetime" in ds.coords:
        dt = ds["datetime"]
        if "batch" in dt.dims:
            dt = dt.isel(batch=0)
        return pd.Timestamp(dt.isel(time=time_index).values)

    # fallback: CENTER_STR + forecast lead
    return pd.Timestamp(CENTER_STR) + pd.to_timedelta(ds.time.values[time_index])

def parse_gamma_and_time(path):
    path = Path(path)

    gamma_match = re.match(r"gamma_([+-]?\d+(?:\.\d+)?)\.nc$", path.name)

    if not gamma_match:
        raise ValueError(f"Could not parse gamma from filename: {path.name}")

    gamma = float(gamma_match.group(1))
    center_time = pd.Timestamp(path.parent.parent.name)

    return gamma, center_time


def discover_files(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "gamma_*.nc")))

    if not files:
        raise FileNotFoundError(f"No gamma_*.nc files found in {input_dir}")

    rows = []
    center_time = pd.Timestamp(CENTER_STR)

    for f in files:
        name = os.path.basename(f)
        m = re.match(r"gamma_([+-]?\d+(?:\.\d+)?)\.nc$", name)

        if not m:
            continue

        rows.append({
            "file": f,
            "gamma": float(m.group(1)),
            "center_time": center_time,
        })

    df = pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)

    print(f"Found {len(df)} forecast files in {input_dir}")
    print("Gammas:", sorted(df["gamma"].unique()))
    print("Center time:", CENTER_STR)

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
    """Format forecast lead time as e.g. 6h, 1d, 1d 6h."""
    hours = int(hours)
    days = hours // 24
    rem_hours = hours % 24

    if days == 0:
        return f"{rem_hours}h"
    if rem_hours == 0:
        return f"{days}d"
    return f"{days}d {rem_hours}h"


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


def load_mask_on_grid(target_time, target_da):
    """
    Load nearest ClimateNet mask for target_time and interpolate to forecast grid.
    Returns:
        mask_interp, mask_path, mask_time, diff_hours
    """
    mask_path = nearest_mask_file(target_time)

    mask_time = pd.Timestamp(os.path.basename(mask_path).replace(".nc", ""))
    diff_hours = abs(mask_time - pd.Timestamp(target_time)) / pd.Timedelta(hours=1)

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

    return mask_interp.astype(bool), mask_path, mask_time, diff_hours

def find_next_timestep_with_mask(ds_full, start_time_index, direction=1):
    """
    Starting from start_time_index, move through forecast timesteps until
    a ClimateNet mask within MAX_MASK_TIME_DIFFERENCE_HOURS is found.

    direction=1  -> search forward
    direction=-1 -> search backward
    """
    n_times = ds_full.sizes["time"]

    if start_time_index < 0:
        start_time_index = n_times + start_time_index

    indices = range(start_time_index, n_times) if direction == 1 else range(start_time_index, -1, -1)

    for t_idx in indices:
        valid_time = get_valid_time(ds_full, t_idx)

        try:
            # Just test whether a suitable mask exists
            mask_path = nearest_mask_file(valid_time)
            mask_time = pd.Timestamp(os.path.basename(mask_path).replace(".nc", ""))
            mask_diff_h = abs(mask_time - valid_time) / pd.Timedelta(hours=1)

            return t_idx, valid_time, mask_path, mask_time, mask_diff_h

        except Exception as e:
            print(f"[NO MASK] forecast step {t_idx}, valid_time={valid_time}: {e}")

    return None, None, None, None, None


# =====================
# PLOTTING
# =====================

def plot_dose_response(summary, metric, ylabel, out_name, out_dir, extra_title=None):
    plt.figure(figsize=(6, 4))

    for center_time, group in summary.groupby("center_time"):
        group = group.sort_values("gamma")
        plt.plot(group["gamma"], group[metric], marker="o", linewidth=2, label=str(center_time))

    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel("Gamma")
    plt.ylabel(ylabel)

    title = ylabel + " vs gamma"
    if extra_title is not None:
        title += "\n" + extra_title

    plt.title(title)
    plt.grid(True, alpha=0.3)

    if summary["center_time"].nunique() <= 6:
        plt.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

def plot_inside_outside_dose_response(
    summary,
    out_name,
    out_dir,
    extra_title=None,
):
    plt.figure(figsize=(7, 5))

    for center_time, group in summary.groupby("center_time"):
        group = group.sort_values("gamma")

        plt.plot(
            group["gamma"],
            group["delta_ivt_inside_mask_mean"],
            marker="o",
            linewidth=2,
            linestyle="-",
            label=f"Inside AR Mask",
        )

        plt.plot(
            group["gamma"],
            group["delta_ivt_outside_mask_mean"],
            marker="s",
            linewidth=2,
            linestyle="--",
            label=f"Outside AR Mask",
        )

    plt.axvline(0, color="black", linewidth=1)
    plt.axhline(0, color="grey", linewidth=0.8)

    plt.xlabel("Gamma")
    plt.ylabel("Mean ΔIVT")

    title = "Dose response of IVT"
    if extra_title is not None:
        title += "\n" + extra_title

    plt.title(title)

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


def plot_delta_wind10_map(delta_wind10, gamma, center_time, out_name, out_dir, lead_label=None):
    lat_name = get_lat_name(delta_wind10)
    lon_name = get_lon_name(delta_wind10)

    values = delta_wind10.values
    vmax = np.nanpercentile(np.abs(values), 99)

    if not np.isfinite(vmax) or vmax == 0:
        vmax = np.nanmax(np.abs(values))

    plt.figure(figsize=(10, 4.8))

    delta_wind10.plot(
        x=lon_name,
        y=lat_name,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        cbar_kwargs={"label": "Δ10 m wind speed [m/s]"},
    )

    title = f"Δ10 m wind: gamma={gamma:+.2f} minus gamma=0, {center_time}"
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

        lead_hours = int(
            pd.to_timedelta(delta.time.values[i]).total_seconds() / 3600
        )

        ax.set_title(
            f"ΔIVT: gamma={gamma:+.2f}, "
            f"{center_time}, T+{format_lead_time(lead_hours)}"
        )
        return []

    anim = FuncAnimation(fig, update, frames=delta.sizes["time"], interval=500)

    video_dir = os.path.join(OUT_DIR, "videos", "delta_ivt")
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

def evaluate_ar(time_selection):
    file_table = discover_files(INPUT_DIR)

    datasets = {}
    ivts = {}

    for _, row in file_table.iterrows():
        key = (row["center_time"], row["gamma"])
        ds = load_prediction(row["file"], time_selection=time_selection)
        datasets[key] = ds
        ivts[key] = compute_ivt(ds)

    for center_time in sorted(file_table["center_time"].unique()):
        center_time = pd.Timestamp(center_time)

        center_dir = os.path.join(
            BASE_DIR,
            center_time.strftime("%Y-%m-%dT%H"),
        )

        out_dir = os.path.join(OUT_DIR, f"lead_{time_selection}")

        os.makedirs(out_dir, exist_ok=True)

        records = []

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

        try:
            if time_selection == "first":
                time_index = 0
            elif time_selection == "last":
                time_index = -1


            control_ds_full = load_prediction(
                file_table[file_table["gamma"] == CONTROL_GAMMA]["file"].iloc[0],
                time_selection=None,
            )

            search_direction = 1 if time_selection == "first" else -1

            matched_time_index, valid_time, mask_path, mask_time, mask_diff_h = find_next_timestep_with_mask(
                control_ds_full,
                time_index,
                direction=search_direction,
            )

            if matched_time_index is None:
                raise ValueError(f"No usable mask found for {time_selection} lead.")

            # Reload prediction at the matched timestep, not necessarily first/last anymore
            for _, row in file_table.iterrows():
                key = (row["center_time"], row["gamma"])
                ds_full = load_prediction(row["file"], time_selection=None)
                ds_step = ds_full.isel(time=matched_time_index)
                datasets[key] = ds_step
                ivts[key] = compute_ivt(ds_step)

            control_ivt = ivts[control_key]

            ar_mask, mask_path, mask_time, mask_diff_h = load_mask_on_grid(
                valid_time,
                control_ivt,
            )

            print(
                f"Using forecast step {matched_time_index}: valid={valid_time}, "
                f"mask={mask_time}, diff={mask_diff_h:.1f} h"
            )
        except Exception as e:
            print(f"[WARN] Could not load mask for {center_time}: {e}")

        for gamma in available_gammas:
            key = (center_time, gamma)
            ds = datasets[key]
            ivt = ivts[key]
            delta_ivt = ivt - control_ivt

            #ar_like = ivt >= IVT_THRESHOLD
            #control_ar_like = control_ivt >= IVT_THRESHOLD

            rec = {
                "time_selection": time_selection,
                "center_time": str(center_time),
                "gamma": gamma,
                "matched_time_index": matched_time_index,
                "forecast_valid_time": str(valid_time),
                "mask_time": str(mask_time),
                "mask_time_diff_h": mask_diff_h,
                "file": file_table[
                    (file_table["center_time"] == center_time)
                    & (file_table["gamma"] == gamma)
                ]["file"].iloc[0],
                "mask_file": mask_path,
                "ivt_global_mean": area_weighted_mean(ivt),
                "ivt_global_max": max_value(ivt),
                "delta_ivt_global_mean": area_weighted_mean(delta_ivt),
                "delta_ivt_global_max_abs": max_value(abs(delta_ivt)),
                #"ar_like_area_fraction": area_fraction(ar_like),
                #"delta_ar_like_area_fraction": area_fraction(ar_like) - area_fraction(control_ar_like),
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
        

    summary = pd.DataFrame(records).sort_values(["center_time", "gamma"])

    summary_path = os.path.join(out_dir, f"gamma_summary_metrics_{time_selection}.csv")
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)
    print(summary)

    extra_title = (
        f"forecast valid: {valid_time.strftime('%Y-%m-%dT%H')} | "
        f"mask: {mask_time.strftime('%Y-%m-%dT%H')} "
        f"({mask_diff_h:.1f} h diff)"
    )

    plot_dose_response(
        summary,
        metric="delta_ivt_global_mean",
        ylabel=f"Global mean ΔIVT ({time_selection} lead)",
        out_name=f"dose_response_delta_ivt_global_mean_{time_selection}.png",
        out_dir=out_dir,
        extra_title = extra_title
    )

    if "delta_ivt_inside_mask_mean" in summary.columns:

        init_time = pd.Timestamp(CENTER_STR)
        lead_hours = int((valid_time - init_time) / pd.Timedelta(hours=1))

        extra_title = (
            f"Forecast: {valid_time:%Y-%m-%d %H:%M} "
            f"(start t + {format_lead_time(lead_hours)})\n"
            f"ClimateNet mask: {mask_time:%Y-%m-%d %H:%M} "
            f"(Δt = {mask_diff_h:.1f} h)"
        )
        plot_inside_outside_dose_response(
            summary,
            out_name=f"dose_response_inside_vs_outside_mask_{time_selection}.png",
            out_dir=out_dir,
            extra_title=extra_title,
        )

    if "delta_precip_inside_mask_mean" in summary.columns:
        plot_dose_response(
            summary,
            metric="delta_precip_inside_mask_mean",
            ylabel=f"Mean Δprecipitation inside AR mask ({time_selection} lead)",
            out_name=f"dose_response_delta_precip_inside_mask_mean_{time_selection}.png",
            out_dir=out_dir,
        )


def evaluate_tc(time_selection):
    file_table = discover_files(INPUT_DIR)

    if time_selection == "first":
        time_index = 0
    elif time_selection == "last":
        time_index = -1
    else:
        raise ValueError(time_selection)

    out_dir = os.path.join(OUT_DIR, f"lead_{time_selection}")
    os.makedirs(out_dir, exist_ok=True)

    records = []

    for center_time in sorted(file_table["center_time"].unique()):
        center_time = pd.Timestamp(center_time)

        group = file_table[file_table["center_time"] == center_time]
        available_gammas = sorted(group["gamma"].unique())

        control_file = group[group["gamma"] == CONTROL_GAMMA]["file"].iloc[0]
        control_full = load_prediction(control_file, time_selection=None)

        matched_time_index = (
            0 if time_selection == "first"
            else control_full.sizes["time"] - 1
        )

        valid_time = get_valid_time(control_full, matched_time_index)

        lead_hours = int((valid_time - center_time) / pd.Timedelta(hours=1))
        lead_label = f"T+{format_lead_time(lead_hours)}"
        control_ds = control_full.isel(time=time_index)
        control_wind10 = compute_10m_wind(control_ds)
        control_mslp = get_mslp(control_ds)

        control_max_wind = max_value(control_wind10)
        control_min_mslp = float(control_mslp.min(skipna=True).values)

        for gamma in available_gammas:
            forecast_file = group[group["gamma"] == gamma]["file"].iloc[0]

            ds_full = load_prediction(forecast_file, time_selection=None)
            ds = ds_full.isel(time=time_index)

            wind10 = compute_10m_wind(ds)
            mslp = get_mslp(ds)

            max_wind = max_value(wind10)
            min_mslp = float(mslp.min(skipna=True).values)

            records.append({
                "time_selection": time_selection,
                "center_time": str(center_time),
                "gamma": gamma,
                "matched_time_index": matched_time_index,
                "forecast_valid_time": str(valid_time),
                "file": forecast_file,
                "max_10m_wind": max_wind,
                "min_mslp_hpa": min_mslp,
                "delta_max_10m_wind": max_wind - control_max_wind,
                "delta_min_mslp_hpa": min_mslp - control_min_mslp,
            })

            if gamma != CONTROL_GAMMA:
                delta_wind10 = wind10 - control_wind10

                safe_time = str(center_time).replace(":", "").replace(" ", "T")
                out_name = (
                    f"delta_wind10_{time_selection}_"
                    f"gamma_{gamma:+.2f}_{safe_time}.png"
                )

                plot_delta_wind10_map(
                    delta_wind10,
                    gamma,
                    center_time,
                    out_name,
                    out_dir,
                    lead_label=lead_label,
                )

    summary = pd.DataFrame(records).sort_values(["center_time", "gamma"])

    summary_path = os.path.join(
        out_dir,
        f"gamma_summary_metrics_{time_selection}.csv",
    )
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)
    print(summary)

    plot_dose_response(
        summary,
        metric="delta_max_10m_wind",
        ylabel=f"Δ maximum 10 m wind [m/s] ({time_selection} lead)",
        out_name=f"dose_response_delta_max_10m_wind_{time_selection}.png",
        out_dir=out_dir,
    )

    plot_dose_response(
        summary,
        metric="delta_min_mslp_hpa",
        ylabel=f"Δ minimum MSLP [hPa] ({time_selection} lead)",
        out_name=f"dose_response_delta_min_mslp_{time_selection}.png",
        out_dir=out_dir,
    )


def plot_tc_intensity_trajectories():
    out_dir = os.path.join(OUT_DIR, "tc_intensity_trajectories")
    os.makedirs(out_dir, exist_ok=True)

    file_table = discover_files(INPUT_DIR)
    records = []

    plt.figure(figsize=(8, 5))

    for _, row in file_table.sort_values("gamma").iterrows():
        gamma = row["gamma"]
        ds = load_prediction(row["file"], time_selection=None)

        wind10 = compute_10m_wind(ds)
        mslp = get_mslp(ds)

        lead_hours = []
        max_winds = []
        min_mslps = []

        for t_idx in range(ds.sizes["time"]):
            lead_h = pd.to_timedelta(ds.time.values[t_idx]).total_seconds() / 3600.0

            wind_t = wind10.isel(time=t_idx)
            mslp_t = mslp.isel(time=t_idx)

            max_wind = float(wind_t.max(skipna=True).values)
            min_mslp = float(mslp_t.min(skipna=True).values)

            lead_hours.append(lead_h)
            max_winds.append(max_wind)
            min_mslps.append(min_mslp)

            records.append({
                "gamma": gamma,
                "lead_hours": lead_h,
                "lead_label": format_lead_time(lead_h),
                "forecast_valid_time": str(get_valid_time(ds, t_idx)),
                "max_10m_wind": max_wind,
                "min_mslp_hpa": min_mslp,
                "file": row["file"],
            })

        plt.plot(lead_hours, max_winds, marker="o", linewidth=2, label=f"γ={gamma:g}")

    plt.xlabel("Forecast lead time [hours]")
    plt.ylabel("Maximum 10 m wind [m/s]")
    plt.title(f"TC max 10 m wind trajectory ({CENTER_STR})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    wind_path = os.path.join(out_dir, "max_10m_wind_by_gamma.png")
    plt.savefig(wind_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", wind_path)

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "tc_intensity_trajectory_metrics.csv")
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    plt.figure(figsize=(8, 5))

    for gamma, group in df.groupby("gamma"):
        group = group.sort_values("lead_hours")
        plt.plot(
            group["lead_hours"],
            group["min_mslp_hpa"],
            marker="o",
            linewidth=2,
            label=f"γ={gamma:g}",
        )

    plt.xlabel("Forecast lead time [hours]")
    plt.ylabel("Minimum MSLP [hPa]")
    plt.title(f"TC minimum MSLP trajectory ({CENTER_STR})")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    mslp_path = os.path.join(out_dir, "min_mslp_by_gamma.png")
    plt.savefig(mslp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", mslp_path)

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
        #selected_gammas = select_video_gammas(group)
        selected_gammas = VIDEO_GAMMA_SELECTION

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

        lead_hours = int(
            pd.to_timedelta(ivt.time.values[i]).total_seconds() / 3600
        )

        ax.set_title(
            f"IVT trajectory: gamma={gamma:+.2f}, "
            f"{center_time}, T+{format_lead_time(lead_hours)}"
        )
        return []

    anim = FuncAnimation(fig, update, frames=ivt.sizes["time"], interval=500)


    video_dir = os.path.join(OUT_DIR, "videos", "ivt")
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

def plot_global_ivt_trajectories():
    """
    Plot the global mean IVT over the forecast trajectory for every gamma.

    Output:
        evaluation/ivt_trajectories/
            global_mean_ivt_by_gamma.png
            global_mean_ivt_by_gamma.csv
    """
    out_dir = os.path.join(OUT_DIR, "ivt_trajectories")
    os.makedirs(out_dir, exist_ok=True)

    file_table = discover_files(INPUT_DIR)

    records = []

    plt.figure(figsize=(8, 5))

    for _, row in file_table.sort_values("gamma").iterrows():
        gamma = row["gamma"]

        ds = load_prediction(row["file"], time_selection=None)
        ivt = compute_ivt(ds)

        if "time" not in ivt.dims:
            print(f"[SKIP] gamma={gamma}: no time dimension")
            continue

        values = []
        lead_hours = []

        for t_idx in range(ivt.sizes["time"]):

            ivt_t = ivt.isel(time=t_idx)

            mean_ivt = area_weighted_mean(ivt_t)

            values.append(mean_ivt)

            # GraphCast lead times are stored in nanoseconds
            lead_h = (
                pd.to_timedelta(ivt.time.values[t_idx]).total_seconds()
                / 3600.0
            )
            lead_hours.append(lead_h)

            records.append({
                "gamma": gamma,
                "lead_hours": lead_h,
                "global_mean_ivt": mean_ivt,
                "file": row["file"],
            })

        plt.plot(
            lead_hours,
            values,
            marker="o",
            linewidth=2,
            label=f"γ={gamma:g}",
        )

    plt.xlabel("Forecast lead time [hours]")
    plt.ylabel("Global mean IVT")
    plt.title(f"Global mean IVT trajectory ({CENTER_STR})")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Gamma", fontsize=8)

    plt.tight_layout()

    fig_path = os.path.join(out_dir, "global_mean_ivt_by_gamma.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    csv_path = os.path.join(out_dir, "global_mean_ivt_by_gamma.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)

    print("Saved:", fig_path)
    print("Saved:", csv_path)



def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for time_selection in TIME_SELECTIONS:
        if WEATHER_FEATURE == "AR":
            evaluate_ar(time_selection)
        elif WEATHER_FEATURE == "TC":
            evaluate_tc(time_selection)


    if WEATHER_FEATURE == "AR":
        print("\n[MAKING GAMMA TRAJECTORY PLOT]\n")
        plot_global_ivt_trajectories()

        if MAKE_TRAJECTORY_VIDEO:
            print("\n[MAKING TRAJECTORY VIDEOS]\n")
            make_all_videos()

            print("\n[MAKING IVT TRAJECTORY VIDEOS]\n")
            make_all_ivt_videos()


    if WEATHER_FEATURE == "TC":
        print("\n[MAKING TC INTENSITY TRAJECTORY PLOTS]\n")
        plot_tc_intensity_trajectories()

    print("[DONE]")


if __name__ == "__main__":
    main()