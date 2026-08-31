"""Atmospheric-river evaluation for GraphCast perturbation experiments."""

import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from evaluation_helpers import (
    area_weighted_mean, discover_files, find_next_timestep_with_mask,
    format_lead_time, gamma_colors, get_lat_name, get_lon_name, get_valid_time,
    load_era5_at_time, load_mask_on_grid, load_prediction, max_value,
)

WEATHER_FEATURE = "AR"
THRESHOLD = 0.9
ACTIVATION_TYPE="raw_activations"
CENTER_STR = "2021-02-12T18"
NODE_HIERARCHY_LEVEL = 6
CONTROL_GAMMA = 0.0
MAX_MASK_TIME_DIFFERENCE_HOURS = 3
TIME_SELECTIONS = ["first", "last"]
MAKE_DELTA_IVT_MAPS = True
MAKE_TRAJECTORY_VIDEO = True
VIDEO_GAMMA_SELECTION = [-0.5, 0.5]
VIDEO_FPS = 2
VIDEO_FORMAT = "mp4"
VIDEO_FRAME_STRIDE = 1

Q_VAR = "specific_humidity"
U_VAR = "u_component_of_wind"
V_VAR = "v_component_of_wind"
TP_VAR = "total_precipitation_6hr"
G = 9.80665
ERA5_DAILY_DIR = "/share/prj-4d/graphcast_shared/data/era5_daily_nc"

BASE_DIR = os.path.join(
    "results", "perturbation", WEATHER_FEATURE,
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}", ACTIVATION_TYPE, CENTER_STR,
)
INPUT_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "evaluation")
MASK_DIR = f"/share/prj-4d/graphcast_shared/data/ClimateNetLarge/{WEATHER_FEATURE}_labels_cleaned"

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

def evaluate_ar(time_selection):
    file_table = discover_files(INPUT_DIR, CENTER_STR)

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
                control_ds_full, time_index, CENTER_STR, MASK_DIR,
                MAX_MASK_TIME_DIFFERENCE_HOURS, direction=search_direction,
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
                valid_time, control_ivt, MASK_DIR, MAX_MASK_TIME_DIFFERENCE_HOURS
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

def select_video_gammas(group):
    """Return requested video gammas that actually exist, plus the control."""
    available = set(group["gamma"].unique())
    requested = set(VIDEO_GAMMA_SELECTION) | {CONTROL_GAMMA}
    return sorted(available & requested)

def make_all_videos():
    file_table = discover_files(INPUT_DIR, CENTER_STR)

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
    file_table = discover_files(INPUT_DIR, CENTER_STR)

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

def extract_era5_global_ivt_trajectory(forecast_ds):
    """
    Calculate global mean ERA5 IVT at every forecast valid time.

    forecast_ds is used only to obtain the valid times and lead times.
    """
    records = []

    for t_idx in range(forecast_ds.sizes["time"]):
        valid_time = get_valid_time(
            forecast_ds,
            t_idx,
            CENTER_STR
        )

        era5_step, era5_file = load_era5_at_time(
            valid_time, ERA5_DAILY_DIR
        )

        era5_ivt = compute_ivt(era5_step)

        global_mean_ivt = area_weighted_mean(
            era5_ivt
        )

        lead_h = (
            pd.to_timedelta(
                forecast_ds.time.values[t_idx]
            ).total_seconds()
            / 3600.0
        )

        records.append({
            "source": "ERA5",
            "lead_hours": lead_h,
            "forecast_valid_time": str(valid_time),
            "global_mean_ivt": global_mean_ivt,
            "era5_file": era5_file,
        })

    return pd.DataFrame(records)

def plot_global_ivt_trajectories():
    """
    Plot global mean IVT over the forecast trajectory for every gamma
    and for ERA5 truth.

    Output:
        evaluation/ivt_trajectories/
            global_mean_ivt_by_gamma.png
            global_mean_ivt_by_gamma.csv
            global_mean_ivt_era5.csv
    """
    out_dir = os.path.join(
        OUT_DIR,
        "ivt_trajectories",
    )
    os.makedirs(out_dir, exist_ok=True)

    file_table = discover_files(INPUT_DIR, CENTER_STR)

    if CONTROL_GAMMA not in file_table["gamma"].values:
        raise ValueError(
            f"No control gamma={CONTROL_GAMMA} found."
        )

    gammas = sorted(file_table["gamma"].unique())
    colors, cmap, norm = gamma_colors(gammas)

    records = []

    fig, ax = plt.subplots(figsize=(8, 5))

    # =====================
    # GAMMA TRAJECTORIES
    # =====================

    for _, row in file_table.sort_values("gamma").iterrows():
        gamma = row["gamma"]

        ds = load_prediction(
            row["file"],
            time_selection=None,
        )

        ivt = compute_ivt(ds)

        if "time" not in ivt.dims:
            print(
                f"[SKIP] gamma={gamma}: no time dimension"
            )
            continue

        values = []
        lead_hours = []

        for t_idx in range(ivt.sizes["time"]):
            ivt_t = ivt.isel(time=t_idx)

            mean_ivt = area_weighted_mean(ivt_t)

            lead_h = (
                pd.to_timedelta(
                    ivt.time.values[t_idx]
                ).total_seconds()
                / 3600.0
            )

            values.append(mean_ivt)
            lead_hours.append(lead_h)

            records.append({
                "source": "GraphCast",
                "gamma": gamma,
                "lead_hours": lead_h,
                "forecast_valid_time": str(
                    get_valid_time(ds, t_idx, CENTER_STR)
                ),
                "global_mean_ivt": mean_ivt,
                "file": row["file"],
            })

        ax.plot(
            lead_hours,
            values,
            marker="o",
            linewidth=2,
            color=colors[gamma],
            label=f"γ={gamma:g}",
        )

    # =====================
    # ERA5 TRAJECTORY
    # =====================

    control_file = file_table.loc[
        file_table["gamma"] == CONTROL_GAMMA,
        "file",
    ].iloc[0]

    control_ds = load_prediction(
        control_file,
        time_selection=None,
    )

    era5_trajectory = (
        extract_era5_global_ivt_trajectory(
            control_ds
        )
    )

    ax.plot(
        era5_trajectory["lead_hours"],
        era5_trajectory["global_mean_ivt"],
        color="black",
        linestyle="--",
        marker="o",
        markersize=4,
        linewidth=3,
        label="ERA5",
        zorder=10,
    )

    # =====================
    # FORMATTING
    # =====================

    ax.set_xlabel("Forecast lead time [hours]")
    ax.set_ylabel("Global mean IVT")
    ax.set_title(
        f"Global mean IVT trajectory ({CENTER_STR})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(title="Trajectory", fontsize=8)

    fig.tight_layout()

    fig_path = os.path.join(
        out_dir,
        "global_mean_ivt_by_gamma.png",
    )
    fig.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    graphcast_csv_path = os.path.join(
        out_dir,
        "global_mean_ivt_by_gamma.csv",
    )
    pd.DataFrame(records).to_csv(
        graphcast_csv_path,
        index=False,
    )

    era5_csv_path = os.path.join(
        out_dir,
        "global_mean_ivt_era5.csv",
    )
    era5_trajectory.to_csv(
        era5_csv_path,
        index=False,
    )

    print("Saved:", fig_path)
    print("Saved:", graphcast_csv_path)
    print("Saved:", era5_csv_path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for time_selection in TIME_SELECTIONS:
        evaluate_ar(time_selection)
    print("\n[MAKING GAMMA TRAJECTORY PLOT]\n")
    plot_global_ivt_trajectories()
    if MAKE_TRAJECTORY_VIDEO:
        print("\n[MAKING TRAJECTORY VIDEOS]\n")
        make_all_videos()
        print("\n[MAKING IVT TRAJECTORY VIDEOS]\n")
        make_all_ivt_videos()
    print("[DONE]")


if __name__ == "__main__":
    main()
