"""
Compare AR perturbation controls.

Plots global mean IVT trajectories for:
    1. Correct event-aligned AR probe, gamma = 0.5
    2. Shuffled-day / permuted-mask probe, gamma = 0.5
    3. Random-label probe, gamma = 0.5
    4. Unperturbed baseline, gamma = 0

All trajectories use the same initialization time.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from evaluation_helpers import (
    area_weighted_mean,
    discover_files,
    get_valid_time,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

WEATHER_FEATURE = "AR"
CENTER_STR = "2021-07-18T00"
NODE_HIERARCHY_LEVEL = 6

PERTURBATION_GAMMA = 0.5
CONTROL_GAMMA = 0.0

Q_VAR = "specific_humidity"
U_VAR = "u_component_of_wind"
V_VAR = "v_component_of_wind"
G = 9.80665


cmap = plt.get_cmap("Set2", 4)

# -----------------------------------------------------------------------------
# Experiment directories
# -----------------------------------------------------------------------------

BASE_ROOT = os.path.join(
    "results",
    "perturbation",
    WEATHER_FEATURE,
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}",
)

EXPERIMENTS = {
    "Correct AR probe": "raw_activations",
    "Shuffled timesteps": "permuted_masks_raw_activations",
    "Random labels": "random_baseline_raw_activations",
}


OUT_DIR = os.path.join(
    "plots",
    "perturbation",
    WEATHER_FEATURE,
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}",
    "control_comparison",
    CENTER_STR,
)

os.makedirs(OUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# Plot settings
# -----------------------------------------------------------------------------

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
# IVT
# -----------------------------------------------------------------------------

def compute_ivt(ds):
    """Compute IVT magnitude integrated over 200--1000 hPa."""

    q = ds[Q_VAR]
    u = ds[U_VAR]
    v = ds[V_VAR]

    if "level" not in q.dims:
        raise ValueError(
            "Expected pressure-level variables with dimension 'level'."
        )

    levels_hpa = ds["level"].values.astype(float)

    keep = (
        (levels_hpa >= 200.0)
        & (levels_hpa <= 1000.0)
    )

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

    ivt_u = (
        np.trapezoid(
            (q * u).values,
            x=p,
            axis=axis,
        )
        / G
    )

    ivt_v = (
        np.trapezoid(
            (q * v).values,
            x=p,
            axis=axis,
        )
        / G
    )

    dims = [
        d for d in q.dims
        if d != "level"
    ]

    coords = {
        d: q.coords[d]
        for d in dims
        if d in q.coords
    }

    return xr.DataArray(
        np.sqrt(ivt_u**2 + ivt_v**2),
        dims=dims,
        coords=coords,
        name="ivt",
    )


def open_forecast(path):
    ds = xr.open_dataset(path)

    if "batch" in ds.dims:
        ds = ds.isel(batch=0)

    return ds


# -----------------------------------------------------------------------------
# Locate forecast
# -----------------------------------------------------------------------------

def find_forecast(activation_type, gamma):

    input_dir = os.path.join(
        BASE_ROOT,
        activation_type,
        CENTER_STR,
        "data",
    )

    print()
    print("----------------------------------------")
    print("Experiment:", activation_type)
    print("Directory:", input_dir)

    file_table = discover_files(
        input_dir,
        CENTER_STR,
    )

    print(
        "Available gammas:",
        sorted(file_table["gamma"].unique()),
    )

    hit = file_table[
        np.isclose(
            file_table["gamma"].astype(float),
            gamma,
        )
    ]

    if hit.empty:
        raise ValueError(
            f"No gamma={gamma:g} forecast found in "
            f"{input_dir}"
        )

    path = hit.iloc[0]["file"]

    print(
        f"Using gamma={gamma:g}:",
        path,
    )

    return path


# -----------------------------------------------------------------------------
# Compute one trajectory
# -----------------------------------------------------------------------------

def compute_trajectory(path):

    ds = open_forecast(path)

    try:
        ivt = compute_ivt(ds)

        values = []
        lead_hours = []

        for t_idx in range(ds.sizes["time"]):

            valid_time = get_valid_time(
                ds,
                t_idx,
                CENTER_STR,
            )

            lead_h = float(
                (
                    valid_time
                    - pd.Timestamp(CENTER_STR)
                )
                / pd.Timedelta(hours=1)
            )

            values.append(
                area_weighted_mean(
                    ivt.isel(time=t_idx)
                )
            )

            lead_hours.append(lead_h)

    finally:
        ds.close()

    return (
        np.asarray(lead_hours),
        np.asarray(values),
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():

    trajectories = {}

    # ---------------------------------------------------------
    # Perturbed control experiments: gamma = +0.5
    # ---------------------------------------------------------

    for label, activation_type in EXPERIMENTS.items():

        path = find_forecast(
            activation_type,
            PERTURBATION_GAMMA,
        )

        lead_hours, values = compute_trajectory(path)

        trajectories[label] = (
            lead_hours,
            values,
        )

    # ---------------------------------------------------------
    # Baseline: gamma = 0
    #
    # Use the baseline from the correct AR experiment.
    # ---------------------------------------------------------

    correct_activation_type = EXPERIMENTS[
        "Correct AR probe"
    ]

    baseline_path = find_forecast(
        correct_activation_type,
        CONTROL_GAMMA,
    )

    baseline_leads, baseline_values = (
        compute_trajectory(baseline_path)
    )

    # ---------------------------------------------------------
    # Sanity check: lead times should be identical
    # ---------------------------------------------------------

    for label, (lead_hours, _) in trajectories.items():

        if not np.allclose(
            lead_hours,
            baseline_leads,
        ):
            raise ValueError(
                f"Lead times for '{label}' do not "
                "match the baseline."
            )

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(5.4, 3.5),
        constrained_layout=True,
    )

    markers = {
        "Correct AR probe": "o",
        "Shuffled timesteps": "o",
        "Random labels": "o",
    }

    linestyles = {
        "Correct AR probe": "-",
        "Shuffled timesteps": "-",
        "Random labels": "-",
    }

    colors = {
        "Correct AR probe": cmap(0), 
        "Shuffled timesteps": cmap(1),
        "Random labels": cmap(2),}

    for label, (lead_hours, values) in trajectories.items():

        ax.plot(
            lead_hours,
            values,
            marker=markers[label],
            markersize=3.2,
            linewidth=1.8,
            linestyle=linestyles[label],
            label=rf"{label} ($\gamma=0.5$)",
            color=colors[label],
        )

    # Baseline last so it stays visible.
    ax.plot(
        baseline_leads,
        baseline_values,
        color="black",
        linewidth=2.2,
        linestyle="--",
        label=r"$\gamma=0$ (baseline)",
        zorder=10,
    )

    ax.set_xlabel(
        "Forecast lead time [h]"
    )

    ax.set_ylabel(
        r"Global mean IVT "
        r"[kg m$^{-1}$ s$^{-1}$]"
    )

    ax.grid(alpha=0.2)

    ax.legend(
        frameon=False,
        fontsize=7.5,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_path = os.path.join(
        OUT_DIR,
        "ar_control_global_ivt_trajectory.png",
    )

    fig.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print("Saved:", output_path)

    # ---------------------------------------------------------
    # Also save underlying numbers
    # ---------------------------------------------------------

    records = []

    for label, (lead_hours, values) in trajectories.items():

        for lead_h, value in zip(
            lead_hours,
            values,
        ):
            records.append({
                "experiment": label,
                "gamma": PERTURBATION_GAMMA,
                "lead_hours": lead_h,
                "global_mean_ivt": value,
            })

    for lead_h, value in zip(
        baseline_leads,
        baseline_values,
    ):
        records.append({
            "experiment": "Baseline",
            "gamma": CONTROL_GAMMA,
            "lead_hours": lead_h,
            "global_mean_ivt": value,
        })

    csv_path = os.path.join(
        OUT_DIR,
        "ar_control_global_ivt_trajectory.csv",
    )

    pd.DataFrame(records).to_csv(
        csv_path,
        index=False,
    )

    print("Saved:", csv_path)
    print("[DONE]")


if __name__ == "__main__":
    main()