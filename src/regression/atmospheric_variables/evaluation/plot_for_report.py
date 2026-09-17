import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps

from matplotlib.ticker import NullLocator




# ============================================================
# Paths
# ============================================================

REGRESSION_TYPE = "ridge"

REPORT_PC_COUNTS = [
    1,
    5,
    10,
    25,
    50,
    100,
    200,
    512,
]

BASE_PATH = (
    f"results/regression/atmospheric_variables/"
    f"PCA/{REGRESSION_TYPE}/l6_nodes"
)

OUT_DIR = os.path.join(
    "plots/regression/"
    f"{REGRESSION_TYPE}/for_report_new_new"
)

os.makedirs(OUT_DIR, exist_ok=True)

SURFACE_CSV = os.path.join(
    BASE_PATH,
    "surface_variables_2019_2020train_2021test.csv",
)


# ============================================================
# Surface variables
# ============================================================

SURFACE_VARIABLES = [
    "2t",
    "10u",
    "10v",
    "msl",
    "tp",
]

SURFACE_LABELS = {
    "2t": "2m temperature",
    "10u": "10m zonal wind",
    "10v": "10m meridional wind",
    "msl": "Mean sea-level pressure",
    "tp": "Total precipitation",
}


# ============================================================
# Pressure-level variables
# ============================================================

PRESSURE_FILES = {
    "temperature": (
        "temperature_all_pressure_levels_"
        "2019_2020train_2021test.csv"
    ),
    "u_component_of_wind": (
        "u_component_of_wind_all_pressure_levels_"
        "2019_2020train_2021test.csv"
    ),
    "v_component_of_wind": (
        "v_component_of_wind_all_pressure_levels_"
        "2019_2020train_2021test.csv"
    ),
    "geopotential": (
        "geopotential_all_pressure_levels_"
        "2019_2020train_2021test.csv"
    ),
    "specific_humidity": (
        "specific_humidity_all_pressure_levels_"
        "2019_2020train_2021test.csv"
    ),
    "vertical_velocity": (
        "vertical_velocity_all_pressure_levels_"
        "2019_2020train_2021test.csv"
    ),
}


# Titles shown above the individual panels
PRESSURE_TITLES = {
    "temperature": "Temperature",
    "geopotential": "Geopotential",
    "v_component_of_wind": "Meridional wind",
    "u_component_of_wind": "Zonal wind",
    "specific_humidity": "Specific humidity",
    "vertical_velocity": "Vertical velocity",
}


# Keep the current 3 x 2 arrangement
PRESSURE_LAYOUT = [
    ["temperature", "geopotential"],
    ["v_component_of_wind", "u_component_of_wind"],
    ["specific_humidity", "vertical_velocity"],
]


PRESSURE_LEVELS_TO_PLOT = [
    1000,
    850,
    700,
    500,
    250,
    100,
    50,
]


# ============================================================
# Colors
# ============================================================

cmap = colormaps["viridis"]

PRESSURE_COLORS = {
    level: cmap(x)
    for level, x in zip(
        PRESSURE_LEVELS_TO_PLOT,
        np.linspace(
            0.15,
            0.90,
            len(PRESSURE_LEVELS_TO_PLOT),
        ),
    )
}


# ============================================================
# Loading
# ============================================================

def load_results(path):

    df = pd.read_csv(path)

    if (
        "n_pcs" not in df.columns
        and "n_features" in df.columns
    ):
        df = df.rename(
            columns={
                "n_features": "n_pcs"
            }
        )

    return df


# ============================================================
# Load all pressure-level data
# ============================================================

def load_pressure_results():

    results = {}

    for variable, filename in PRESSURE_FILES.items():

        csv_path = os.path.join(
            BASE_PATH,
            filename,
        )

        if not os.path.exists(csv_path):
            print(
                f"Skipping missing file: "
                f"{csv_path}"
            )
            continue

        results[variable] = load_results(
            csv_path
        )

    return results


# ============================================================
# Draw one pressure-level variable onto an existing axis
# ============================================================

def draw_pressure_variable(
    ax,
    df,
    variable,
):

    plot_df = df[
        df["level"].isin(PRESSURE_LEVELS_TO_PLOT)
    ].copy()

    plot_df = plot_df[
        plot_df["n_pcs"].isin(REPORT_PC_COUNTS)
    ].copy()

    plot_df = plot_df.sort_values(
        ["level", "n_pcs"]
    )

    # Plot in the explicit pressure-level order so that
    # colors and legend order are consistent.
    for level in PRESSURE_LEVELS_TO_PLOT:

        g = plot_df[
            plot_df["level"] == level
        ].sort_values("n_pcs")

        if g.empty:
            continue

        ax.plot(
            g["n_pcs"],
            g["r2_test"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=PRESSURE_COLORS[level],
            label=f"{level} hPa",
        )

    ax.set_xscale("log")

    ax.set_ylim(
        0,
        1,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    # Variable name above the plot.
    # No (b), (c), ... subfigure captions.
    ax.set_title(
        PRESSURE_TITLES[variable],
        fontsize=10,
        fontweight="normal",
        pad=5,
    )


# ============================================================
# Combined 3 x 2 pressure-level figure
# ============================================================

def plot_pressure_variables(results):

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(9.0, 8.2),
        sharex=True,
        sharey=True,
    )

    for row_idx, row in enumerate(
        PRESSURE_LAYOUT
    ):
        for col_idx, variable in enumerate(
            row
        ):

            ax = axes[
                row_idx,
                col_idx,
            ]

            if variable not in results:
                ax.axis("off")
                continue

            draw_pressure_variable(
                ax,
                results[variable],
                variable,
            )

    # Use only the PC counts reported in the figure.
    # All axes share x, so this gives every panel identical locations.
    for ax in axes.flat:
        ax.set_xticks(REPORT_PC_COUNTS)
        ax.xaxis.set_minor_locator(NullLocator())

    # Only the bottom row displays the tick labels.
    for ax in axes[-1, :]:
        ax.set_xticklabels([str(x) for x in REPORT_PC_COUNTS])


    # --------------------------------------------------------
    # Shared-axis appearance
    # --------------------------------------------------------

    # Because sharex=True, only the bottom row needs x ticks.
    for ax in axes[:-1, :].flat:
        ax.tick_params(
            axis="x",
            labelbottom=False,
        )

    # Because sharey=True, only the left column needs y ticks.
    for ax in axes[:, 1]:
        ax.tick_params(
            axis="y",
            labelleft=False,
        )

    # Remove individual axis labels.
    for ax in axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")

    # One shared label for the entire six-panel block.
    fig.supxlabel(
        "Number of principal components",
        fontsize=11,
        y=0.035,
    )

    fig.supylabel(
        r"Test $R^2$",
        fontsize=11,
        x=0.035,
    )

    # --------------------------------------------------------
    # One shared pressure-level legend
    # --------------------------------------------------------

    legend_handles = []

    for level in PRESSURE_LEVELS_TO_PLOT:

        handle, = axes[0, 0].plot(
            [],
            [],
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=PRESSURE_COLORS[level],
            label=f"{level} hPa",
        )

        legend_handles.append(handle)

    fig.legend(
        handles=legend_handles,
        labels=[
            f"{level} hPa"
            for level in PRESSURE_LEVELS_TO_PLOT
        ],
        loc="upper center",
        ncol=len(PRESSURE_LEVELS_TO_PLOT),
        frameon=False,
        title="Pressure level",
        fontsize=9,
        title_fontsize=10,
        handlelength=1.8,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 0.975),
    )

    # --------------------------------------------------------
    # Spacing
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.10,
        top=0.88,
        wspace=0.10,
        hspace=0.22,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    png_path = os.path.join(
        OUT_DIR,
        "r2_pressure_variables.png",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {png_path}"
    )


# ============================================================
# Plot surface variables
# ============================================================
def plot_surface_variables(df):

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    plot_df = df[
        df["target"].isin(SURFACE_VARIABLES)
    ].copy()

    plot_df = plot_df[
        plot_df["n_pcs"].isin(REPORT_PC_COUNTS)
    ].copy()

    for target in SURFACE_VARIABLES:

        g = plot_df[
            plot_df["target"] == target
        ].sort_values("n_pcs")

        if g.empty:
            continue

        ax.plot(
            g["n_pcs"],
            g["r2_test"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=SURFACE_LABELS[target],
        )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xscale("log")

    # Show only the PC counts included in the report
    ax.set_xticks(REPORT_PC_COUNTS)
    ax.set_xticklabels(
        [str(x) for x in REPORT_PC_COUNTS]
    )

    # Remove automatic minor ticks from the logarithmic axis
    ax.xaxis.set_minor_locator(NullLocator())

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Number of principal components"
    )

    ax.set_ylabel(
        r"Test $R^2$"
    )

    ax.set_ylim(0, 1)

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.set_title(
        "Surface variables",
        fontsize=10,
        fontweight="normal",
        pad=5,
    )

    ax.legend(
        fontsize=9,
        frameon=False,
    )

    fig.tight_layout()

    png_path = os.path.join(
        OUT_DIR,
        "r2_surface_variables.png",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {png_path}")

# ============================================================
# Main
# ============================================================

def main():

    pressure_results = (
        load_pressure_results()
    )

    plot_pressure_variables(
        pressure_results
    )

    surface_df = load_results(
        SURFACE_CSV
    )

    plot_surface_variables(
        surface_df
    )


if __name__ == "__main__":
    main()