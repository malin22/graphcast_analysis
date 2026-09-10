import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BASE_PATH = (
    "results/malins_regression/"
    "PCA/linear/l6_nodes"
)

OUT_DIR = os.path.join(
    "malins_plots/regression/"
    "figures/for_report"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True,
)

SURFACE_CSV = os.path.join(
    BASE_PATH,
    "surface_variables_2019_2020train_2021test.csv",
)

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
# Pressure-level files
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


PRESSURE_LEVELS_TO_PLOT = [
    1000,
    850,
    700,
    600,
    500,
    250,
    50,
]


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
# Plot one pressure-level variable
# ============================================================

def plot_pressure_variable(
    df,
    variable,
):

    plot_df = df[
        df["level"].isin(
            PRESSURE_LEVELS_TO_PLOT
        )
    ].copy()

    plot_df = plot_df.sort_values(
        [
            "level",
            "n_pcs",
        ]
    )

    fig, ax = plt.subplots(
        figsize=(6, 4.0)
    )

    levels = sorted(
        plot_df["level"]
        .dropna()
        .unique()
    )

    for level in levels:

        g = plot_df[
            plot_df["level"] == level
        ].sort_values(
            "n_pcs"
        )

        ax.plot(
            g["n_pcs"],
            g["r2_test"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=f"{int(level)} hPa",
        )

    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    pc_counts = sorted(
        plot_df["n_pcs"]
        .dropna()
        .unique()
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xticks(
        pc_counts
    )

    ax.set_xticklabels(
        [
            str(int(x))
            for x in pc_counts
        ]
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Number of principal components"
    )

    ax.set_ylabel(
        r"Test $R^2$"
    )

    # No title:
    # use LaTeX subfigure captions instead

    ax.set_ylim(
        0,
        1,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    # Important:
    # no legend in individual plots

    fig.tight_layout()

    # --------------------------------------------------------
    # Save PNG as well
    # --------------------------------------------------------

    png_path = os.path.join(
        OUT_DIR,
        f"r2_{variable}.png",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {png_path}"
    )


# ============================================================
# Standalone shared legend
# ============================================================

def save_pressure_legend():

    fig, ax = plt.subplots(
        figsize=(9, 0.8)
    )

    # Dummy lines used only to create the legend
    for level in PRESSURE_LEVELS_TO_PLOT:

        ax.plot(
            [],
            [],
            marker="o",
            markersize=4,
            linewidth=1.4,
            label=f"{level} hPa",
        )

    ax.axis(
        "off"
    )

    legend = ax.legend(
        loc="center",
        ncol=len(
            PRESSURE_LEVELS_TO_PLOT
        ),
        frameon=False,
        title="Pressure level",
        fontsize=9,
        title_fontsize=10,
        handlelength=2.0,
        columnspacing=1.2,
    )

    # --------------------------------------------------------
    # Save legend
    # --------------------------------------------------------


    png_path = os.path.join(
        OUT_DIR,
        "r2_pressure_legend.png",
    )


    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {png_path}"
    )


# ============================================================
# Plot surface variables
# ============================================================

def plot_surface_variables(df):

    plot_df = df[
        df["target"].isin(
            SURFACE_VARIABLES
        )
    ].copy()

    fig, ax = plt.subplots(
        figsize=(6.5, 4.0)
    )

    for target in SURFACE_VARIABLES:

        g = plot_df[
            plot_df["target"] == target
        ].sort_values(
            "n_pcs"
        )

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

    pc_counts = sorted(
        plot_df["n_pcs"]
        .dropna()
        .unique()
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xticks(
        pc_counts
    )

    ax.set_xticklabels(
        [
            str(int(x))
            for x in pc_counts
        ]
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "Number of principal components"
    )

    ax.set_ylabel(
        r"Test $R^2$"
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    # --------------------------------------------------------
    # Surface-variable legend
    # --------------------------------------------------------

    ax.legend(
        fontsize=9,
        title_fontsize=10,
        frameon=False,
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    png_path = os.path.join(
        OUT_DIR,
        "r2_surface_variables.png",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved: {png_path}"
    )

# ============================================================
# Main
# ============================================================

def main():

    for variable, filename in (
        PRESSURE_FILES.items()
    ):

        csv_path = os.path.join(
            BASE_PATH,
            filename,
        )

        if not os.path.exists(
            csv_path
        ):
            print(
                f"Skipping missing file: "
                f"{csv_path}"
            )
            continue

        df = load_results(
            csv_path
        )

        plot_pressure_variable(
            df,
            variable,
        )

    save_pressure_legend()


    surface_df = load_results(SURFACE_CSV)
    plot_surface_variables(surface_df)


if __name__ == "__main__":
    main()