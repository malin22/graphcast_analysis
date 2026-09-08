import os
import pandas as pd
import matplotlib.pyplot as plt


BASE_PATH = (
    "results/malins_regression/"
    "PCA/linear/l6_nodes"
)



OUT_DIR = os.path.join(
    "malins_plots/regression/"
    "figures",
)
os.makedirs(OUT_DIR, exist_ok=True)


SURFACE_CSV = os.path.join(
    BASE_PATH,
    "surface_variables_2019_2020train_2021test.csv",
)


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


PLOT_TITLES = {
    "temperature": "Temperature",
    "u_component_of_wind": "Zonal wind",
    "v_component_of_wind": "Meridional wind",
    "geopotential": "Geopotential",
    "specific_humidity": "Specific humidity",
    "vertical_velocity": "Vertical velocity",
}


SURFACE_VARIABLES = [
    "2t",
    "10u",
    "10v",
    "msl",
    "tp",
]

PRESSURE_LEVELS_TO_PLOT = [
    1000,
    850,
    700,
    600,
    500,
    250,
    50
]


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


def setup_pc_axis(df):
    pc_counts = sorted(
        df["n_pcs"].unique()
    )

    plt.xscale("log")

    plt.xticks(
        pc_counts,
        labels=[
            str(x)
            for x in pc_counts
        ],
    )


def plot_surface_variables(df):

    plot_df = df[
        df["target"].isin(
            SURFACE_VARIABLES
        )
    ].copy()

    plt.figure(
        figsize=(9, 5.5)
    )

    for target in SURFACE_VARIABLES:

        g = plot_df[
            plot_df["target"] == target
        ].sort_values(
            "n_pcs"
        )

        if g.empty:
            continue

        plt.plot(
            g["n_pcs"],
            g["r2_test"],
            marker="o",
            linewidth=2,
            label=target,
        )

    setup_pc_axis(
        plot_df
    )

    plt.xlabel(
        "Number of PCs"
    )

    plt.ylabel(
        "Test R²"
    )

    plt.title(
        "Decodability of surface ERA5 variables "
        "from GraphCast PCs"
    )

    plt.ylim(
        0,
        1,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend(
        title="Variable",
        fontsize=9,
    )

    out_path = os.path.join(
        OUT_DIR,
        "r2_surface_variables.png",
    )

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {out_path}"
    )


def plot_pressure_variable(
    df,
    variable,
):

    plot_df = df[
        df["level"].isin(PRESSURE_LEVELS_TO_PLOT)
    ].copy()

    plot_df = plot_df.sort_values(
        [
            "level",
            "n_pcs",
        ]
    )

    plt.figure(
        figsize=(11, 7)
    )

    levels = sorted(
        plot_df["level"].dropna().unique()
    )


    for level in levels:

        g = plot_df[
            plot_df["level"] == level
        ].sort_values(
            "n_pcs"
        )

        plt.plot(
            g["n_pcs"],
            g["r2_test"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=f"{int(level)} hPa",
        )

    setup_pc_axis(
        plot_df
    )

    plt.xlabel(
        "Number of PCs"
    )

    plt.ylabel(
        "Test R²"
    )

    plt.title(
        f"Decodability of "
        f"{PLOT_TITLES[variable]} "
        f"from GraphCast PCs"
    )

    plt.ylim(
        0,
        1,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend(
        title="Pressure level",
        ncol=4,
        fontsize=7,
        title_fontsize=8,
        bbox_to_anchor=(
            1.02,
            1,
        ),
        loc="upper left",
    )

    out_path = os.path.join(
        OUT_DIR,
        f"r2_{variable}_levels.png",
    )

    plt.tight_layout()

    plt.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {out_path}"
    )


# --------------------------------------------------
# Surface variables
# --------------------------------------------------

if os.path.exists(
    SURFACE_CSV
):
    surface_df = load_results(
        SURFACE_CSV
    )

    plot_surface_variables(
        surface_df
    )

else:
    print(
        f"Skipping missing file: "
        f"{SURFACE_CSV}"
    )


# --------------------------------------------------
# Pressure-level variables
# --------------------------------------------------

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