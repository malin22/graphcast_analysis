import os
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = "plots/malins_experiments/2021_regression/PCA/ridge/l5_nodes/pc_regression_physical_variables.csv"

OUT_DIR = "plots/malins_experiments/2021_regression/PCA/ridge/l5_nodes/figures"
os.makedirs(OUT_DIR, exist_ok=True)


SURFACE_VARIABLES = ["2t", "10u", "10v", "msl", "tp"]

ATMOSPHERIC_GROUPS = {
    "temperature": ["t50", "t250", "t500", "t600", "t700", "t850", "t1000"],
    "u_wind": ["u50", "u250", "u500", "u600", "u700", "u850", "u1000"],
    "v_wind": ["v50", "v250", "v500", "v600", "v700", "v850", "v1000"],
    "geopotential": ["z50", "z250", "z500", "z600", "z700", "z850", "z1000"],
    "specific_humidity": ["q50", "q250", "q500", "q600", "q700", "q850", "q1000"],
    "vertical_velocity": ["w50", "w250", "w500", "w600", "w700", "w850", "w1000"],
}


df = pd.read_csv(CSV_PATH)

if "n_pcs" not in df.columns and "n_features" in df.columns:
    df = df.rename(columns={"n_features": "n_pcs"})


def plot_r2_group(df, targets, title, filename):
    plot_df = df[df["target"].isin(targets)].copy()

    order = [t for t in targets if t in plot_df["target"].unique()]
    plot_df["target"] = pd.Categorical(
        plot_df["target"],
        categories=order,
        ordered=True,
    )
    plot_df = plot_df.sort_values(["target", "n_pcs"])

    plt.figure(figsize=(9, 5.5))

    for target, g in plot_df.groupby("target", observed=True):
        plt.plot(
            g["n_pcs"],
            g["r2_test"],
            marker="o",
            linewidth=2,
            label=str(target),
        )

    plt.xscale("log")
    plt.xticks(
        sorted(plot_df["n_pcs"].unique()),
        labels=[str(x) for x in sorted(plot_df["n_pcs"].unique())],
    )

    plt.xlabel("Number of PCs")
    plt.ylabel("Test R²")
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Target", ncol=3, fontsize=9)

    out_path = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


# Surface variables together
plot_r2_group(
    df,
    SURFACE_VARIABLES,
    "Decodability of surface ERA5 variables from GraphCast PCs",
    "r2_surface_variables.png",
)


# One plot per atmospheric variable, all levels together
for group_name, targets in ATMOSPHERIC_GROUPS.items():
    plot_r2_group(
        df,
        targets,
        f"Decodability of {group_name.replace('_', ' ')} across pressure levels",
        f"r2_{group_name}_levels.png",
    )