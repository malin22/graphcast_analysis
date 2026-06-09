import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================
# CONFIG
# =====================

BASE_PATH = "plots/malins_experiments/2021_pc_physical_variable_regression/derived_targets/"

CSV_PATH = os.path.join(BASE_PATH, "pc_regression_physical_variables.csv")

OUT_DIR = os.path.join(BASE_PATH, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD = 0.8

VARIABLE_ORDER = [
    "windspeed850",
    "rh850",
    "thickness500_850",
]


# =====================
# HELPERS
# =====================

def load_results(csv_path):
    df = pd.read_csv(csv_path)

    required = {"target", "n_features", "r2_test", "rmse_test", "corr_test"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["target"] = pd.Categorical(
        df["target"],
        categories=VARIABLE_ORDER,
        ordered=True,
    )
    df = df.sort_values(["target", "n_features"])

    return df


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# =====================
# PLOTS
# =====================

def plot_r2_curves(df):
    plt.figure(figsize=(8, 5))

    for target, g in df.groupby("target", observed=True):
        if pd.isna(target):
            continue
        plt.plot(
            g["n_features"],
            g["r2_test"],
            marker="o",
            linewidth=2,
            label=str(target),
        )

    plt.xscale("log")
    plt.xticks(
        sorted(df["n_features"].unique()),
        labels=[str(x) for x in sorted(df["n_features"].unique())],
    )
    plt.xlabel("Number of PCs")
    plt.ylabel("Test R²")
    plt.title("Physical-variable decodability from GraphCast latent PCs")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Target", ncol=2)
    savefig("r2_vs_number_of_pcs.png")




def plot_r2_heatmap(df):
    pivot = df.pivot(index="target", columns="n_features", values="r2_test")
    pivot = pivot.loc[[v for v in VARIABLE_ORDER if v in pivot.index]]

    plt.figure(figsize=(8, 4.5))
    im = plt.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)

    plt.colorbar(im, label="Test R²")

    plt.yticks(
        ticks=np.arange(len(pivot.index)),
        labels=pivot.index,
    )
    plt.xticks(
        ticks=np.arange(len(pivot.columns)),
        labels=pivot.columns,
    )

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]
            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.xlabel("Number of PCs")
    plt.ylabel("Target variable")
    plt.title("Decodability heatmap")
    savefig("r2_heatmap.png")


def plot_pcs_needed_for_threshold(df):
    rows = []

    for target, g in df.groupby("target", observed=True):
        if pd.isna(target):
            continue

        g = g.sort_values("n_features")
        max_r2 = g["r2_test"].max()
        threshold_value = THRESHOLD * max_r2

        reached = g[g["r2_test"] >= threshold_value]

        if len(reached) == 0:
            pcs_needed = np.nan
        else:
            pcs_needed = int(reached.iloc[0]["n_features"])

        rows.append({
            "target": str(target),
            "max_r2": max_r2,
            "threshold_r2": threshold_value,
            "pcs_needed": pcs_needed,
        })

    summary = pd.DataFrame(rows)
    summary["target"] = pd.Categorical(
        summary["target"],
        categories=VARIABLE_ORDER,
        ordered=True,
    )
    summary = summary.sort_values("target")

    summary_path = os.path.join(
        OUT_DIR,
        f"pcs_needed_for_{int(THRESHOLD * 100)}pct_max_r2.csv"
    )
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    plt.figure(figsize=(7, 4.5))
    plt.bar(summary["target"].astype(str), summary["pcs_needed"])

    plt.xlabel("Target variable")
    plt.ylabel(f"PCs needed for {int(THRESHOLD * 100)}% of max R²")
    plt.title("Effective PC dimensionality by physical variable")
    plt.grid(axis="y", alpha=0.3)

    savefig(f"pcs_needed_for_{int(THRESHOLD * 100)}pct_max_r2.png")


def main():
    df = load_results(CSV_PATH)

    plot_r2_curves(df)
    #plot_r2_heatmap(df)
    #plot_pcs_needed_for_threshold(df)

    print("\nDone.")


if __name__ == "__main__":
    main()