import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = "plots/malins_experiments/2021_pc_spatial_pattern_stability"
OUT_DIR_DATA = os.path.join(OUT_DIR, "data")

MODES = {
    "z_scored": os.path.join(OUT_DIR, "z_scored"),
    "dot_product": os.path.join(OUT_DIR, "dot_product"),
}

FIRST_N = 5


def plot_first_n(mode, out_dir_mode, first_n=5):
    plt.figure(figsize=(10, 5))

    for pc in range(1, first_n + 1):
        path = os.path.join(
            OUT_DIR_DATA,
            f"{mode}_PC{pc}_lagged_spatial_similarity.csv",
        )

        if not os.path.exists(path):
            print(f"Missing file, skipping: {path}")
            continue

        lag_df = pd.read_csv(path)

        plt.plot(
            lag_df["lag_days"],
            lag_df["mean_similarity"],
            label=f"PC{pc}",
            linewidth=2,
        )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Temporal lag [days]")

    if mode == "dot_product":
        plt.ylabel("Mean dot product")
        title = "Temporal persistence of dot-product similarity (first 5 PCs)"
    else:
        plt.ylabel("Mean spatial correlation")
        title = "Temporal persistence of spatial patterns (first 5 PCs)"

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(
        out_dir_mode,
        f"{mode}_first5_lagged_spatial_similarity.png",
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


def main():
    for mode, out_dir_mode in MODES.items():
        os.makedirs(out_dir_mode, exist_ok=True)
        plot_first_n(mode, out_dir_mode, first_n=FIRST_N)


if __name__ == "__main__":
    main()