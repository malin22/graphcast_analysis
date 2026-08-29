import os
import re
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt




def load_activations(path: str) -> np.ndarray:
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    if x.ndim != 2:
        raise ValueError(f"Expected [nodes, features], got shape {x.shape}")

    return x.astype(np.float32)


def extract_date_from_filename(path):
    fname = os.path.basename(path)
    match = re.search(r"\d{4}-\d{2}-\d{2}", fname)
    if match:
        return match.group(0)
    return fname


def main():
    ACTS_DIR = "/share/prj-4d/graphcast_shared/data/graphcast_activation_2021"
    PCA_DIR = "/share/prj-4d/graphcast_shared/data/pca_components"

    P_COMPONENTS = os.path.join(PCA_DIR, "pca_components_2021.npy")
    P_MEAN = os.path.join(PCA_DIR, "pca_mean_2021.npy")

    OUT_DIR = "plots/malins_experiments/2021_pc_temporal_analysis"
    os.makedirs(OUT_DIR, exist_ok=True)

    N_TOP_PCS = 5



    pca_components = np.load(P_COMPONENTS)
    pca_mean = np.load(P_MEAN)

    pca_components = pca_components[:N_TOP_PCS]

    npy_files = sorted(glob(os.path.join(ACTS_DIR, "*.npy")))

    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {ACTS_DIR}")

    rows = []

    for i, f in enumerate(npy_files, start=1):
        date = extract_date_from_filename(f)

        X = load_activations(f)

        if i % 10 == 0 :
            print(f"[{i}/{len(npy_files)}] Processing {os.path.basename(f)} ", flush=True,)

        if np.isnan(X).any():
            print(f"WARNING: skipping {os.path.basename(f)} because it contains NaNs")
            continue

        scores = (X - pca_mean) @ pca_components.T
        # scores shape: [nodes, PCs]

        row = {"date": date, "file": os.path.basename(f)}

        for pc_idx in range(N_TOP_PCS):
            pc_name = f"PC{pc_idx + 1}"
            pc_scores = scores[:, pc_idx]

            row[f"{pc_name}_mean"] = np.mean(pc_scores)
            row[f"{pc_name}_std"] = np.std(pc_scores)
            row[f"{pc_name}_min"] = np.min(pc_scores)
            row[f"{pc_name}_max"] = np.max(pc_scores)
            row[f"{pc_name}_abs_mean"] = np.mean(np.abs(pc_scores))

        rows.append(row)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")

    csv_path = os.path.join(OUT_DIR, "daily_pc_temporal_summary.csv")
    df.to_csv(csv_path, index=False)

    print(f"Saved daily PC summary to:\n{csv_path}")

    # -----------------------------
    # Plot 1: daily global mean PC score
    # -----------------------------
    plt.figure(figsize=(12, 5))

    for pc in range(1, N_TOP_PCS + 1):
        plt.plot(df["date"], df[f"PC{pc}_mean"], label=f"PC{pc}")

    plt.axhline(0, linewidth=1)
    plt.xlabel("Date")
    plt.ylabel("Mean PC score across nodes")
    plt.title("Daily global mean PC activation")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "daily_pc_mean_timeseries.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Plot 2: daily spatial variability
    # -----------------------------
    plt.figure(figsize=(12, 5))

    for pc in range(1, N_TOP_PCS + 1):
        plt.plot(df["date"], df[f"PC{pc}_std"], label=f"PC{pc}")

    plt.xlabel("Date")
    plt.ylabel("Spatial std of PC scores")
    plt.title("Daily spatial variability of PC activations")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "daily_pc_spatial_std_timeseries.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


    # -----------------------------
    # Individual plots per PC
    # -----------------------------
    individual_dir = os.path.join(OUT_DIR, "individual_pcs")
    os.makedirs(individual_dir, exist_ok=True)

    for pc in range(1, N_TOP_PCS + 1):
        # Mean activation
        plt.figure(figsize=(12, 4))
        plt.plot(df["date"], df[f"PC{pc}_mean"], linewidth=1.5)
        plt.axhline(0, linewidth=1)
        plt.xlabel("Date")
        plt.ylabel("Mean PC score across nodes")
        plt.title(f"PC{pc}: Daily global mean activation")
        plt.tight_layout()

        out_path = os.path.join(
            individual_dir,
            f"PC{pc}_daily_mean_activation.png",
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Spatial std
        plt.figure(figsize=(12, 4))
        plt.plot(df["date"], df[f"PC{pc}_std"], linewidth=1.5)
        plt.xlabel("Date")
        plt.ylabel("Spatial std of PC scores")
        plt.title(f"PC{pc}: Daily spatial variability")
        plt.tight_layout()

        out_path = os.path.join(
            individual_dir,
            f"PC{pc}_daily_spatial_std.png",
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Optional: abs mean
        plt.figure(figsize=(12, 4))
        plt.plot(df["date"], df[f"PC{pc}_abs_mean"], linewidth=1.5)
        plt.xlabel("Date")
        plt.ylabel("Mean absolute PC score")
        plt.title(f"PC{pc}: Daily mean absolute activation")
        plt.tight_layout()

        out_path = os.path.join(
            individual_dir,
            f"PC{pc}_daily_abs_mean_activation.png",
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

   

    print(f"Saved temporal plots to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()