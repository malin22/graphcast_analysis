import os
import re
import time
from glob import glob

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Loading / preprocessing
# -----------------------------
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


def extract_date_from_filename(path: str) -> str:
    fname = os.path.basename(path)
    match = re.search(r"\d{4}-\d{2}-\d{2}", fname)
    if match:
        return match.group(0)
    return fname


def zscore_nodes(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


# -----------------------------
# Similarity functions
# -----------------------------
def compute_similarity_matrix(maps, mode: str) -> np.ndarray:
    """
    maps: list of arrays, each [nodes]
    mode:
        - "z_scored": Pearson correlation of z-scored maps
        - "dot_product": raw dot product of raw maps, scaled by number of nodes and normalized per PC
    """
    M = np.stack(maps, axis=0).astype(np.float32)

    if mode == "z_scored":
        # Maps are already z-scored across nodes.
        # Equivalent to Pearson correlation.
        sim = (M @ M.T) / M.shape[1]
        sim = np.clip(sim, -1, 1)

    elif mode == "dot_product":

        # Normalize whole PC by its global scale across time/nodes
        global_scale = np.std(M) + 1e-8

        M_normed = M / global_scale

        # Dot product keeps:
        # - spatial structure
        # - temporal magnitude changes
        # but now comparable across PCs
        sim = (M_normed @ M_normed.T) / M.shape[1]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return sim


def compute_lagged_similarity(sim: np.ndarray, timesteps_per_day: int) -> pd.DataFrame:
    n_timesteps = sim.shape[0]

    rows = []
    for lag in range(1, n_timesteps):
        vals = np.diag(sim, k=lag)

        rows.append({
            "lag_timesteps": lag,
            "lag_days": lag / timesteps_per_day,
            "mean_similarity": np.mean(vals),
            "std_similarity": np.std(vals),
        })

    return pd.DataFrame(rows)


def get_lag_value(lag_df: pd.DataFrame, lag_timesteps: int) -> float:
    vals = lag_df.loc[
        lag_df["lag_timesteps"] == lag_timesteps,
        "mean_similarity",
    ].values

    if len(vals) == 0:
        return np.nan

    return float(vals[0])


def summarize_similarity(
    sim: np.ndarray,
    lag_df: pd.DataFrame,
    pc_number: int,
    timesteps_per_day: int,
    mode: str,
) -> dict:
    n_timesteps = sim.shape[0]
    off_diag = sim[~np.eye(n_timesteps, dtype=bool)]

    return {
        "mode": mode,
        "PC": pc_number,
        "mean_offdiagonal_similarity": np.mean(off_diag),
        "std_offdiagonal_similarity": np.std(off_diag),
        "one_step_similarity": get_lag_value(lag_df, 1),
        "one_day_similarity": get_lag_value(lag_df, timesteps_per_day),
        "seven_day_similarity": get_lag_value(lag_df, 7 * timesteps_per_day),
        "thirty_day_similarity": get_lag_value(lag_df, 30 * timesteps_per_day),
        "ninety_day_similarity": get_lag_value(lag_df, 90 * timesteps_per_day),
        "one_year_similarity": get_lag_value(lag_df, 365 * timesteps_per_day),
        "min_similarity": np.min(off_diag),
        "max_similarity": np.max(off_diag),
    }


# -----------------------------
# Plotting / saving
# -----------------------------
def save_similarity_matrix(
    sim: np.ndarray,
    timestep_labels,
    out_dir_data: str,
    pc_number: int,
    mode: str,
):
    npy_path = os.path.join(
        out_dir_data,
        f"{mode}_PC{pc_number}_spatial_similarity.npy",
    )
    np.save(npy_path, sim)

    csv_path = os.path.join(
        out_dir_data,
        f"{mode}_PC{pc_number}_spatial_similarity.csv",
    )

    sim_df = pd.DataFrame(
        sim,
        index=timestep_labels,
        columns=timestep_labels,
    )
    sim_df.to_csv(csv_path)

    return npy_path, csv_path


def save_heatmap(
    sim: np.ndarray,
    plot_dates: pd.Series,
    out_dir: str,
    pc_number: int,
    mode: str,
):
    n_timesteps = sim.shape[0]

    plt.figure(figsize=(8, 7))
    if mode == "dot_product":
        vmax = np.percentile(np.abs(sim), 99)

        im = plt.imshow(
            sim,
            vmin=-vmax,
            vmax=vmax,
            cmap="coolwarm",
            aspect="auto",
        )

    else:
        im = plt.imshow(
            sim,
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            aspect="auto",
        )
    plt.colorbar(im, label="Spatial similarity")

    tick_idx = np.linspace(0, n_timesteps - 1, 8, dtype=int)
    tick_labels = plot_dates.strftime("%Y-%m-%d")[tick_idx]

    plt.xticks(tick_idx, tick_labels, rotation=45, ha="right")
    plt.yticks(tick_idx, tick_labels)

    if mode == "z_scored":
        title = "pattern-only similarity"
    elif mode == "dot_product":
        title = "dot-product similarity"
    else:
        title = mode

    plt.title(f"PC{pc_number}: {title}")
    plt.xlabel("Timestep")
    plt.ylabel("Timestep")
    plt.tight_layout()

    path = os.path.join(
        out_dir,
        f"{mode}_PC{pc_number}_spatial_similarity_heatmap.png",
    )
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return path


def save_lagged_similarity(
    lag_df: pd.DataFrame,
    out_dir_data: str,
    pc_number: int,
    mode: str,
):
    path = os.path.join(
        out_dir_data,
        f"{mode}_PC{pc_number}_lagged_spatial_similarity.csv",
    )
    lag_df.to_csv(path, index=False)
    return path





def save_combined_lag_plot(
    lag_dfs_by_pc: dict,
    out_dir: str,
    mode: str,
):
    plt.figure(figsize=(10, 5))

    for pc_number, lag_df in lag_dfs_by_pc.items():
        plt.plot(
            lag_df["lag_days"],
            lag_df["mean_similarity"],
            label=f"PC{pc_number}",
            linewidth=2,
        )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Temporal lag [days]")

    if mode == "dot_product":
        y_label = "Mean dot product"
    else:
        y_label = "Mean spatial correlation"
    plt.ylabel(y_label)

    if mode == "z_scored":
        title = "Temporal persistence of spatial patterns"
    elif mode == "dot_product":
        title = "Temporal persistence of dot-product similarity"
    else:
        title = f"Temporal persistence: {mode}"

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(
        out_dir,
        f"{mode}_all_pcs_lagged_spatial_similarity.png",
    )
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return path






def save_lag_plot(
    lag_df: pd.DataFrame,
    out_dir: str,
    pc_number: int,
    mode: str,
):
    plt.figure(figsize=(8, 4))

    plt.plot(
        lag_df["lag_days"],
        lag_df["mean_similarity"],
        linewidth=2,
    )

    plt.fill_between(
        lag_df["lag_days"],
        lag_df["mean_similarity"] - lag_df["std_similarity"],
        lag_df["mean_similarity"] + lag_df["std_similarity"],
        alpha=0.2,
    )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Temporal lag [days]")
    plt.ylabel("Mean spatial similarity")

    if mode == "z_scored":
        title = "pattern-only persistence"
    elif mode == "dot_product":
        title = "dot-product persistence"
    else:
        title = mode

    plt.title(f"PC{pc_number}: {title}")
    plt.tight_layout()

    path = os.path.join(
        out_dir,
        f"{mode}_PC{pc_number}_lagged_spatial_similarity.png",
    )
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    return path



def main():
    ACTS_DIR = "/share/prj-4d/graphcast_shared/data/graphcast_activation_2021"
    PCA_DIR = "/share/prj-4d/graphcast_shared/data/pca_components"

    P_COMPONENTS = os.path.join(PCA_DIR, "pca_components_2021.npy")
    P_MEAN = os.path.join(PCA_DIR, "pca_mean_2021.npy")

    OUT_DIR = "plots/malins_experiments/2021_pc_spatial_pattern_stability"
    OUT_DIR_DATA = os.path.join(OUT_DIR, "data")
    OUT_DIR_Z = os.path.join(OUT_DIR, "z_scored")
    OUT_DIR_RAW = os.path.join(OUT_DIR, "dot_product")


    for d in [OUT_DIR, OUT_DIR_DATA, OUT_DIR_Z, OUT_DIR_RAW]:
        os.makedirs(d, exist_ok=True)

    N_TOP_PCS = 10
    TIMESTEPS_PER_DAY = 4

    modes = {
        "z_scored": OUT_DIR_Z,
        "dot_product": OUT_DIR_RAW,
    }

    pca_components = np.load(P_COMPONENTS)[:N_TOP_PCS]
    pca_mean = np.load(P_MEAN)

    npy_files = sorted(glob(os.path.join(ACTS_DIR, "*.npy")))

    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {ACTS_DIR}")

    timestep_labels = []
    plot_dates = []

    pc_maps_z = [[] for _ in range(N_TOP_PCS)]
    pc_maps_raw = [[] for _ in range(N_TOP_PCS)]


    pc_norms = [[] for _ in range(N_TOP_PCS)]
    pc_stds = [[] for _ in range(N_TOP_PCS)]

    print(f"Found {len(npy_files)} activation files")


    for i, f in enumerate(npy_files, start=1):
        date = extract_date_from_filename(f)

        if i % 10 == 0 :
            print(f"[{i}/{len(npy_files)}] Processing {os.path.basename(f)} ", flush=True,)

        X = load_activations(f)

        if np.isnan(X).any():
            print("    WARNING: skipping file because it contains NaNs", flush=True)
            continue

        scores = (X - pca_mean) @ pca_components.T

        timestep_labels.append(f"{date}_t{i:04d}")
        plot_dates.append(date)

        for pc_idx in range(N_TOP_PCS):
            spatial_map = scores[:, pc_idx]

            # raw maps
            pc_maps_raw[pc_idx].append(spatial_map.astype(np.float32))

            # z-scored maps
            spatial_map_z = zscore_nodes(spatial_map)
            pc_maps_z[pc_idx].append(spatial_map_z.astype(np.float32))


    plot_dates = pd.to_datetime(plot_dates)
    n_timesteps = len(timestep_labels)

    print(f"\nValid timesteps: {n_timesteps}")
    print(f"Approx valid days: {n_timesteps / TIMESTEPS_PER_DAY:.1f}")

    if n_timesteps < 2:
        raise ValueError("Need at least two valid timesteps for similarity analysis")



    print(f"Saved norm plot to:\n{norm_plot_path}")
    print(f"Saved spatial std plot to:\n{std_plot_path}")
    summary_rows = []

    map_sets = {
        "z_scored": pc_maps_z,
        "dot_product": pc_maps_raw,
    }

    for mode, pc_maps in map_sets.items():
        print(f"\n=== Running mode: {mode} ===")

        out_dir_mode = modes[mode]
        lag_dfs_by_pc = {}

        for pc_idx in range(N_TOP_PCS):
            pc_number = pc_idx + 1
            print(f"Computing PC{pc_number} ({mode})", flush=True)

            sim = compute_similarity_matrix(
                maps=pc_maps[pc_idx],
                mode=mode,
            )

            save_similarity_matrix(
                sim=sim,
                timestep_labels=timestep_labels,
                out_dir_data=OUT_DIR_DATA,
                pc_number=pc_number,
                mode=mode,
            )

            save_heatmap(
                sim=sim,
                plot_dates=plot_dates,
                out_dir=out_dir_mode,
                pc_number=pc_number,
                mode=mode,
            )



            lag_df = compute_lagged_similarity(
                sim=sim,
                timesteps_per_day=TIMESTEPS_PER_DAY,
            )

            lag_dfs_by_pc[pc_number] = lag_df

            save_lagged_similarity(
                lag_df=lag_df,
                out_dir_data=OUT_DIR_DATA,
                pc_number=pc_number,
                mode=mode,
            )

            save_lag_plot(
                lag_df=lag_df,
                out_dir=out_dir_mode,
                pc_number=pc_number,
                mode=mode,
            )


            summary_rows.append(
                summarize_similarity(
                    sim=sim,
                    lag_df=lag_df,
                    pc_number=pc_number,
                    timesteps_per_day=TIMESTEPS_PER_DAY,
                    mode=mode,
                )
            )

        combined_path = save_combined_lag_plot(
            lag_dfs_by_pc=lag_dfs_by_pc,
            out_dir=out_dir_mode,
            mode=mode,
        )

        print(f"Saved combined lag plot:\n{combined_path}")

    summary_df = pd.DataFrame(summary_rows)

    summary_path = os.path.join(
        OUT_DIR,
        "pc_spatial_pattern_stability_summary.csv",
    )
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Spatial pattern stability summary ===")
    print(summary_df)

    print(f"\nSaved all outputs to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()