import os
import re
from glob import glob

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
ACTS_DIR = "/share/prj-4d/graphcast_shared/data/graphcast_activation_2021"

PCA_DIR = "/share/prj-4d/graphcast_shared/data/pca_components"
P_COMPONENTS = os.path.join(PCA_DIR, "pca_components_2021.npy")
P_MEAN = os.path.join(PCA_DIR, "pca_mean_2021.npy")

OUT_DIR = "plots/malins_experiments/pc_dynamic_behaviour"
os.makedirs(OUT_DIR, exist_ok=True)

N_TOP_PCS = 20
TIMESTEPS_PER_DAY = 4

# lags to evaluate
LAGS_DAYS = [1, 3, 7, 14, 30, 60, 90, 120, 180, 270, 363]


# =========================================================
# HELPERS
# =========================================================
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


def zscore_nodes(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


# =========================================================
# SIMILARITY
# =========================================================
def similarity_zscored(a, b):
    """
    Pearson correlation of spatial maps.
    Maps already z-scored across nodes.
    """
    return np.mean(a * b)


def similarity_dot(a, b):
    """
    Dot product normalized by average map energy.
    Preserves magnitude differences.
    More comparable across PCs.
    """

    dot = np.mean(a * b)

    norm_a = np.sqrt(np.mean(a ** 2))
    norm_b = np.sqrt(np.mean(b ** 2))

    denom = 0.5 * (norm_a**2 + norm_b**2) + 1e-8

    return dot / denom


# =========================================================
# LAG ANALYSIS
# =========================================================
def compute_lag_curve(maps, lag_timesteps_list, mode):
    """
    maps shape:
        [timesteps, nodes]
    """

    n_timesteps = maps.shape[0]

    rows = []

    for lag in lag_timesteps_list:

        vals = []

        for t in range(n_timesteps - lag):

            a = maps[t]
            b = maps[t + lag]

            if mode == "z_scored":
                sim = similarity_zscored(a, b)

            elif mode == "dot_product":
                sim = similarity_dot(a, b)

            else:
                raise ValueError(mode)

            vals.append(sim)

        vals = np.asarray(vals)

        rows.append(
            {
                "lag_timesteps": lag,
                "lag_days": lag / TIMESTEPS_PER_DAY,
                "mean_similarity": np.mean(vals),
                "std_similarity": np.std(vals),
            }
        )

    return pd.DataFrame(rows)


# =========================================================
# PERIODICITY
# =========================================================
def estimate_annual_periodicity(signal_1d):
    """
    FFT-based seasonal strength estimate.
    """

    x = signal_1d - np.mean(signal_1d)

    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2

    freqs = np.fft.rfftfreq(len(x), d=1.0)

    if len(freqs) < 2:
        return np.nan

    target_freq = 1 / 365.0

    idx = np.argmin(np.abs(freqs - target_freq))

    annual_power = power[idx]
    total_power = np.sum(power[1:]) + 1e-8

    return annual_power / total_power


# =========================================================
# PLOTTING
# =========================================================
def plot_top_ranked_pcs(
    lag_curves,
    ranking_df,
    ranking_column,
    mode,
    top_k=5,
):
    top_df = ranking_df.sort_values(
        ranking_column,
        ascending=False,
    ).head(top_k)

    plt.figure(figsize=(10, 5))

    for _, row in top_df.iterrows():

        pc = int(row["PC"])

        lag_df = lag_curves[pc]

        plt.plot(
            lag_df["lag_days"],
            lag_df["mean_similarity"],
            linewidth=2,
            label=f"PC{pc}",
        )

    plt.axhline(0, linewidth=1)

    plt.xlabel("Lag [days]")

    if mode == "z_scored":
        plt.ylabel("Spatial correlation")
    else:
        plt.ylabel("Normalized dot product")

    plt.title(f"Top {top_k} PCs by {ranking_column}")

    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(
        OUT_DIR,
        f"{mode}_top_{top_k}_{ranking_column}.png",
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


# =========================================================
# MAIN
# =========================================================
def main():

    print("Loading PCA...")

    pca_components = np.load(P_COMPONENTS)[:N_TOP_PCS]
    pca_mean = np.load(P_MEAN)

    npy_files = sorted(glob(os.path.join(ACTS_DIR, "*.npy")))

    if len(npy_files) == 0:
        raise RuntimeError("No files found")

    print(f"Found {len(npy_files)} activation files")

    # -----------------------------------------------------
    # Load projected PC maps
    # -----------------------------------------------------
    pc_maps_raw = [[] for _ in range(N_TOP_PCS)]
    pc_maps_z = [[] for _ in range(N_TOP_PCS)]

    dates = []

    for i, path in enumerate(npy_files):

        if i % 20 == 0:
            print(f"[{i}/{len(npy_files)}]")

        X = load_activations(path)

        if np.isnan(X).any():
            continue

        scores = (X - pca_mean) @ pca_components.T

        dates.append(extract_date_from_filename(path))

        for pc_idx in range(N_TOP_PCS):

            spatial_map = scores[:, pc_idx].astype(np.float32)

            pc_maps_raw[pc_idx].append(spatial_map)

            spatial_map_z = zscore_nodes(spatial_map)

            pc_maps_z[pc_idx].append(
                spatial_map_z.astype(np.float32)
            )

    dates = pd.to_datetime(dates)

    # -----------------------------------------------------
    # Convert to arrays
    # -----------------------------------------------------
    pc_maps_raw = [
        np.stack(x, axis=0)
        for x in pc_maps_raw
    ]

    pc_maps_z = [
        np.stack(x, axis=0)
        for x in pc_maps_z
    ]

    lag_timesteps_list = [
        d * TIMESTEPS_PER_DAY
        for d in LAGS_DAYS
    ]

    # =====================================================
    # RUN MODES
    # =====================================================
    for mode, maps_all_pcs in [
        ("z_scored", pc_maps_z),
        ("dot_product", pc_maps_raw),
    ]:

        print(f"\n=== MODE: {mode} ===")

        rows = []

        lag_curves = {}

        for pc_idx in range(N_TOP_PCS):

            pc = pc_idx + 1

            print(f"PC{pc}")

            maps = maps_all_pcs[pc_idx]

            # -------------------------------------------------
            # Lag curve
            # -------------------------------------------------
            lag_df = compute_lag_curve(
                maps=maps,
                lag_timesteps_list=lag_timesteps_list,
                mode=mode,
            )

            lag_curves[pc] = lag_df

            # save lag df
            lag_path = os.path.join(
                OUT_DIR,
                f"{mode}_PC{pc}_lag_curve.csv",
            )

            lag_df.to_csv(lag_path, index=False)

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------
            lag_lookup = {
                int(r["lag_days"]): r["mean_similarity"]
                for _, r in lag_df.iterrows()
            }

            sim_1d = lag_lookup.get(1, np.nan)
            sim_7d = lag_lookup.get(7, np.nan)
            sim_30d = lag_lookup.get(30, np.nan)
            sim_180d = lag_lookup.get(180, np.nan)
            sim_365d = lag_lookup.get(365, np.nan)

            stability_score = np.nanmean(
                [sim_1d, sim_7d, sim_30d]
            )

            change_score = sim_1d - sim_30d

            seasonal_score = sim_365d - sim_180d

            # -------------------------------------------------
            # periodicity estimate
            # -------------------------------------------------
            abs_mean_signal = np.mean(
                np.abs(maps),
                axis=1,
            )

            annual_periodicity = estimate_annual_periodicity(
                abs_mean_signal
            )

            rows.append(
                {
                    "PC": pc,
                    "mode": mode,

                    "sim_1d": sim_1d,
                    "sim_7d": sim_7d,
                    "sim_30d": sim_30d,
                    "sim_180d": sim_180d,
                    "sim_365d": sim_365d,

                    "stability_score": stability_score,
                    "change_score": change_score,
                    "seasonal_score": seasonal_score,

                    "annual_periodicity": annual_periodicity,
                }
            )

        ranking_df = pd.DataFrame(rows)

        # -----------------------------------------------------
        # Save rankings
        # -----------------------------------------------------
        ranking_path = os.path.join(
            OUT_DIR,
            f"{mode}_pc_rankings.csv",
        )

        ranking_df.to_csv(ranking_path, index=False)

        print(f"Saved rankings:\n{ranking_path}")

        # -----------------------------------------------------
        # Plot top PCs
        # -----------------------------------------------------
        plot_top_ranked_pcs(
            lag_curves,
            ranking_df,
            ranking_column="seasonal_score",
            mode=mode,
            top_k=5,
        )

        plot_top_ranked_pcs(
            lag_curves,
            ranking_df,
            ranking_column="change_score",
            mode=mode,
            top_k=5,
        )

        plot_top_ranked_pcs(
            lag_curves,
            ranking_df,
            ranking_column="stability_score",
            mode=mode,
            top_k=5,
        )


if __name__ == "__main__":
    main()