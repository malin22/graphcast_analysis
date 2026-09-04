import os

import numpy as np
import pandas as pd

from malins_regression.run_logistic_probe import (
    run_logistic_experiment,
)


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

WEATHER_FEATURE = "TC"  # "AR" or "TC"

FINE_MESH_LEVEL = 6
NODE_HIERARCHY_LEVEL = 6

TOP_K_COUNTS = [
    1,
    2,
    3,
    5,
    8,
    10,
    15,
    20,
    25,
    30,
    40,
    50,
    75,
    100,
    150,
    200,
    250,
    300,
    400,
    512,
]

LABEL_MODE = "intersection"
MAX_TIME_DIFFERENCE_HOURS = 3

THRESHOLDS = [0.1, 0.2, 0.3, 0.5]


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

TRAIN_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2020-11-01")

VAL_START = pd.Timestamp("2020-11-01")
VAL_END = pd.Timestamp("2021-01-01")

TEST_START = pd.Timestamp("2021-01-01")
TEST_END = pd.Timestamp("2022-01-01")


# ============================================================
# PCA DATA PATHS
# ============================================================

PC_SCORES_PATHS = {
    2019: (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2019_from_2019_2020_pca_per_timestep.npy"
    ),
    2020: (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2020_from_2019_2020_pca_per_timestep.npy"
    ),
    2021: (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2021_from_2019_2020_pca_per_timestep.npy"
    ),
}

TIMESTEP_FILES_TXTS = {
    2019: (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2019_from_2019_2020_pca_per_timestep_files.txt"
    ),
    2020: (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2020_from_2019_2020_pca_per_timestep_files.txt"
    ),
    2021: (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2021_from_2019_2020_pca_per_timestep_files.txt"
    ),
}


MASK_DIR = (
    f"/share/prj-4d/graphcast_shared/data/"
    f"ClimateNetLarge/{WEATHER_FEATURE}_labels_cleaned"
)


# ============================================================
# FULL-PC PROBE USED FOR RANKING
# ============================================================

# This must point to the probe direction produced by the FIRST-K PCA
# experiment for k=512.
#
# Ranking is based on the standardized logistic-regression coefficients:
#
#     importance_j = abs(coef_z[j])
#
# The selected-PC experiment then trains NEW probes using only the
# top-k ranked PCs. The 512-PC probe is used only to define the ranking.

BASE_PROBE_PATH = (
    f"results/logistic_regression/"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"first_k_pcs/"
    f"probe_direction_{WEATHER_FEATURE}_"
    f"first_k_pcs_{LABEL_MODE}_"
    f"M{NODE_HIERARCHY_LEVEL}_"
    f"512_features_"
    f"2019_2020_train_only.npz"
)


OUT_DIR = (
    f"results/logistic_regression/"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"selected_pcs_after_coefs/"
)


# ============================================================
# PC RANKING
# ============================================================

def load_pc_ranking(probe_path):
    """
    Load the full-PC logistic probe and rank PCs by |coef_z|.

    coef_z is the coefficient vector in standardized feature space.
    Ranking by |coef_z| therefore measures probe relevance under the
    exact standardized L2 logistic model used in the baseline probe.
    """
    if not os.path.exists(probe_path):
        raise FileNotFoundError(
            "Could not find the full-PC probe used for ranking:\n"
            f"{probe_path}\n\n"
            "Run experiment_first_pcs.py with k=512 first."
        )

    with np.load(probe_path, allow_pickle=True) as probe:
        if "coef_z" not in probe:
            raise KeyError(
                f"{probe_path} does not contain 'coef_z'. "
                f"Available keys: {list(probe.keys())}"
            )

        coef_z = np.asarray(
            probe["coef_z"],
            dtype=np.float64,
        ).reshape(-1)

    if coef_z.size < max(TOP_K_COUNTS):
        raise ValueError(
            f"Probe contains only {coef_z.size} coefficients, "
            f"but TOP_K_COUNTS requests up to {max(TOP_K_COUNTS)} PCs."
        )

    pc_ranking = np.argsort(
        np.abs(coef_z)
    )[::-1]

    return pc_ranking, coef_z


# ============================================================
# RUN
# ============================================================

def main():
    pc_ranking, coef_z = load_pc_ranking(
        BASE_PROBE_PATH
    )

    print("Loaded ranking probe:", BASE_PROBE_PATH)
    print("Number of ranked PCs:", len(pc_ranking))
    print("Top 20 PC indices:", pc_ranking[:20])
    print(
        "Top 20 |coef_z|:",
        np.abs(coef_z[pc_ranking[:20]]),
    )

    def select_ranked_pcs(k):
        """
        Select the k PCs with the largest absolute full-probe coefficient.
        """
        return pc_ranking[:k]

    run_logistic_experiment(
        experiment_name="selected_pcs_after_coefs",
        weather_feature=WEATHER_FEATURE,
        feature_source="pca",
        feature_counts=TOP_K_COUNTS,
        selected_features_fn=select_ranked_pcs,
        fine_mesh_level=FINE_MESH_LEVEL,
        node_hierarchy_level=NODE_HIERARCHY_LEVEL,
        label_mode=LABEL_MODE,
        max_time_difference_hours=MAX_TIME_DIFFERENCE_HOURS,
        thresholds=THRESHOLDS,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        val_start=VAL_START,
        val_end=VAL_END,
        test_start=TEST_START,
        test_end=TEST_END,
        mask_dir=MASK_DIR,
        out_dir=OUT_DIR,
        pc_scores_paths=PC_SCORES_PATHS,
        timestep_files_txts=TIMESTEP_FILES_TXTS,
        extra_metadata={
            "ranking_source_probe": BASE_PROBE_PATH,
            "ranking_method": "descending_abs_coef_z",
        },
    )


if __name__ == "__main__":
    main()
