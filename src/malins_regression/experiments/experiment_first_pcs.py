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

# Test how much ClimateNet-relevant information is retained as
# progressively more variance-ranked principal components are used.
PC_COUNTS = [5, 10, 25, 50, 100, 200, 300, 400, 512]

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

# Update these paths if your PCA output directory names differ.
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

OUT_DIR = (
    f"results/logistic_regression/"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"first_k_pcs/"
)


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_first_pcs(k):
    """
    Select the first k principal components.

    PCA components are ordered by explained variance, so this experiment
    tests PCs [0, ..., k-1] for each requested k.
    """
    return np.arange(k)


# ============================================================
# RUN
# ============================================================

def main():
    run_logistic_experiment(
        experiment_name="first_k_pcs",
        weather_feature=WEATHER_FEATURE,
        feature_source="pca",
        feature_counts=PC_COUNTS,
        selected_features_fn=select_first_pcs,
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
    )


if __name__ == "__main__":
    main()
