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
        "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
        "pc_scores_2019_from_2019_2020_pca_per_timestep.npy"
    ),
    2020: (
        "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
        "pc_scores_2020_from_2019_2020_pca_per_timestep.npy"
    ),
    2021: (
        "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
        "pc_scores_2021_from_2019_2020_pca_per_timestep.npy"
    ),
}

TIMESTEP_FILES_TXTS = {
    2019: (
        "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
        "pc_scores_2019_from_2019_2020_pca_per_timestep_files.txt"
    ),
    2020: (
        "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
        "pc_scores_2020_from_2019_2020_pca_per_timestep_files.txt"
    ),
    2021: (
        "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
        "pc_scores_2021_from_2019_2020_pca_per_timestep_files.txt"
    ),
}


MASK_DIR = (
    f"/share/prj-4d/graphcast_shared/data/"
    f"ClimateNetLarge/{WEATHER_FEATURE}_labels_cleaned"
)


# ============================================================
# SAVED L1 SELECTION
# ============================================================

SELECTION_PATH = (
    f"results/logistic_regression/"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"l1_pc_selection/"
    f"selected_pcs_from_l1.npz"
)


OUT_DIR = (
    f"plots/malins_experiments/logistic_regression/"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"l1_selected_pcs_l2/"
)


# ============================================================
# LOAD SELECTION
# ============================================================

def load_selected_pcs(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Could not find saved L1 PC selection:\n"
            f"{path}\n\n"
            "Run experiment_l1_select_pcs.py first."
        )

    with np.load(
        path,
        allow_pickle=True,
    ) as data:
        if "selected_pcs" not in data:
            raise KeyError(
                f"{path} does not contain 'selected_pcs'. "
                f"Available keys: {list(data.keys())}"
            )

        selected_pcs = np.asarray(
            data["selected_pcs"],
            dtype=int,
        )

        best_C = (
            float(data["best_C"])
            if "best_C" in data
            else np.nan
        )

        selection_val_ap = (
            float(data["val_average_precision"])
            if "val_average_precision" in data
            else np.nan
        )

    if selected_pcs.size == 0:
        raise ValueError(
            "Saved L1 selection contains zero PCs."
        )

    return (
        selected_pcs,
        best_C,
        selection_val_ap,
    )


# ============================================================
# RUN
# ============================================================

def main():
    (
        selected_pcs,
        best_C,
        selection_val_ap,
    ) = load_selected_pcs(
        SELECTION_PATH
    )

    print("Loaded L1 selection:", SELECTION_PATH)
    print("Selected PCs:", len(selected_pcs))
    print("Selected PC indices:", selected_pcs)
    print("L1 best C:", best_C)
    print("L1 selection validation AP:", selection_val_ap)

    def use_l1_selected_pcs(_):
        return selected_pcs

    run_logistic_experiment(
        experiment_name="l1_selected_pcs_l2",
        weather_feature=WEATHER_FEATURE,
        feature_source="pca",
        feature_counts=[len(selected_pcs)],
        selected_features_fn=use_l1_selected_pcs,
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
            "feature_selection": "l1_logistic_nonzero_coefficients",
            "selection_path": SELECTION_PATH,
            "l1_best_C": best_C,
            "l1_selection_val_average_precision": selection_val_ap,
        },
    )


if __name__ == "__main__":
    main()
