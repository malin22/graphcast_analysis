import os

import numpy as np
import pandas as pd

from malins_helper_scripts.activation_preprocessing import (
    build_graphcast_time_table,
    load_pca_metadata,
)
from malins_regression.logistic_probe_pipeline import (
    build_pca_X_for_split,
    build_split_masks,
    filter_finite_rows,
    match_climatenet_events,
)
from malins_helper_scripts.mesh_context import (
    get_coarse_mesh_node_indices,
    get_mesh_latlon,
)


# ============================================================
# CONFIG
# ============================================================

WEATHER_FEATURE = "AR"

FINE_MESH_LEVEL = 6
NODE_HIERARCHY_LEVEL = 6

N_CANDIDATE_PCS = 512

LABEL_MODE = "intersection"
MAX_TIME_DIFFERENCE_HOURS = 3

TRAIN_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2020-11-01")

VAL_START = pd.Timestamp("2020-11-01")
VAL_END = pd.Timestamp("2021-01-01")


# ============================================================
# PATHS
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
}

MASK_DIR = (
    f"/share/prj-4d/graphcast_shared/data/"
    f"ClimateNetLarge/{WEATHER_FEATURE}_labels_cleaned"
)

CACHE_DIR = (
    f"results/logistic_regression/"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"l1_pc_selection/cache/"
)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Mesh
    # --------------------------------------------------------

    lat, lon = get_mesh_latlon(
        splits=FINE_MESH_LEVEL,
    )
    lon = lon % 360

    all_nodes = get_coarse_mesh_node_indices(
        fine_splits=FINE_MESH_LEVEL,
        coarse_splits=NODE_HIERARCHY_LEVEL,
    )

    samples_per_t = len(all_nodes)

    print("Nodes per timestep:", samples_per_t)

    # --------------------------------------------------------
    # PCA metadata
    # --------------------------------------------------------

    (
        pc_scores_by_year,
        timestamps_by_year,
        max_features,
    ) = load_pca_metadata(
        PC_SCORES_PATHS,
        TIMESTEP_FILES_TXTS,
    )

    if N_CANDIDATE_PCS > max_features:
        raise ValueError(
            f"N_CANDIDATE_PCS={N_CANDIDATE_PCS}, "
            f"but only {max_features} PCA features are available."
        )

    graphcast_df = build_graphcast_time_table(
        timestamps_by_year
    )

    # --------------------------------------------------------
    # ClimateNet matching
    # --------------------------------------------------------

    matched_df, y, _ = match_climatenet_events(
        graphcast_df,
        MASK_DIR,
        lat,
        lon,
        all_nodes,
        label_mode=LABEL_MODE,
        max_time_difference_hours=MAX_TIME_DIFFERENCE_HOURS,
        include_activation_file=False,
    )

    split_masks = build_split_masks(
        matched_df,
        samples_per_t,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        val_start=VAL_START,
        val_end=VAL_END,
        test_start=None,
        test_end=None,
    )

    candidate_pcs = np.arange(
        N_CANDIDATE_PCS,
        dtype=int,
    )

    # --------------------------------------------------------
    # Build matrices ONCE
    # --------------------------------------------------------

    print("Building X_train...")

    X_train = build_pca_X_for_split(
        matched_df,
        split_masks["event_train"],
        pc_scores_by_year,
        all_nodes,
        candidate_pcs,
    )

    print("Building X_val...")

    X_val = build_pca_X_for_split(
        matched_df,
        split_masks["event_val"],
        pc_scores_by_year,
        all_nodes,
        candidate_pcs,
    )

    y_train = y[
        split_masks["train"]
    ]

    y_val = y[
        split_masks["val"]
    ]

    X_train, y_train, _ = filter_finite_rows(
        X_train,
        y_train,
    )

    X_val, y_val, _ = filter_finite_rows(
        X_val,
        y_val,
    )

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Training set contains only one class."
        )

    if len(np.unique(y_val)) < 2:
        raise ValueError(
            "Validation set contains only one class."
        )

    # float32 is sufficient here and cuts storage roughly in half.
    X_train = np.asarray(
        X_train,
        dtype=np.float32,
    )

    X_val = np.asarray(
        X_val,
        dtype=np.float32,
    )

    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        os.path.join(CACHE_DIR, "X_train.npy"),
        X_train,
    )

    np.save(
        os.path.join(CACHE_DIR, "X_val.npy"),
        X_val,
    )

    np.save(
        os.path.join(CACHE_DIR, "y_train.npy"),
        y_train,
    )

    np.save(
        os.path.join(CACHE_DIR, "y_val.npy"),
        y_val,
    )

    print()
    print("Saved cached L1 data:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)


if __name__ == "__main__":
    main()