import numpy as np
import pandas as pd

from malins_regression.run_logistic_probe import (
    run_logistic_experiment,
)


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

WEATHER_FEATURE = "AR"  # "AR" or "TC"

FINE_MESH_LEVEL = 6
NODE_HIERARCHY_LEVEL = 6

FEATURE_COUNTS = [512]

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
# DATA PATHS
# ============================================================

ACTS_DIRS = {
    2019: "/share/prj-4d/graphcast_shared/data/graphcast_activation_2019",
    2020: "/share/prj-4d/graphcast_shared/data/graphcast_activation_2020",
    2021: "/share/prj-4d/graphcast_shared/data/graphcast_activation_2021",
}

MASK_DIR = (
    f"/share/prj-4d/graphcast_shared/data/"
    f"ClimateNetLarge/{WEATHER_FEATURE}_labels_cleaned"
)

OUT_DIR = (
    f"results/malins_experiments/logistic_regression/second_test"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"raw_activations/"
)


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_raw_features(k):
    """
    Use the first k raw activation dimensions.

    For the current baseline experiment k=512, so this selects all
    raw activation features.
    """
    return np.arange(k)


# ============================================================
# RUN
# ============================================================

def main():
    run_logistic_experiment(
        experiment_name="raw_activations",
        weather_feature=WEATHER_FEATURE,
        feature_source="raw",
        feature_counts=FEATURE_COUNTS,
        selected_features_fn=select_raw_features,
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
        acts_dirs=ACTS_DIRS,
    )


if __name__ == "__main__":
    main()
