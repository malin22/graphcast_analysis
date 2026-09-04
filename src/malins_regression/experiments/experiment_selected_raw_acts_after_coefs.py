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
# RAW ACTIVATION DATA PATHS
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


# ============================================================
# FULL RAW-ACTIVATION PROBE USED FOR RANKING
# ============================================================

# This must point to the 512-feature raw-activation probe produced by
# experiment_raw_acts.py.
#
# Ranking is based on standardized logistic-regression coefficients:
#
#     importance_j = abs(coef_z[j])
#
# The full 512-feature probe is used only to DEFINE the ranking.
# Each top-k experiment trains a NEW logistic probe using only those
# selected raw activation dimensions.

BASE_PROBE_PATH = (
    f"results/logistic_regression/"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"raw_activations/"
    f"probe_direction_{WEATHER_FEATURE}_"
    f"raw_activations_{LABEL_MODE}_"
    f"M{NODE_HIERARCHY_LEVEL}_"
    f"512_features_"
    f"2019_2020_train_only.npz"
)


OUT_DIR = (
    f"results/logistic_regression/"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"selected_raw_acts_after_coefs/"
)


# ============================================================
# RAW-FEATURE RANKING
# ============================================================

def load_raw_feature_ranking(probe_path):
    """
    Load the full raw-activation logistic probe and rank activation
    dimensions by |coef_z|.

    coef_z is the coefficient vector in standardized feature space, so
    ranking by |coef_z| measures feature relevance under the exact
    standardized L2 logistic model used for the baseline raw probe.
    """
    if not os.path.exists(probe_path):
        raise FileNotFoundError(
            "Could not find the full raw-activation probe used for ranking:\n"
            f"{probe_path}\n\n"
            "Run experiment_raw_acts.py with 512 features first."
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
            f"but TOP_K_COUNTS requests up to {max(TOP_K_COUNTS)} "
            "raw activation dimensions."
        )

    feature_ranking = np.argsort(
        np.abs(coef_z)
    )[::-1]

    return feature_ranking, coef_z


# ============================================================
# RUN
# ============================================================

def main():
    feature_ranking, coef_z = load_raw_feature_ranking(
        BASE_PROBE_PATH
    )

    print("Loaded ranking probe:", BASE_PROBE_PATH)
    print("Number of ranked raw features:", len(feature_ranking))
    print("Top 20 raw feature indices:", feature_ranking[:20])
    print(
        "Top 20 |coef_z|:",
        np.abs(coef_z[feature_ranking[:20]]),
    )

    def select_ranked_raw_features(k):
        """
        Select the k raw activation dimensions with the largest
        absolute full-probe coefficient.
        """
        return feature_ranking[:k]

    run_logistic_experiment(
        experiment_name="selected_raw_acts_after_coefs",
        weather_feature=WEATHER_FEATURE,
        feature_source="raw",
        feature_counts=TOP_K_COUNTS,
        selected_features_fn=select_ranked_raw_features,
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
        extra_metadata={
            "ranking_source_probe": BASE_PROBE_PATH,
            "ranking_method": "descending_abs_coef_z",
        },
    )


if __name__ == "__main__":
    main()
