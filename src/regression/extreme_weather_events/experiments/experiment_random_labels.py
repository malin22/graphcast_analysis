import numpy as np
import pandas as pd

from regression.extreme_weather_events.run_logistic_probe import (
    run_logistic_experiment,
)


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

WEATHER_FEATURE = "AR"  # random-label control for AR

FINE_MESH_LEVEL = 6
NODE_HIERARCHY_LEVEL = 6

FEATURE_COUNTS = [512]

LABEL_MODE = "intersection"
MAX_TIME_DIFFERENCE_HOURS = 3

# Random-label control:
# randomly permute individual node labels within each split.
#
# This preserves the number / fraction of positive labels,
# but destroys both:
#   - temporal correspondence
#   - spatial correspondence
RANDOMIZE_LABELS = True
RANDOM_LABEL_SEED = 0


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
    2019: (
        "/share/prj-4d/graphcast_shared/data/"
        "graphcast_activation_2019"
    ),
    2020: (
        "/share/prj-4d/graphcast_shared/data/"
        "graphcast_activation_2020"
    ),
    2021: (
        "/share/prj-4d/graphcast_shared/data/"
        "graphcast_activation_2021"
    ),
}

MASK_DIR = (
    f"/share/prj-4d/graphcast_shared/data/"
    f"ClimateNetLarge/{WEATHER_FEATURE}_labels_cleaned"
)


# ============================================================
# FEATURE SELECTION
# ============================================================

def select_raw_features(k):
    """
    Use the first k raw activation dimensions.

    For k=512 this is the complete raw latent representation.
    """
    return np.arange(k)


# ============================================================
# RUN
# ============================================================

def main():

    OUT_DIR = (
        f"results/regression/extreme_weather_events/"
        f"{WEATHER_FEATURE}/"
        f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
        f"random_labels_raw_activations/"
        f"seed_{RANDOM_LABEL_SEED}/"
    )

    run_logistic_experiment(
        experiment_name="random_labels_raw_activations",
        weather_feature=WEATHER_FEATURE,
        feature_source="raw",
        feature_counts=FEATURE_COUNTS,
        selected_features_fn=select_raw_features,
        fine_mesh_level=FINE_MESH_LEVEL,
        node_hierarchy_level=NODE_HIERARCHY_LEVEL,
        label_mode=LABEL_MODE,
        max_time_difference_hours=MAX_TIME_DIFFERENCE_HOURS,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        val_start=VAL_START,
        val_end=VAL_END,
        test_start=TEST_START,
        test_end=TEST_END,
        mask_dir=MASK_DIR,
        out_dir=OUT_DIR,
        acts_dirs=ACTS_DIRS,

        # -----------------------------------------------
        # Do NOT shuffle complete days/masks
        # -----------------------------------------------
        permute_masks=False,

        # -----------------------------------------------
        # Random-label control
        # -----------------------------------------------
        randomize_labels=RANDOMIZE_LABELS,
        random_label_seed=RANDOM_LABEL_SEED,

        extra_metadata={
            "baseline": "random_label_permutation",
            "random_label_seed": RANDOM_LABEL_SEED,
        },
    )


if __name__ == "__main__":
    main()