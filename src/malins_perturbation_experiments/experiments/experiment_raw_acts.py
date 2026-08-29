from pathlib import Path

import numpy as np

from malins_perturbation_experiments.perturbation import PerturbationDirection
from malins_perturbation_experiments.run_perturbation import run_perturbation


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

WEATHER_FEATURE = "AR"
NODE_HIERARCHY_LEVEL = 6

THRESHOLD = 0.9

EXPERIMENT_NAME = "raw_activations"


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/home/student/m/mbraatz/share/graphcast_analysis"
)

PROBE_PATH = (
    PROJECT_ROOT
    / "results"
    / "malins_experiments"
    / "logistic_regression"
    / WEATHER_FEATURE
    / f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}"
    / "raw_activations"
    / (
        f"probe_direction_{WEATHER_FEATURE}_raw_activations_"
        f"intersection_M{NODE_HIERARCHY_LEVEL}_512_features_"
        "2019_2020_train_only.npz"
    )
)


# ============================================================
# BUILD INTERVENTION
# ============================================================

def build_intervention() -> PerturbationDirection:
    """
    Build the perturbation intervention from a logistic probe trained
    directly on the 512 raw GraphCast activation features.

    No PCA transformation is needed.
    """

    # --------------------------------------------------------
    # Check probe file
    # --------------------------------------------------------

    if not PROBE_PATH.exists():
        raise FileNotFoundError(
            f"Probe file not found: {PROBE_PATH}"
        )

    # --------------------------------------------------------
    # Load logistic probe
    # --------------------------------------------------------

    probe = np.load(PROBE_PATH)

    scaler_mean = np.asarray(
        probe["scaler_mean"],
        dtype=np.float32,
    )

    scaler_scale = np.asarray(
        probe["scaler_scale"],
        dtype=np.float32,
    )

    coef_z = np.asarray(
        probe["coef_z"],
        dtype=np.float32,
    )

    intercept = float(
        np.ravel(probe["intercept"])[0]
    )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    if coef_z.shape != (512,):
        raise ValueError(
            f"Expected coef_z shape (512,), got {coef_z.shape}"
        )

    if scaler_mean.shape != (512,):
        raise ValueError(
            f"Expected scaler_mean shape (512,), "
            f"got {scaler_mean.shape}"
        )

    if scaler_scale.shape != (512,):
        raise ValueError(
            f"Expected scaler_scale shape (512,), "
            f"got {scaler_scale.shape}"
        )

    # ========================================================
    # CONVERT STANDARDIZED PROBE TO RAW ACTIVATION SPACE
    # ========================================================

    # The logistic probe was trained on:
    #
    #   z = (x - scaler_mean) / scaler_scale
    #
    # and:
    #
    #   logit = z @ coef_z + intercept
    #
    # Therefore the equivalent classifier directly on raw
    # activations is:
    #
    #   logit = x @ probe_weight + probe_bias

    probe_weight = (
        coef_z / (scaler_scale + 1e-8)
    ).astype(np.float32)

    probe_bias = (
        intercept
        - float(
            np.dot(
                scaler_mean,
                probe_weight,
            )
        )
    )

    # ========================================================
    # PERTURBATION DIRECTION
    # ========================================================

    # Perturb along the logistic-regression direction in the
    # original 512-D GraphCast activation space.
    direction = probe_weight.copy()

    norm = np.linalg.norm(direction)

    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            f"Invalid perturbation direction norm: {norm}"
        )

    direction /= norm

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("============================================")
    print("Raw-activation perturbation")
    print("============================================")
    print("Weather feature:", WEATHER_FEATURE)
    print("Node hierarchy:", NODE_HIERARCHY_LEVEL)
    print("Threshold:", THRESHOLD)
    print()
    print("Probe:", PROBE_PATH)
    print()
    print("Probe weight:", probe_weight.shape)
    print("Probe bias:", probe_bias)
    print("Direction:", direction.shape)
    print("Direction norm:", np.linalg.norm(direction))
    print("============================================")

    return PerturbationDirection(
        direction=direction.astype(np.float32),
        probe_weight=probe_weight.astype(np.float32),
        probe_bias=float(probe_bias),
        threshold=THRESHOLD,
    )


# ============================================================
# RUN EXPERIMENT
# ============================================================

def main() -> None:
    intervention = build_intervention()

    run_perturbation(
        intervention=intervention,
        experiment_name=EXPERIMENT_NAME,
        weather_feature=WEATHER_FEATURE,
        node_hierarchy_level=NODE_HIERARCHY_LEVEL,
    )


if __name__ == "__main__":
    main()