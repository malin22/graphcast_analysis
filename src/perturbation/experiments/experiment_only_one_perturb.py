from pathlib import Path

import numpy as np

from perturbation.perturbation import (
    PerturbationDirection,
    run_first_step_perturbation_experiment,
)


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

WEATHER_FEATURE = "AR"
NODE_HIERARCHY_LEVEL = 6

THRESHOLD = 0.9

START_TIME = "2021-02-12T18"

EXPERIMENT_NAME = "raw_activations_single_perturbation"

N_DAYS = 5

GAMMAS = [
    -1.0,
    -0.5,
    -0.2,
    0.0,
    0.2,
    0.5,
    1.0,
]

INJECTION_STEPS = (8,)
INJECTION_NODE_SETS = ("mesh_nodes",)

RANDOM_SEED = 0


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/home/student/m/mbraatz/share/graphcast_analysis"
)

ERA5_DATA_DIR = Path(
    "/share/prj-4d/graphcast_shared/data/era5_daily_nc"
)

PROBE_PATH = (
    PROJECT_ROOT
    / "results"
    /"regression"
    / "extreme_weather_events"
    / WEATHER_FEATURE
    / f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}"
    / "raw_activations"
    / (
        f"probe_direction_{WEATHER_FEATURE}_raw_activations_"
        f"intersection_M{NODE_HIERARCHY_LEVEL}_512_features_"
        "2019_2020_train_only.npz"
    )
)

OUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "perturbation"
    / WEATHER_FEATURE
    / f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}"
    / EXPERIMENT_NAME
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
    #direction = probe_weight.copy()

    #norm = np.linalg.norm(direction)

    #if not np.isfinite(norm) or norm <= 0:
    #    raise ValueError(
    #        f"Invalid perturbation direction norm: {norm}"
    #    )

    #direction /= norm

    direction_z = coef_z.copy()

    norm_z = np.linalg.norm(direction_z)

    if not np.isfinite(norm_z) or norm_z <= 0:
        raise ValueError(
            f"Invalid standardized direction norm: {norm_z}"
        )

    direction_z /= norm_z

    direction = (
        scaler_scale * direction_z
    ).astype(np.float32)

    print(
    "Standardized direction norm:",
    np.linalg.norm(direction / scaler_scale),
    )

    print(
        "Raw-space direction norm:",
        np.linalg.norm(direction),
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("============================================")
    print("Single-injection raw-activation perturbation")
    print("============================================")
    print("Weather feature:", WEATHER_FEATURE)
    print("Node hierarchy:", NODE_HIERARCHY_LEVEL)
    print("Threshold:", THRESHOLD)
    print("Start time:", START_TIME)
    print("Forecast days:", N_DAYS)
    print("Injection processor steps:", INJECTION_STEPS)
    print("Injection node sets:", INJECTION_NODE_SETS)
    print()
    print("Probe:", PROBE_PATH)
    print("Output:", OUT_DIR)
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

    run_first_step_perturbation_experiment(
        intervention=intervention,
        gammas=GAMMAS,
        start_times=[START_TIME],
        n_days=N_DAYS,
        era5_data_dir=ERA5_DATA_DIR,
        out_dir=OUT_DIR,
        injection_steps=INJECTION_STEPS,
        injection_node_sets=INJECTION_NODE_SETS,
        random_seed=RANDOM_SEED,
    )


if __name__ == "__main__":
    main()