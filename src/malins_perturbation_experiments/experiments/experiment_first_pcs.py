from pathlib import Path

import numpy as np

from malins_perturbation.perturbation import PerturbationDirection
from malins_perturbation.run_perturbation import run_perturbation


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

WEATHER_FEATURE = "AR"
NODE_HIERARCHY_LEVEL = 6

N_PCS = 200
THRESHOLD = 0.9

EXPERIMENT_NAME = f"first_{N_PCS}_pcs"


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/home/student/m/mbraatz/share/graphcast_analysis"
)

PCA_COMPONENTS_PATH = Path(
    "/share/prj-4d/graphcast_shared/data/"
    "pca_components/512_PCs/layer8_only/"
    "pca_components_2019_2020_layer8.npy"
)

PCA_MEAN_PATH = Path(
    "/share/prj-4d/graphcast_shared/data/"
    "pca_components/512_PCs/layer8_only/"
    "pca_mean_2019_2020_layer8.npy"
)

PROBE_PATH = (
    PROJECT_ROOT
    / "results"
    / "logistic_regression"
    / WEATHER_FEATURE
    / f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}"
    / "first_k_pcs"
    / (
        f"probe_direction_{WEATHER_FEATURE}_PCA_"
        f"intersection_M{NODE_HIERARCHY_LEVEL}_"
        f"{N_PCS}_features_2019_2020_train_only.npz"
    )
)


# ============================================================
# BUILD INTERVENTION
# ============================================================

def build_intervention() -> PerturbationDirection:
    """
    Build the perturbation intervention for a logistic probe trained
    on the first N_PCS principal components.

    Both the logistic classifier and the perturbation direction are
    converted into GraphCast's original 512-D activation space.
    """

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not PROBE_PATH.exists():
        raise FileNotFoundError(
            f"Probe file not found: {PROBE_PATH}"
        )

    if not PCA_COMPONENTS_PATH.exists():
        raise FileNotFoundError(
            f"PCA components file not found: {PCA_COMPONENTS_PATH}"
        )

    if not PCA_MEAN_PATH.exists():
        raise FileNotFoundError(
            f"PCA mean file not found: {PCA_MEAN_PATH}"
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
    # Load PCA basis
    # --------------------------------------------------------

    all_pca_components = np.load(
        PCA_COMPONENTS_PATH,
        mmap_mode="r",
    )

    pca_components = np.asarray(
        all_pca_components[:N_PCS],
        dtype=np.float32,
    )

    pca_mean = np.asarray(
        np.load(
            PCA_MEAN_PATH,
            mmap_mode="r",
        ),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    if coef_z.shape != (N_PCS,):
        raise ValueError(
            f"Expected coef_z shape ({N_PCS},), "
            f"got {coef_z.shape}"
        )

    if scaler_mean.shape != (N_PCS,):
        raise ValueError(
            f"Expected scaler_mean shape ({N_PCS},), "
            f"got {scaler_mean.shape}"
        )

    if scaler_scale.shape != (N_PCS,):
        raise ValueError(
            f"Expected scaler_scale shape ({N_PCS},), "
            f"got {scaler_scale.shape}"
        )

    if pca_components.shape != (N_PCS, 512):
        raise ValueError(
            f"Expected PCA components shape ({N_PCS}, 512), "
            f"got {pca_components.shape}"
        )

    if pca_mean.shape != (512,):
        raise ValueError(
            f"Expected PCA mean shape (512,), "
            f"got {pca_mean.shape}"
        )

    # ========================================================
    # CONVERT LOGISTIC PROBE TO RAW 512-D ACTIVATION SPACE
    # ========================================================

    # The logistic probe was trained on standardized PCA scores:
    #
    # z = (pc_scores - scaler_mean) / scaler_scale
    #
    # Therefore first undo the standardization of the weights.
    probe_weight_pc = (
        coef_z / (scaler_scale + 1e-8)
    )

    # PCA projection:
    #
    # pc_scores = (x - pca_mean) @ pca_components.T
    #
    # Convert the classifier weights back into raw activation space.
    probe_weight = (
        pca_components.T @ probe_weight_pc
    ).astype(np.float32)

    # Convert the intercept as well so that:
    #
    # old:
    #   logit = standardized_PCs @ coef_z + intercept
    #
    # becomes exactly:
    #
    # new:
    #   logit = raw_activation @ probe_weight + probe_bias
    probe_bias = (
        intercept
        - float(
            np.dot(
                scaler_mean,
                probe_weight_pc,
            )
        )
        - float(
            np.dot(
                pca_mean,
                probe_weight,
            )
        )
    )

    # ========================================================
    # PERTURBATION DIRECTION
    # ========================================================

    # For the first-PC experiment, perturb along the same direction
    # as the converted logistic probe in raw activation space.
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
    print("First-PC perturbation")
    print("============================================")
    print("Weather feature:", WEATHER_FEATURE)
    print("Node hierarchy:", NODE_HIERARCHY_LEVEL)
    print("N_PCS:", N_PCS)
    print("Threshold:", THRESHOLD)
    print()
    print("Probe:", PROBE_PATH)
    print()
    print("PCA components:", pca_components.shape)
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