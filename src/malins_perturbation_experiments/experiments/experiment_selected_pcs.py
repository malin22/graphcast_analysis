from pathlib import Path

import numpy as np

from malins_perturbation_experiments.perturbation import PerturbationDirection
from malins_perturbation_experiments.run_perturbation import run_perturbation


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

WEATHER_FEATURE = "AR"
NODE_HIERARCHY_LEVEL = 6

N_SELECTED_PCS = 200

THRESHOLD = 0.9

EXPERIMENT_NAME = f"selected_top_{N_SELECTED_PCS}_pcs"

START_TIME = "2021-02-12T18"


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


# ============================================================
# RANKING PROBE
# ============================================================

# This is the ORIGINAL 512-PC probe used to rank PCs.
RANKING_PROBE_PATH = (
    PROJECT_ROOT
    / "results"
    / "logistic_regression"
    / WEATHER_FEATURE
    / f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}"
    / "first_k_pcs"
    / (
        f"probe_direction_{WEATHER_FEATURE}_"
        f"first_k_pcs_intersection_"
        f"M{NODE_HIERARCHY_LEVEL}_"
        f"512_features_"
        f"2019_2020_train_only.npz"
    )
)


# ============================================================
# SELECTED-PC PROBE
# ============================================================

# This must be the NEW probe trained using only the selected PCs.
#
# Adjust the filename here if run_logistic_experiment uses a
# slightly different naming convention.
SELECTED_PROBE_PATH = (
    PROJECT_ROOT
    / "results"
    / "logistic_regression"
    / WEATHER_FEATURE
    / f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}"
    / "selected_pcs_after_coefs"
    / (
        f"probe_direction_{WEATHER_FEATURE}_"
        f"selected_pcs_after_coefs_intersection_"
        f"M{NODE_HIERARCHY_LEVEL}_"
        f"{N_SELECTED_PCS}_features_"
        f"2019_2020_train_only.npz"
    )
)


# ============================================================
# BUILD INTERVENTION
# ============================================================

def build_intervention() -> PerturbationDirection:
    """
    Build a perturbation direction from a logistic probe trained
    on the top-k PCs ranked by |coef_z| of the original 512-PC probe.

    The selected-PC probe is converted from standardized PCA space
    back into GraphCast's original 512-D activation space.
    """

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path, description in [
        (RANKING_PROBE_PATH, "ranking probe"),
        (SELECTED_PROBE_PATH, "selected-PC probe"),
        (PCA_COMPONENTS_PATH, "PCA components"),
        (PCA_MEAN_PATH, "PCA mean"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{description} file not found: {path}"
            )

    # ========================================================
    # 1. RECONSTRUCT THE PC RANKING
    # ========================================================

    with np.load(RANKING_PROBE_PATH) as ranking_probe:
        ranking_coef_z = np.asarray(
            ranking_probe["coef_z"],
            dtype=np.float64,
        ).reshape(-1)

    if ranking_coef_z.shape != (512,):
        raise ValueError(
            "Expected ranking probe to contain 512 coefficients, "
            f"got {ranking_coef_z.shape}"
        )

    # Exactly the same ranking used during training:
    #
    # pc_ranking = np.argsort(abs(coef_z))[::-1]
    pc_ranking = np.argsort(
        np.abs(ranking_coef_z)
    )[::-1]

    selected_pc_indices = pc_ranking[:N_SELECTED_PCS]

    # ========================================================
    # 2. LOAD THE NEW SELECTED-PC PROBE
    # ========================================================

    with np.load(SELECTED_PROBE_PATH) as probe:
        print(
            "Selected probe keys:",
            list(probe.keys()),
        )

        scaler_mean = np.asarray(
            probe["scaler_mean"],
            dtype=np.float32,
        ).reshape(-1)

        scaler_scale = np.asarray(
            probe["scaler_scale"],
            dtype=np.float32,
        ).reshape(-1)

        coef_z = np.asarray(
            probe["coef_z"],
            dtype=np.float32,
        ).reshape(-1)

        intercept = float(
            np.ravel(probe["intercept"])[0]
        )

    # ========================================================
    # 3. LOAD THE SELECTED PCA DIRECTIONS
    # ========================================================

    all_pca_components = np.load(
        PCA_COMPONENTS_PATH,
        mmap_mode="r",
    )

    if all_pca_components.shape != (512, 512):
        raise ValueError(
            "Expected full PCA component matrix shape (512, 512), "
            f"got {all_pca_components.shape}"
        )

    # IMPORTANT:
    #
    # Use the ranked PCA indices, NOT [:N_SELECTED_PCS].
    #
    # Their order also needs to match the order used when training
    # the selected-PC logistic regression.
    selected_pca_components = np.asarray(
        all_pca_components[selected_pc_indices],
        dtype=np.float32,
    )

    pca_mean = np.asarray(
        np.load(
            PCA_MEAN_PATH,
            mmap_mode="r",
        ),
        dtype=np.float32,
    )

    # ========================================================
    # 4. SHAPE CHECKS
    # ========================================================

    expected_shape = (N_SELECTED_PCS,)

    if coef_z.shape != expected_shape:
        raise ValueError(
            f"Expected coef_z shape {expected_shape}, "
            f"got {coef_z.shape}"
        )

    if scaler_mean.shape != expected_shape:
        raise ValueError(
            f"Expected scaler_mean shape {expected_shape}, "
            f"got {scaler_mean.shape}"
        )

    if scaler_scale.shape != expected_shape:
        raise ValueError(
            f"Expected scaler_scale shape {expected_shape}, "
            f"got {scaler_scale.shape}"
        )

    if selected_pca_components.shape != (
        N_SELECTED_PCS,
        512,
    ):
        raise ValueError(
            "Expected selected PCA component matrix shape "
            f"({N_SELECTED_PCS}, 512), "
            f"got {selected_pca_components.shape}"
        )

    if pca_mean.shape != (512,):
        raise ValueError(
            f"Expected PCA mean shape (512,), got {pca_mean.shape}"
        )

    if np.any(~np.isfinite(scaler_scale)):
        raise ValueError(
            "scaler_scale contains non-finite values"
        )

    if np.any(scaler_scale <= 0):
        raise ValueError(
            "scaler_scale contains non-positive values"
        )

    # ========================================================
    # 5. CONVERT SELECTED-PC PROBE TO RAW 512-D SPACE
    # ========================================================

    # During training:
    #
    #     z_j = (pc_j - scaler_mean_j) / scaler_scale_j
    #
    # and
    #
    #     logit = z @ coef_z + intercept
    #
    # Undo the standardization of the classifier weights.
    probe_weight_pc = (
        coef_z / scaler_scale
    )

    # Selected PCA scores are:
    #
    #     pc_selected =
    #         (x - pca_mean) @ selected_pca_components.T
    #
    # Therefore:
    #
    #     probe_weight_raw =
    #         selected_pca_components.T @ probe_weight_pc
    #
    probe_weight = (
        selected_pca_components.T @ probe_weight_pc
    ).astype(np.float32)

    # Correct the intercept for:
    #
    # 1. standardization mean
    # 2. PCA centering
    #
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
    # 6. PERTURBATION DIRECTION
    # ========================================================

    direction = probe_weight.copy()

    norm = float(
        np.linalg.norm(direction)
    )

    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(
            f"Invalid perturbation direction norm: {norm}"
        )

    direction /= norm

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    print()
    print("============================================")
    print("Selected-PC perturbation")
    print("============================================")
    print("Weather feature:", WEATHER_FEATURE)
    print("Node hierarchy:", NODE_HIERARCHY_LEVEL)
    print("Number selected:", N_SELECTED_PCS)
    print("Threshold:", THRESHOLD)
    print()

    print("Ranking probe:")
    print(RANKING_PROBE_PATH)
    print()

    print("Selected-PC probe:")
    print(SELECTED_PROBE_PATH)
    print()

    print("Selected PC indices:")
    print(selected_pc_indices)
    print()

    print("Ranking |coef_z|:")
    print(
        np.abs(
            ranking_coef_z[selected_pc_indices]
        )
    )
    print()

    print(
        "Selected PCA matrix:",
        selected_pca_components.shape,
    )
    print(
        "Selected probe coef_z:",
        coef_z.shape,
    )
    print(
        "Raw probe weight:",
        probe_weight.shape,
    )
    print(
        "Raw probe bias:",
        probe_bias,
    )
    print(
        "Direction:",
        direction.shape,
    )
    print(
        "Direction norm:",
        np.linalg.norm(direction),
    )
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
        start_times=[START_TIME],
    )


if __name__ == "__main__":
    main()