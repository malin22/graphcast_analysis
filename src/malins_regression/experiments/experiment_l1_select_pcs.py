import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from malins_helper_scripts.activation_preprocessing import (
    build_graphcast_time_table,
    load_pca_metadata,
)
from malins_helper_scripts.logistic_probe_pipeline import (
    build_pca_X_for_split,
    build_split_masks,
    evaluate_validation,
    filter_finite_rows,
    fit_logistic_probe,
    match_climatenet_events,
)
from malins_helper_scripts.mesh_context import (
    get_coarse_mesh_node_indices,
    get_mesh_latlon,
)


# ============================================================
# CONFIG
# ============================================================

WEATHER_FEATURE = "AR"  # "AR" or "TC"

FINE_MESH_LEVEL = 6
NODE_HIERARCHY_LEVEL = 6

N_CANDIDATE_PCS = 512

# Smaller C -> stronger L1 regularization -> usually fewer PCs.
C_VALUES = np.logspace(-5, 0, 16)

N_SELECTOR_SAMPLES = 1_000_000
SELECTOR_RANDOM_SEED = 0
NONZERO_TOL = 1e-8

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

OUT_DIR = (
    f"results/malins_experiments/logistic_regression/"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"l1_pc_selection/"
)

SELECTION_OUT = os.path.join(
    OUT_DIR,
    "selected_pcs_from_l1.npz",
)

SWEEP_OUT = os.path.join(
    OUT_DIR,
    "l1_pc_selection_validation_results.csv",
)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

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
            f"N_CANDIDATE_PCS={N_CANDIDATE_PCS}, but only "
            f"{max_features} PCA features are available."
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

    # We only need train + validation for feature selection.
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

    # Build each full candidate matrix once.
    X_train = build_pca_X_for_split(
        matched_df,
        split_masks["event_train"],
        pc_scores_by_year,
        all_nodes,
        candidate_pcs,
    )

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

    # --------------------------------------------------------
    # Fixed training-only subsample for L1 selector
    # --------------------------------------------------------

    rng = np.random.default_rng(
        SELECTOR_RANDOM_SEED
    )

    n_selector = min(
        N_SELECTOR_SAMPLES,
        len(y_train),
    )

    selector_indices = rng.choice(
        len(y_train),
        size=n_selector,
        replace=False,
    )

    X_selector = X_train[
        selector_indices
    ]

    y_selector = y_train[
        selector_indices
    ]

    print()
    print("=" * 80)
    print("L1 PC FEATURE SELECTION")
    print("=" * 80)
    print("Candidate PCs:", N_CANDIDATE_PCS)
    print("Selector samples:", f"{len(y_selector):,}")
    print("Full training samples:", f"{len(y_train):,}")
    print(
        "Selector positive rate:",
        float(np.mean(y_selector)),
    )

    # --------------------------------------------------------
    # Sweep C
    # --------------------------------------------------------

    rows = []
    best = None

    for C in C_VALUES:
        print()
        print("-" * 80)
        print(f"L1 C={C:.8g}")

        selector = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l1",
                solver="saga",
                class_weight="balanced",
                C=float(C),
                max_iter=5000,
                tol=1e-4,
                random_state=SELECTOR_RANDOM_SEED,
            ),
        )

        selector.fit(
            X_selector,
            y_selector,
        )

        l1_clf = selector.named_steps[
            "logisticregression"
        ]

        coef_l1 = np.asarray(
            l1_clf.coef_[0],
            dtype=np.float32,
        )

        selected_pcs = np.flatnonzero(
            np.abs(coef_l1) > NONZERO_TOL
        )

        n_selected = len(selected_pcs)

        if n_selected == 0:
            print("Selected PCs: 0")

            rows.append({
                "C": float(C),
                "n_selected": 0,
                "selected_pc_indices": "",
                "selected_pc_numbers": "",
                "val_average_precision": np.nan,
                "val_roc_auc": np.nan,
                "val_f1": np.nan,
                "val_precision": np.nan,
                "val_recall": np.nan,
                "val_best_threshold": np.nan,
            })

            continue

        # Evaluate this selected subset with the standard L2 probe.
        model = fit_logistic_probe(
            X_train[:, selected_pcs],
            y_train,
        )

        val_metrics, _ = evaluate_validation(
            model,
            X_val[:, selected_pcs],
            y_val,
        )

        row = {
            "C": float(C),
            "n_selected": int(n_selected),
            "selected_pc_indices": ",".join(
                str(i)
                for i in selected_pcs
            ),
            "selected_pc_numbers": ",".join(
                str(i + 1)
                for i in selected_pcs
            ),
            **val_metrics,
        }

        rows.append(row)

        print(
            f"Selected PCs={n_selected} | "
            f"AP={val_metrics['val_average_precision']:.4f} | "
            f"AUC={val_metrics['val_roc_auc']:.4f} | "
            f"F1={val_metrics['val_f1']:.4f}"
        )

        score = val_metrics[
            "val_average_precision"
        ]

        if (
            best is None
            or score > best["val_average_precision"]
            or (
                np.isclose(
                    score,
                    best["val_average_precision"],
                )
                and n_selected < best["n_selected"]
            )
        ):
            best = {
                "C": float(C),
                "selected_pcs": selected_pcs.copy(),
                "coef_l1": coef_l1.copy(),
                "n_selected": int(n_selected),
                **val_metrics,
            }

    if best is None:
        raise ValueError(
            "No L1 C value selected any PCs."
        )

    # --------------------------------------------------------
    # Save selector outputs
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        rows
    )

    results_df.to_csv(
        SWEEP_OUT,
        index=False,
    )

    np.savez(
        SELECTION_OUT,
        selected_pcs=best["selected_pcs"],
        selected_pc_numbers=best["selected_pcs"] + 1,
        coef_l1=best["coef_l1"],
        best_C=best["C"],
        n_selected=best["n_selected"],
        selection_metric="validation_average_precision",
        val_average_precision=best["val_average_precision"],
        val_roc_auc=best["val_roc_auc"],
        val_f1=best["val_f1"],
        val_precision=best["val_precision"],
        val_recall=best["val_recall"],
        val_best_threshold=best["val_best_threshold"],
        n_candidate_pcs=N_CANDIDATE_PCS,
        n_selector_samples=len(y_selector),
        selector_random_seed=SELECTOR_RANDOM_SEED,
        nonzero_tolerance=NONZERO_TOL,
        train_start=str(TRAIN_START),
        train_end=str(TRAIN_END),
        val_start=str(VAL_START),
        val_end=str(VAL_END),
        weather_feature=WEATHER_FEATURE,
        label_mode=LABEL_MODE,
        fine_mesh_level=FINE_MESH_LEVEL,
        node_hierarchy_level=NODE_HIERARCHY_LEVEL,
    )

    print()
    print("=" * 80)
    print("L1 SELECTION FINISHED")
    print("=" * 80)
    print("Best C:", best["C"])
    print("Selected PCs:", best["n_selected"])
    print("Selected PC indices:", best["selected_pcs"])
    print(
        "Validation AP:",
        best["val_average_precision"],
    )
    print("Saved sweep:", SWEEP_OUT)
    print("Saved selection:", SELECTION_OUT)


if __name__ == "__main__":
    main()
