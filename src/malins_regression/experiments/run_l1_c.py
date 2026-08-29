import argparse
import os

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from malins_regression.logistic_probe_pipeline import (
    evaluate_validation,
    fit_logistic_probe,
)


# ============================================================
# CONFIG
# ============================================================

WEATHER_FEATURE = "TC"
NODE_HIERARCHY_LEVEL = 6

C_VALUES = np.logspace(-5, 0, 16)

N_SELECTOR_SAMPLES = 100_000
SELECTOR_RANDOM_SEED = 0
NONZERO_TOL = 1e-8


# ============================================================
# PATHS
# ============================================================

OUT_DIR = (
    f"results/malins_experiments/logistic_regression/"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"l1_pc_selection/"
)

CACHE_DIR = os.path.join(
    OUT_DIR,
    "cache",
)

JOB_RESULTS_DIR = os.path.join(
    OUT_DIR,
    "c_sweep_results",
)


# ============================================================
# RUN ONE C
# ============================================================

def run_one_c(c_index: int):
    if c_index < 0 or c_index >= len(C_VALUES):
        raise ValueError(
            f"c_index must be between 0 and "
            f"{len(C_VALUES) - 1}"
        )

    C = float(C_VALUES[c_index])

    os.makedirs(
        JOB_RESULTS_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load cached arrays
    # --------------------------------------------------------

    X_train = np.load(
        os.path.join(CACHE_DIR, "X_train.npy"),
        mmap_mode="r",
    )

    X_val = np.load(
        os.path.join(CACHE_DIR, "X_val.npy"),
        mmap_mode="r",
    )

    y_train = np.load(
        os.path.join(CACHE_DIR, "y_train.npy"),
        mmap_mode="r",
    )

    y_val = np.load(
        os.path.join(CACHE_DIR, "y_val.npy"),
        mmap_mode="r",
    )

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)

    # --------------------------------------------------------
    # Deterministic selector subset
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

    X_selector = np.asarray(
        X_train[selector_indices],
        dtype=np.float32,
    )

    y_selector = np.asarray(
        y_train[selector_indices],
    )

    # --------------------------------------------------------
    # L1 selector
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(f"C index: {c_index}")
    print(f"C: {C:.8g}")
    print("=" * 80)

    selector = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l1",
            solver="saga",
            class_weight="balanced",
            C=C,
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

    print("Selected PCs:", n_selected)

    # --------------------------------------------------------
    # Evaluate selected PCs using standard L2 probe
    # --------------------------------------------------------

    if n_selected == 0:
        metrics = {
            "val_average_precision": np.nan,
            "val_roc_auc": np.nan,
            "val_f1": np.nan,
            "val_precision": np.nan,
            "val_recall": np.nan,
            "val_best_threshold": np.nan,
        }

    else:
        # Materialize only selected columns.
        X_train_selected = np.asarray(
            X_train[:, selected_pcs],
            dtype=np.float32,
        )

        X_val_selected = np.asarray(
            X_val[:, selected_pcs],
            dtype=np.float32,
        )

        model = fit_logistic_probe(
            X_train_selected,
            np.asarray(y_train),
        )

        metrics, _ = evaluate_validation(
            model,
            X_val_selected,
            np.asarray(y_val),
        )

    # --------------------------------------------------------
    # Save this C independently
    # --------------------------------------------------------

    out_path = os.path.join(
        JOB_RESULTS_DIR,
        f"c_{c_index:02d}.npz",
    )

    np.savez(
        out_path,
        c_index=c_index,
        C=C,
        selected_pcs=selected_pcs,
        selected_pc_numbers=selected_pcs + 1,
        coef_l1=coef_l1,
        n_selected=n_selected,
        n_selector_samples=n_selector,
        **metrics,
    )

    print()
    print("Validation AP:", metrics["val_average_precision"])
    print("Saved:", out_path)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--c-index",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    run_one_c(
        args.c_index
    )


if __name__ == "__main__":
    main()