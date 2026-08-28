import os
import re
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)

from graphcast import icosahedral_mesh


# ============================================================
# CONFIG
# ============================================================

WEATHER_FEATURE = "AR"
NODE_HIERARCHY_LEVEL = 6
LABEL_MODE = "intersection"

MAX_TIME_DIFFERENCE_HOURS = 3

# Use first 200 PCs as the candidate pool
N_CANDIDATE_PCS = 512

# Smaller C = stronger L1 regularization = usually fewer PCs
C_VALUES = np.logspace(-5, 0, 16)

TRAIN_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2020-11-01")

VAL_START = pd.Timestamp("2020-11-01")
VAL_END = pd.Timestamp("2021-01-01")

N_CANDIDATE_PCS = 512
N_SELECTOR_SAMPLES = 1_000_000

C_VALUES = np.logspace(-5, 0, 16)

SELECTOR_RANDOM_SEED = 0


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
    f"plots/malins_experiments/logistic_regression/"
    f"{WEATHER_FEATURE}/Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"PCA_L1_feature_selection/"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def parse_activation_timestamp(path):
    fname = os.path.basename(path)

    m = re.search(
        r"t(\d{4})-(\d{2})-(\d{2})T(\d{2})",
        fname,
    )

    if not m:
        raise ValueError(
            f"Could not parse timestamp from {fname}"
        )

    y, mo, d, h = map(int, m.groups())
    return pd.Timestamp(y, mo, d, h)


def load_timestamps(files_txt):
    with open(files_txt, "r") as f:
        files = [
            line.strip()
            for line in f
            if line.strip()
        ]

    timestamps = pd.to_datetime([
        parse_activation_timestamp(p)
        for p in files
    ])

    return files, timestamps


def parse_mask_timestamp(path):
    fname = os.path.basename(path).replace(".nc", "")
    return pd.Timestamp(fname)


def vertices_to_latlon(vertices):
    lat = np.degrees(
        np.arcsin(vertices[:, 2])
    )

    lon = np.degrees(
        np.arctan2(
            vertices[:, 1],
            vertices[:, 0],
        )
    ) % 360

    return lat, lon


def get_mesh_latlon(splits=6):
    meshes = (
        icosahedral_mesh
        .get_hierarchy_of_triangular_meshes_for_sphere(
            splits=splits
        )
    )

    vertices = meshes[splits].vertices
    return vertices_to_latlon(vertices)


def get_coarse_mesh_node_indices(
    fine_splits=6,
    coarse_splits=6,
    decimals=8,
):
    meshes = (
        icosahedral_mesh
        .get_hierarchy_of_triangular_meshes_for_sphere(
            splits=fine_splits
        )
    )

    fine_vertices = meshes[fine_splits].vertices
    coarse_vertices = meshes[coarse_splits].vertices

    fine_keys = {
        tuple(np.round(v, decimals)): i
        for i, v in enumerate(fine_vertices)
    }

    coarse_indices = []

    for v in coarse_vertices:
        key = tuple(
            np.round(v, decimals)
        )

        if key not in fine_keys:
            raise ValueError(
                "Could not match coarse vertex to fine mesh."
            )

        coarse_indices.append(
            fine_keys[key]
        )

    return np.array(
        coarse_indices,
        dtype=int,
    )


def load_mask_at_nodes(
    mask_path,
    lat,
    lon,
    node_indices,
    label_mode="intersection",
):
    ds = xr.open_dataset(mask_path)

    try:
        label = ds["label"]

        if label_mode == "intersection":
            mask = label.min("annotator")

        elif label_mode == "union":
            mask = label.max("annotator")

        elif label_mode == "soft":
            mask = label.mean("annotator")

        else:
            raise ValueError(
                f"Unknown label mode: {label_mode}"
            )

        node_lat = xr.DataArray(
            lat[node_indices],
            dims="sample",
        )

        node_lon = xr.DataArray(
            lon[node_indices],
            dims="sample",
        )

        mask_nodes = mask.interp(
            latitude=node_lat,
            longitude=node_lon,
            method="nearest",
        ).values

    finally:
        ds.close()

    return mask_nodes.astype(np.float32)


def nearest_graphcast_row(
    mask_time,
    graphcast_df,
    max_hours=3,
):
    diffs = np.abs(
        graphcast_df["time"] - mask_time
    )

    idx = int(
        diffs.argmin()
    )

    if diffs.iloc[idx] > pd.Timedelta(
        hours=max_hours
    ):
        return None

    return graphcast_df.iloc[idx]


def load_pca_metadata(
    pc_score_paths,
    timestep_files_txts,
):
    pc_scores_by_year = {}
    timestamps_by_year = {}

    for year in sorted(pc_score_paths):
        pc_scores = np.load(
            pc_score_paths[year],
            mmap_mode="r",
        )

        _, timestamps = load_timestamps(
            timestep_files_txts[year]
        )

        if len(timestamps) != pc_scores.shape[0]:
            raise ValueError(
                f"{year}: "
                f"{len(timestamps)} timestamps but "
                f"{pc_scores.shape[0]} PC-score timesteps."
            )

        pc_scores_by_year[year] = pc_scores
        timestamps_by_year[year] = pd.to_datetime(
            timestamps
        )

        print(
            f"PC scores {year}:",
            pc_scores.shape,
        )

    return (
        pc_scores_by_year,
        timestamps_by_year,
    )


def build_graphcast_time_table(
    timestamps_by_year
):
    rows = []

    for year, times in (
        timestamps_by_year.items()
    ):
        for t_idx, t in enumerate(times):
            rows.append({
                "year": year,
                "t_idx": t_idx,
                "time": t,
            })

    return (
        pd.DataFrame(rows)
        .sort_values("time")
        .reset_index(drop=True)
    )


def build_X_first_n_pcs(
    matched_df,
    split_mask_events,
    pc_scores_by_year,
    all_nodes,
    n_pcs,
):
    X_parts = []

    selected_rows = matched_df.loc[
        split_mask_events
    ]

    for _, row in selected_rows.iterrows():
        year = int(row["year"])
        t_idx = int(row["t_idx"])

        X_t = pc_scores_by_year[
            year
        ][
            t_idx,
            all_nodes,
            :n_pcs
        ]

        X_parts.append(
            np.asarray(
                X_t,
                dtype=np.float32,
            )
        )

    if not X_parts:
        return np.empty(
            (0, n_pcs),
            dtype=np.float32,
        )

    return np.concatenate(
        X_parts,
        axis=0,
    )


def metrics_at_best_f1_threshold(
    y_true,
    y_prob,
):
    precision, recall, thresholds = (
        precision_recall_curve(
            y_true,
            y_prob,
        )
    )

    precision_t = precision[:-1]
    recall_t = recall[:-1]

    denominator = (
        precision_t + recall_t
    )

    f1_scores = np.divide(
        2 * precision_t * recall_t,
        denominator,
        out=np.zeros_like(
            denominator
        ),
        where=denominator > 0,
    )

    best_idx = int(
        np.argmax(f1_scores)
    )

    best_threshold = float(
        thresholds[best_idx]
    )

    y_pred = (
        y_prob >= best_threshold
    )

    return {
        "best_threshold": best_threshold,
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Mesh
    # --------------------------------------------------------

    lat, lon = get_mesh_latlon(
        splits=6
    )

    all_nodes = (
        get_coarse_mesh_node_indices(
            fine_splits=6,
            coarse_splits=NODE_HIERARCHY_LEVEL,
        )
    )

    samples_per_t = len(all_nodes)

    print(
        "Nodes per timestep:",
        samples_per_t,
    )


    # --------------------------------------------------------
    # Load PCA scores
    # --------------------------------------------------------

    (
        pc_scores_by_year,
        timestamps_by_year,
    ) = load_pca_metadata(
        PC_SCORES_PATHS,
        TIMESTEP_FILES_TXTS,
    )

    graphcast_df = (
        build_graphcast_time_table(
            timestamps_by_year
        )
    )


    # --------------------------------------------------------
    # Match ClimateNet masks to PC-score timesteps
    # --------------------------------------------------------

    mask_files = sorted(
        glob(
            os.path.join(
                MASK_DIR,
                "*.nc",
            )
        )
    )

    y_parts = []
    matched_rows = []

    for i, mask_path in enumerate(mask_files):

        mask_time = parse_mask_timestamp(
            mask_path
        )

        row = nearest_graphcast_row(
            mask_time,
            graphcast_df,
            max_hours=MAX_TIME_DIFFERENCE_HOURS,
        )

        if row is None:
            continue

        y_nodes = load_mask_at_nodes(
            mask_path,
            lat,
            lon,
            all_nodes,
            label_mode=LABEL_MODE,
        )

        if LABEL_MODE != "soft":
            y_nodes = (
                y_nodes > 0
            ).astype(np.int8)

        y_parts.append(
            y_nodes
        )

        matched_rows.append({
            "mask_file": os.path.basename(
                mask_path
            ),
            "mask_time": mask_time,
            "graphcast_time": row["time"],
            "year": int(row["year"]),
            "t_idx": int(row["t_idx"]),
        })

        if (i + 1) % 100 == 0:
            print(
                f"Processed "
                f"{i + 1}/{len(mask_files)} masks"
            )


    if not y_parts:
        raise ValueError(
            "No masks matched GraphCast timestamps."
        )


    matched_df = pd.DataFrame(
        matched_rows
    )

    y = np.concatenate(
        y_parts,
        axis=0,
    ).astype(np.int8)


    matched_times = pd.to_datetime(
        matched_df["graphcast_time"].values
    )


    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    event_train_mask = (
        (matched_times >= TRAIN_START)
        &
        (matched_times < TRAIN_END)
    )

    event_val_mask = (
        (matched_times >= VAL_START)
        &
        (matched_times < VAL_END)
    )


    train_mask = np.repeat(
        event_train_mask,
        samples_per_t,
    )

    val_mask = np.repeat(
        event_val_mask,
        samples_per_t,
    )


    y_train = y[
        train_mask
    ]

    y_val = y[
        val_mask
    ]


    print()
    print(
        "Train events:",
        event_train_mask.sum(),
    )

    print(
        "Validation events:",
        event_val_mask.sum(),
    )

    print(
        "Train samples:",
        len(y_train),
    )

    print(
        "Validation samples:",
        len(y_val),
    )

    print(
        "Train positive rate:",
        np.mean(y_train),
    )

    print(
        "Validation positive rate:",
        np.mean(y_val),
    )


    # --------------------------------------------------------
    # Build first-200 PC matrices once
    # --------------------------------------------------------

    print()
    print(
        f"Building first {N_CANDIDATE_PCS} PCs..."
    )

    X_train_200 = build_X_first_n_pcs(
        matched_df,
        event_train_mask,
        pc_scores_by_year,
        all_nodes,
        N_CANDIDATE_PCS,
    )

    X_val_200 = build_X_first_n_pcs(
        matched_df,
        event_val_mask,
        pc_scores_by_year,
        all_nodes,
        N_CANDIDATE_PCS,
    )


    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    valid_train = (
        np.all(
            np.isfinite(X_train_200),
            axis=1,
        )
        &
        np.isfinite(y_train)
    )

    valid_val = (
        np.all(
            np.isfinite(X_val_200),
            axis=1,
        )
        &
        np.isfinite(y_val)
    )


    X_train_200 = X_train_200[
        valid_train
    ]

    y_train = y_train[
        valid_train
    ]

    X_val_200 = X_val_200[
        valid_val
    ]

    y_val = y_val[
        valid_val
    ]


    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Training set contains only one class."
        )

    if len(np.unique(y_val)) < 2:
        raise ValueError(
            "Validation set contains only one class."
        )


    # ============================================================
    # Fixed subsample for L1 feature selection only
    # ============================================================

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

    X_selector = X_train_200[
        selector_indices
    ]

    y_selector = y_train[
        selector_indices
    ]

    print()
    print(
        f"L1 selector uses {len(y_selector):,} "
        f"of {len(y_train):,} training samples"
    )

    print(
        "Selector positive rate:",
        np.mean(y_selector),
    )


    # --------------------------------------------------------
    # 200-PC L2 baseline
    # --------------------------------------------------------

    baseline_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
        ),
    )

    baseline_model.fit(
        X_train_200,
        y_train,
    )

    baseline_prob = (
        baseline_model
        .predict_proba(
            X_val_200
        )[:, 1]
    )

    baseline_ap = (
        average_precision_score(
            y_val,
            baseline_prob,
        )
    )

    baseline_auc = (
        roc_auc_score(
            y_val,
            baseline_prob,
        )
    )

    baseline_f1 = (
        metrics_at_best_f1_threshold(
            y_val,
            baseline_prob,
        )
    )


    print()
    print("=" * 70)
    print("200-PC L2 BASELINE")
    print("=" * 70)

    print(
        f"AP={baseline_ap:.4f} | "
        f"AUC={baseline_auc:.4f} | "
        f"F1={baseline_f1['f1']:.4f}"
    )


    # --------------------------------------------------------
    # L1 selection experiments
    # --------------------------------------------------------

    results = []

    print("Full train positive rate:", np.mean(y_train))
    print("Selector positive rate:", np.mean(y_selector))


    for C in C_VALUES:

        print()
        print("=" * 70)
        print(
            f"L1 C = {C:.8g}"
        )
        print("=" * 70)


        # ====================================================
        # 1. L1 selector
        # ====================================================

        selector = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l1",

                # saga is appropriate for L1 and larger datasets
                solver="saga",

                class_weight="balanced",

                C=float(C),

                max_iter=5000,

                tol=1e-4,

                random_state=0,
            ),
        )


        selector.fit(
            X_selector,
            y_selector,
        )


        l1_clf = (
            selector
            .named_steps[
                "logisticregression"
            ]
        )


        coef_l1 = (
            l1_clf.coef_[0]
            .astype(np.float32)
        )


        # Non-zero L1 coefficients define selected PCs
        selected_pcs = np.flatnonzero(
            np.abs(coef_l1) > 1e-8
        )


        n_selected = len(
            selected_pcs
        )


        print(
            "Selected PCs:",
            n_selected,
        )


        if n_selected == 0:

            print(
                "No PCs selected. Skipping L2 refit."
            )

            results.append({
                "C": C,
                "n_selected": 0,
                "selected_pc_numbers": "",
                "val_average_precision": np.nan,
                "val_roc_auc": np.nan,
                "val_f1": np.nan,
                "val_precision": np.nan,
                "val_recall": np.nan,
                "val_threshold": np.nan,

                "baseline_200_ap": baseline_ap,
                "ap_fraction_of_baseline": np.nan,
                "ap_drop_from_baseline": np.nan,
            })

            continue


        print(
            "PC numbers:",
            selected_pcs + 1
        )


        # ====================================================
        # 2. Select those PCs
        # ====================================================

        X_train_selected = (
            X_train_200[
                :,
                selected_pcs
            ]
        )

        X_val_selected = (
            X_val_200[
                :,
                selected_pcs
            ]
        )


        # ====================================================
        # 3. Fresh L2 logistic probe
        # ====================================================

        l2_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2000,
            ),
        )


        l2_model.fit(
            X_train_selected,
            y_train,
        )


        # ====================================================
        # 4. Validation
        # ====================================================

        y_val_prob = (
            l2_model
            .predict_proba(
                X_val_selected
            )[:, 1]
        )


        val_ap = (
            average_precision_score(
                y_val,
                y_val_prob,
            )
        )


        val_auc = (
            roc_auc_score(
                y_val,
                y_val_prob,
            )
        )


        val_best = (
            metrics_at_best_f1_threshold(
                y_val,
                y_val_prob,
            )
        )


        ap_fraction = (
            val_ap / baseline_ap
        )


        ap_drop = (
            baseline_ap - val_ap
        )


        print(
            f"L2 refit with "
            f"{n_selected} PCs | "
            f"AP={val_ap:.4f} | "
            f"AUC={val_auc:.4f} | "
            f"F1={val_best['f1']:.4f} | "
            f"AP/baseline={ap_fraction:.4f}"
        )


        # ====================================================
        # Save result row
        # ====================================================

        results.append({
            "C": float(C),

            "n_selected": int(
                n_selected
            ),

            "selected_pc_numbers": (
                ",".join(
                    str(i + 1)
                    for i in selected_pcs
                )
            ),

            "selected_pc_indices": (
                ",".join(
                    str(i)
                    for i in selected_pcs
                )
            ),

            "val_average_precision": float(
                val_ap
            ),

            "val_roc_auc": float(
                val_auc
            ),

            "val_f1": float(
                val_best["f1"]
            ),

            "val_precision": float(
                val_best["precision"]
            ),

            "val_recall": float(
                val_best["recall"]
            ),

            "val_threshold": float(
                val_best[
                    "best_threshold"
                ]
            ),

            "baseline_200_ap": float(
                baseline_ap
            ),

            "ap_fraction_of_baseline": float(
                ap_fraction
            ),

            "ap_drop_from_baseline": float(
                ap_drop
            ),
        })


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )


    results_path = os.path.join(
        OUT_DIR,
        "l1_selected_pcs_l2_validation_results.csv",
    )


    results_df.to_csv(
        results_path,
        index=False,
    )


    # Sort by number selected to make the sparsity curve easier to inspect
    sorted_df = (
        results_df
        .sort_values(
            [
                "n_selected",
                "val_average_precision",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )


    sorted_path = os.path.join(
        OUT_DIR,
        "l1_selected_pcs_l2_validation_results_sorted.csv",
    )


    sorted_df.to_csv(
        sorted_path,
        index=False,
    )


    # --------------------------------------------------------
    # Save baseline separately
    # --------------------------------------------------------

    baseline_path = os.path.join(
        OUT_DIR,
        "baseline_200pc_l2_validation.csv",
    )


    pd.DataFrame([
        {
            "n_pcs": N_CANDIDATE_PCS,
            "val_average_precision": baseline_ap,
            "val_roc_auc": baseline_auc,
            "val_f1": baseline_f1["f1"],
            "val_precision": baseline_f1["precision"],
            "val_recall": baseline_f1["recall"],
            "val_threshold": baseline_f1["best_threshold"],
        }
    ]).to_csv(
        baseline_path,
        index=False,
    )


    # --------------------------------------------------------
    # Print concise final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("L1 SELECTION SUMMARY")
    print("=" * 70)

    summary_cols = [
        "C",
        "n_selected",
        "val_average_precision",
        "ap_fraction_of_baseline",
        "val_f1",
    ]

    print(
        sorted_df[
            summary_cols
        ].to_string(
            index=False
        )
    )


    print()
    print(
        "200-PC baseline AP:",
        baseline_ap,
    )

    print(
        "Saved results:",
        results_path,
    )

    print(
        "Saved sorted results:",
        sorted_path,
    )

    print(
        "Saved baseline:",
        baseline_path,
    )


if __name__ == "__main__":
    main()