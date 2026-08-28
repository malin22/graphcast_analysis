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

WEATHER_FEATURE = "TC"
NODE_HIERARCHY_LEVEL = 6
LABEL_MODE = "intersection"

MAX_TIME_DIFFERENCE_HOURS = 3

# Ranking is based on the existing 200-PC probe
RANKING_BASE_N_PCS = 512

TOP_K_COUNTS = [
    1, 2, 3, 5, 8, 10,
    15, 20, 25, 30, 40, 50,
    75, 100, 150, 200, 250, 300, 400, 512,
]


# ------------------------------------------------------------
# Train / validation split
#
# We deliberately do NOT use 2021 here.
# ------------------------------------------------------------

TRAIN_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2020-11-01")

VAL_START = pd.Timestamp("2020-11-01")
VAL_END = pd.Timestamp("2021-01-01")


# ------------------------------------------------------------
# Existing PC scores
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# IMPORTANT:
# Change this filename to the NEW 200-PC probe trained only
# on Jan 2019 - Oct 2020.
# ------------------------------------------------------------

RANKING_PROBE_PATH = (
    f"plots/malins_experiments/logistic_regression/"
    f"{WEATHER_FEATURE}/Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/PCA/"
    f"probe_direction_{WEATHER_FEATURE}_PCA_intersection_"
    f"M{NODE_HIERARCHY_LEVEL}_{RANKING_BASE_N_PCS}_features_"
    f"2019_2020_train_only.npz"
)


OUT_DIR = (
    f"plots/malins_experiments/logistic_regression/"
    f"{WEATHER_FEATURE}/Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"PCA_ranked_feature_selection/"
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
                "Could not match coarse vertex "
                "to fine mesh."
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

    return mask_nodes.astype(
        np.float32
    )


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

    for year in sorted(
        pc_score_paths
    ):

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


def build_X_for_selected_pcs(
    matched_df,
    split_mask_events,
    pc_scores_by_year,
    all_nodes,
    selected_pcs,
):

    X_parts = []

    selected_rows = matched_df.loc[
        split_mask_events
    ]

    for _, row in (
        selected_rows.iterrows()
    ):

        year = int(row["year"])
        t_idx = int(row["t_idx"])

        # [nodes, all PCs]
        X_t = pc_scores_by_year[
            year
        ][
            t_idx,
            all_nodes,
            :
        ]

        # Select arbitrary PCs
        X_t = X_t[
            :,
            selected_pcs
        ]

        X_parts.append(
            np.asarray(
                X_t,
                dtype=np.float32,
            )
        )

    if not X_parts:
        return np.empty(
            (
                0,
                len(selected_pcs),
            ),
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


def fit_and_evaluate(
    X_train,
    y_train,
    X_val,
    y_val,
):

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
        ),
    )

    model.fit(
        X_train,
        y_train,
    )

    y_val_prob = model.predict_proba(
        X_val
    )[:, 1]

    ap = average_precision_score(
        y_val,
        y_val_prob,
    )

    if len(
        np.unique(y_val)
    ) == 2:

        auc = roc_auc_score(
            y_val,
            y_val_prob,
        )

    else:
        auc = np.nan

    best = metrics_at_best_f1_threshold(
        y_val,
        y_val_prob,
    )

    return {
        "val_average_precision": ap,
        "val_roc_auc": auc,
        "val_f1": best["f1"],
        "val_precision": best["precision"],
        "val_recall": best["recall"],
        "val_threshold": best["best_threshold"],
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

    samples_per_t = len(
        all_nodes
    )

    print(
        "Nodes per timestep:",
        samples_per_t,
    )


    # --------------------------------------------------------
    # Load PC scores + timestamps
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
    # Match ClimateNet masks
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

    for i, mask_path in enumerate(
        mask_files
    ):

        mask_time = (
            parse_mask_timestamp(
                mask_path
            )
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
        matched_df[
            "graphcast_time"
        ].values
    )


    # --------------------------------------------------------
    # Train / validation splits
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
    # Load existing 200-PC ranking probe
    # --------------------------------------------------------

    print()
    print(
        "Loading ranking probe:"
    )

    print(
        RANKING_PROBE_PATH
    )

    ranking_probe = np.load(
        RANKING_PROBE_PATH
    )

    coef_200 = (
        ranking_probe[
            "coef_z"
        ]
        .astype(np.float32)
    )


    if len(coef_200) != RANKING_BASE_N_PCS:
        raise ValueError(
            f"Expected "
            f"{RANKING_BASE_N_PCS} coefficients, "
            f"got {len(coef_200)}."
        )


    # Largest |coefficient| first
    pc_ranking = np.argsort(
        np.abs(coef_200)
    )[::-1]


    # --------------------------------------------------------
    # Save / print ranking
    # --------------------------------------------------------

    ranking_rows = []

    print()
    print("=" * 70)
    print(f"{RANKING_BASE_N_PCS}-PC COEFFICIENT RANKING")
    print("=" * 70)

    for rank, pc_idx in enumerate(
        pc_ranking,
        start=1,
    ):

        ranking_rows.append({
            "rank": rank,

            # Human-readable PC number
            "pc_number": int(
                pc_idx + 1
            ),

            # Actual NumPy index
            "pc_index": int(
                pc_idx
            ),

            "coefficient": float(
                coef_200[pc_idx]
            ),

            "abs_coefficient": float(
                abs(
                    coef_200[pc_idx]
                )
            ),
        })


        if rank <= 50:

            print(
                f"{rank:>3d}. "
                f"PC{pc_idx + 1:<3d} "
                f"coef="
                f"{coef_200[pc_idx]:+.6f} "
                f"|coef|="
                f"{abs(coef_200[pc_idx]):.6f}"
            )


    ranking_df = pd.DataFrame(
        ranking_rows
    )

    ranking_df.to_csv(
        os.path.join(
            OUT_DIR,
            f"pc_coefficient_ranking_from_{RANKING_BASE_N_PCS}.csv",
        ),
        index=False,
    )


    # --------------------------------------------------------
    # Build first 200 PCs ONCE
    #
    # Then all k experiments simply slice these matrices.
    # Much cheaper than rebuilding them for every k.
    # --------------------------------------------------------

    first_200 = np.arange(
        RANKING_BASE_N_PCS
    )

    print()
    print(
        "Building 200-PC training matrix..."
    )

    X_train_200 = (
        build_X_for_selected_pcs(
            matched_df,
            event_train_mask,
            pc_scores_by_year,
            all_nodes,
            first_200,
        )
    )

    print(
        "Building 200-PC validation matrix..."
    )

    X_val_200 = (
        build_X_for_selected_pcs(
            matched_df,
            event_val_mask,
            pc_scores_by_year,
            all_nodes,
            first_200,
        )
    )


    # --------------------------------------------------------
    # Finite filtering
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


    # --------------------------------------------------------
    # Top-k experiments
    # --------------------------------------------------------

    results = []


    for k in TOP_K_COUNTS:

        if k > RANKING_BASE_N_PCS:
            continue


        print()
        print("=" * 70)
        print(
            f"k = {k}"
        )
        print("=" * 70)


        # ====================================================
        # A) Ranked top-k PCs
        # ====================================================

        ranked_indices = (
            pc_ranking[:k]
        )


        X_train_ranked = (
            X_train_200[
                :,
                ranked_indices
            ]
        )

        X_val_ranked = (
            X_val_200[
                :,
                ranked_indices
            ]
        )


        ranked_metrics = (
            fit_and_evaluate(
                X_train_ranked,
                y_train,
                X_val_ranked,
                y_val,
            )
        )


        results.append({
            "selection": "ranked_top_k",
            "k": k,

            "selected_pcs": ",".join(
                str(i + 1)
                for i in ranked_indices
            ),

            **ranked_metrics,
        })


        # ====================================================
        # B) First-k PCs baseline
        # ====================================================

        first_k_indices = np.arange(
            k
        )


        X_train_first = (
            X_train_200[
                :,
                first_k_indices
            ]
        )

        X_val_first = (
            X_val_200[
                :,
                first_k_indices
            ]
        )


        first_metrics = (
            fit_and_evaluate(
                X_train_first,
                y_train,
                X_val_first,
                y_val,
            )
        )


        results.append({
            "selection": "first_k",
            "k": k,

            "selected_pcs": ",".join(
                str(i + 1)
                for i in first_k_indices
            ),

            **first_metrics,
        })


        # ----------------------------------------------------
        # Console comparison
        # ----------------------------------------------------

        print(
            f"RANKED top-{k:>3d} | "
            f"AP={ranked_metrics['val_average_precision']:.4f} | "
            f"AUC={ranked_metrics['val_roc_auc']:.4f} | "
            f"F1={ranked_metrics['val_f1']:.4f}"
        )

        print(
            f"FIRST  top-{k:>3d} | "
            f"AP={first_metrics['val_average_precision']:.4f} | "
            f"AUC={first_metrics['val_roc_auc']:.4f} | "
            f"F1={first_metrics['val_f1']:.4f}"
        )


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )


    results_path = os.path.join(
        OUT_DIR,
        "ranked_vs_first_k_validation_results.csv",
    )


    results_df.to_csv(
        results_path,
        index=False,
    )


    # --------------------------------------------------------
    # Useful summary table
    # --------------------------------------------------------

    summary = results_df.pivot(
        index="k",
        columns="selection",
        values="val_average_precision",
    )


    if (
        "ranked_top_k" in summary.columns
        and
        "first_k" in summary.columns
    ):
        summary[
            "ranked_minus_first_AP"
        ] = (
            summary["ranked_top_k"]
            -
            summary["first_k"]
        )


    summary_path = os.path.join(
        OUT_DIR,
        "ranked_vs_first_k_AP_summary.csv",
    )


    summary.to_csv(
        summary_path
    )


    print()
    print("=" * 70)
    print("VALIDATION AP SUMMARY")
    print("=" * 70)

    print(
        summary.to_string()
    )


    print()
    print(
        "Saved results:",
        results_path,
    )

    print(
        "Saved summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()