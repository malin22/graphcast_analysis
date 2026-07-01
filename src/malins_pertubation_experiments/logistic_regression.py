import os
import re
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

import joblib

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from graphcast import icosahedral_mesh


# =====================
# CONFIG
# =====================

WEATHER_FEATURE = "AR"  # "AR" or "TC"
REPRESENTATION = "PCA"  # "raw_activations" or "PCA"
NODE_HIERARCHY_LEVEL = 6

FEATURE_COUNTS_RAW = [512]
PC_COUNTS = [5, 10, 25, 50, 100, 200, 400]

# Train/validation/test design:
# - train probe on Jan-Oct 2020
# - use Nov-Dec 2020 only for validation/model-selection diagnostics
# - keep all of 2021 held out for final probe evaluation and later perturbation experiments
TRAIN_START = pd.Timestamp("2020-01-01")
TRAIN_END = pd.Timestamp("2021-01-01")
TEST_START = pd.Timestamp("2021-01-01")
TEST_END = pd.Timestamp("2022-01-01")



# Raw activation directories. Adjust these paths if your 2020 activations live elsewhere.
ACTS_DIRS = {
    2020: "/share/prj-4d/graphcast_shared/data/graphcast_activation_2020",
    2021: "/share/prj-4d/graphcast_shared/data/graphcast_activation_2021",
}

# PCA inputs, if REPRESENTATION == "PCA".
PC_SCORES_PATHS = {
    2020: "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/pc_scores_2020_per_timestep.npy",
    2021: "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/pc_scores_2021_per_timestep.npy",
}
TIMESTEP_FILES_TXTS = {
    2020: "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/pc_scores_2020_per_timestep_files.txt",
    2021: "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/pc_scores_2021_per_timestep_files.txt",
}

MASK_DIR = f"/share/prj-4d/graphcast_shared/data/ClimateNetLarge/{WEATHER_FEATURE}_labels_cleaned"

OUT_DIR = (
    f"plots/malins_experiments/logistic_regression/"
    f"{WEATHER_FEATURE}/{REPRESENTATION}"
)
os.makedirs(OUT_DIR, exist_ok=True)

LABEL_MODE = "intersection"
# "intersection" = both annotators agree
# "union" = at least one annotator
# "soft" = mean annotation, not recommended for logistic classification yet

MAX_TIME_DIFFERENCE_HOURS = 3

# Evaluation thresholds for reporting. Do not tune these on 2021.
THRESHOLDS = [0.1, 0.2, 0.3, 0.5]

# =====================
# HELPERS
# =====================

def event_region_metrics(y_true, y_prob, event_id, threshold=0.5):
    rows = []

    for eid in np.unique(event_id):
        m = event_id == eid

        yt = y_true[m].astype(bool)
        yp = y_prob[m]

        if yt.sum() == 0:
            continue

        pred = yp >= threshold

        true_area = int(yt.sum())
        pred_area = int(pred.sum())
        overlap = int((yt & pred).sum())
        union = int((yt | pred).sum())

        rows.append({
            "event_id": int(eid),
            "threshold": threshold,
            "true_area": true_area,
            "pred_area": pred_area,
            "overlap_area": overlap,
            "event_found": int(overlap > 0),
            "coverage_recall": overlap / true_area,
            "precision": overlap / pred_area if pred_area > 0 else 0.0,
            "iou": overlap / union if union > 0 else 0.0,
            "area_ratio": pred_area / true_area if true_area > 0 else np.nan,
            "mean_prob_inside": float(yp[yt].mean()),
            "mean_prob_outside": float(yp[~yt].mean()),
            "max_prob_inside": float(yp[yt].max()),
            "max_prob_outside": float(yp[~yt].max()),
        })

    return pd.DataFrame(rows)


def load_activations(path):
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    if x.ndim != 2:
        raise ValueError(f"Expected [nodes, features], got shape {x.shape}")

    return x.astype(np.float32)

def parse_activation_timestamp(path):
    fname = os.path.basename(path)
    m = re.search(r"t(\d{4})-(\d{2})-(\d{2})T(\d{2})", fname)
    if not m:
        raise ValueError(f"Could not parse timestamp from {fname}")
    y, mo, d, h = map(int, m.groups())
    return pd.Timestamp(y, mo, d, h)


def load_timestamps(files_txt):
    with open(files_txt, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    timestamps = pd.to_datetime([parse_activation_timestamp(p) for p in files])
    return files, timestamps


def parse_mask_timestamp(path):
    fname = os.path.basename(path).replace(".nc", "")
    return pd.Timestamp(fname)


def vertices_to_latlon(vertices):
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0])) % 360
    return lat, lon


def get_mesh_latlon(splits=6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=splits
    )
    vertices = meshes[splits].vertices
    return vertices_to_latlon(vertices)


def get_coarse_mesh_node_indices(fine_splits=6, coarse_splits=4, decimals=8):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=fine_splits
    )

    fine_vertices = meshes[fine_splits].vertices
    coarse_vertices = meshes[coarse_splits].vertices

    fine_keys = {
        tuple(np.round(v, decimals)): i
        for i, v in enumerate(fine_vertices)
    }

    coarse_indices = []
    missing = 0

    for v in coarse_vertices:
        key = tuple(np.round(v, decimals))
        if key in fine_keys:
            coarse_indices.append(fine_keys[key])
        else:
            missing += 1

    if missing > 0:
        raise ValueError(f"Could not match {missing} coarse vertices to fine mesh")

    return np.array(coarse_indices, dtype=int)


def nearest_graphcast_index(mask_time, graphcast_times, max_hours=3):
    diffs = np.abs(graphcast_times - mask_time)
    idx = int(np.argmin(diffs))

    if diffs[idx] > pd.Timedelta(hours=max_hours):
        return None

    return idx


def load_mask_at_nodes(mask_path, lat, lon, node_indices, label_mode="intersection"):
    ds = xr.open_dataset(mask_path)

    label = ds["label"]

    if label_mode == "intersection":
        mask = label.min("annotator")
    elif label_mode == "union":
        mask = label.max("annotator")
    elif label_mode == "soft":
        mask = label.mean("annotator")
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    node_lat = xr.DataArray(lat[node_indices], dims="sample")
    node_lon = xr.DataArray(lon[node_indices], dims="sample")

    mask_nodes = mask.interp(
        latitude=node_lat,
        longitude=node_lon,
        method="nearest",
    ).values

    return mask_nodes.astype(np.float32)


def safe_metrics(y_true, y_prob, threshold=0.5):
    y_pred = y_prob >= threshold

    out = {
        "average_precision": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "positive_rate": float(np.mean(y_true)),
        "n_positive": int(np.sum(y_true)),
        "n_total": int(len(y_true)),
    }

    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        out["roc_auc"] = np.nan

    return out



def load_raw_activation_years(acts_dirs):
    """Collect raw activation files and timestamps without loading all activations into RAM."""
    all_files = []

    for year, acts_dir in sorted(acts_dirs.items()):
        files = sorted(glob(os.path.join(acts_dir, "*.npy")))
        print(f"Found {len(files)} raw activation files for {year} in {acts_dir}")
        all_files.extend(files)

    valid_files = []
    for f in all_files:
        X_t = load_activations(f)
        if np.isnan(X_t).any():
            print(f"Skipping NaN activation file: {os.path.basename(f)}")
            continue
        valid_files.append(f)

    act_files = sorted(valid_files, key=parse_activation_timestamp)
    graphcast_times = pd.to_datetime([parse_activation_timestamp(p) for p in act_files])

    if len(act_files) == 0:
        raise ValueError("No valid raw activation files found.")

    example = load_activations(act_files[0])
    max_features = example.shape[1]

    print(f"Using {len(act_files)} valid raw activation files.")
    print("Raw activation feature dimension:", max_features)

    return act_files, graphcast_times, max_features


def load_pca_years(pc_score_paths, timestep_files_txts, all_nodes, samples_per_t):
    """Load PCA scores from several years and return flattened X plus timestamps."""
    X_parts = []
    time_parts = []
    max_features = None

    for year in sorted(pc_score_paths):
        pc_scores_path = pc_score_paths[year]
        timestep_files_txt = timestep_files_txts[year]

        _, timestamps = load_timestamps(timestep_files_txt)
        pc_scores = np.load(pc_scores_path, mmap_mode="r")
        T, N, K = pc_scores.shape

        if len(timestamps) != T:
            raise ValueError(f"{len(timestamps)} timestamps but {T} PC-score timesteps for {year}")

        year_max_features = min(max(PC_COUNTS), K)
        if max_features is None:
            max_features = year_max_features
        else:
            max_features = min(max_features, year_max_features)

        X_year = np.asarray(pc_scores[:, all_nodes, :max_features], dtype=np.float32)
        X_parts.append(X_year)
        time_parts.append(pd.to_datetime(timestamps))

        print(f"PC scores {year}:", pc_scores.shape)

    X_all = np.concatenate(X_parts, axis=0)
    graphcast_times = pd.to_datetime(np.concatenate([t.values for t in time_parts]))
    order = np.argsort(graphcast_times.values)
    X_all = X_all[order]
    graphcast_times = graphcast_times[order]

    X_all = X_all.reshape(len(graphcast_times) * samples_per_t, max_features)
    print("Combined PC-score matrix:", X_all.shape)
    print("Using max PCs:", max_features)
    return X_all, graphcast_times, max_features


def split_name_masks(time_index, valid):
    train_mask = (time_index >= TRAIN_START) & (time_index < TRAIN_END) & valid
    test_mask = (time_index >= TEST_START) & (time_index < TEST_END) & valid
    return train_mask, test_mask


def evaluate_split(model, X_split, y_split, split_name, threshold=0.5):
    y_prob = model.predict_proba(X_split)[:, 1]
    metrics = safe_metrics(y_split, y_prob, threshold=threshold)
    metrics = {f"{split_name}_{k}": v for k, v in metrics.items()}
    return y_prob, metrics

# =====================
# MAIN
# =====================

def main():

    lat, lon = get_mesh_latlon(splits=6)

    coarse_nodes = get_coarse_mesh_node_indices(
        fine_splits=6,
        coarse_splits=NODE_HIERARCHY_LEVEL,
    )

    all_nodes = coarse_nodes
    samples_per_t = len(all_nodes)

    print("Nodes per timestep:", samples_per_t)
    print(f"Using M{NODE_HIERARCHY_LEVEL} mesh nodes")

    if REPRESENTATION == "raw_activations":
        act_files, graphcast_times, max_features = load_raw_activation_years(ACTS_DIRS)
        X_all = None
        feature_counts = FEATURE_COUNTS_RAW

    elif REPRESENTATION == "PCA":
        X_all, graphcast_times, max_features = load_pca_years(
            PC_SCORES_PATHS,
            TIMESTEP_FILES_TXTS,
            all_nodes,
            samples_per_t,
        )
        feature_counts = PC_COUNTS

    else:
        raise ValueError(f"Unknown REPRESENTATION: {REPRESENTATION}")



    

    mask_files = sorted(glob(os.path.join(MASK_DIR, "*.nc")))

    y_parts = []
    x_parts = []
    matched_rows = []

    event_parts = []

    for i, mask_path in enumerate(mask_files):
        mask_time = parse_mask_timestamp(mask_path)
        t_idx = nearest_graphcast_index(
            mask_time,
            graphcast_times,
            max_hours=MAX_TIME_DIFFERENCE_HOURS,
        )

        if t_idx is None:
            continue


        y_nodes = load_mask_at_nodes(
            mask_path,
            lat,
            lon,
            all_nodes,
            label_mode=LABEL_MODE,
        )

        if LABEL_MODE != "soft":
            y_nodes = (y_nodes > 0).astype(np.int8)

        if REPRESENTATION == "raw_activations":
            X_t_full = load_activations(act_files[t_idx])
            X_t = X_t_full[all_nodes, :max_features]
        else:
            start = t_idx * samples_per_t
            end = (t_idx + 1) * samples_per_t
            X_t = X_all[start:end, :max_features]

        y_parts.append(y_nodes)
        x_parts.append(X_t)

        event_idx = len(matched_rows)

        event_parts.append(
            np.full(samples_per_t, event_idx, dtype=np.int32)
        )

        matched_rows.append({
            "mask_file": os.path.basename(mask_path),
            "mask_time": mask_time,
            "graphcast_time": graphcast_times[t_idx],
            "time_difference_hours": abs(graphcast_times[t_idx] - mask_time).total_seconds() / 3600,
            "positive_nodes": int(np.sum(y_nodes > 0)),
            "positive_fraction": float(np.mean(y_nodes > 0)),
        })

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(mask_files)} mask files")

    if not y_parts:
        raise ValueError(f"No {WEATHER_FEATURE} mask files matched GraphCast timestamps.")

    X = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0).astype(np.int8)
    event_id = np.concatenate(event_parts)

    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_csv(os.path.join(OUT_DIR, "matched_files.csv"), index=False)

    matched_times = pd.to_datetime(matched_df["graphcast_time"].values)
    time_index = np.repeat(matched_times.values, samples_per_t)
    time_index = pd.to_datetime(time_index)

    valid_X = np.all(np.isfinite(X), axis=1)
    valid_y = np.isfinite(y)
    valid = valid_X & valid_y

    train_mask, test_mask = split_name_masks(time_index, valid)

    print(f"Matched {WEATHER_FEATURE} files:", len(matched_df))
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Overall positive rate:", np.mean(y))
    print(f"Train window: {TRAIN_START.date()} to {TRAIN_END.date()} exclusive")
    print(f"Held-out test window: {TEST_START.date()} to {TEST_END.date()} exclusive")
    print("Train samples:", train_mask.sum())
    print("Test samples:", test_mask.sum())
    print("Train positives:", y[train_mask].sum())
    print("Test positives:", y[test_mask].sum())

    results = []

    for n_features in feature_counts:


        if n_features > X.shape[1]:
            print(f"Skipping {n_features}: only {X.shape[1]} features available")
            continue

        X_train = X[train_mask, :n_features]
        X_test = X[test_mask, :n_features]

        y_train = y[train_mask]
        y_test = y[test_mask]

        if len(np.unique(y_train)) < 2:
            print(f"Skipping {n_features} features: train set has only one class")
            continue


        if len(np.unique(y_test)) < 2:
            print(f"Skipping {n_features} features: held-out 2021 test set has only one class")
            continue

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
            ),
        )

        model.fit(X_train, y_train)


        direction_training_window = "2020_train_only"

        # =====================
        # SAVE PROBE DIRECTION FOR INTERVENTION
        # =====================

        scaler = model.named_steps["standardscaler"]
        clf = model.named_steps["logisticregression"]

        coef_z = clf.coef_[0].astype(np.float32)
        coef_z_unit = coef_z / np.linalg.norm(coef_z)

        direction_out = os.path.join(
            OUT_DIR,
            f"probe_direction_{WEATHER_FEATURE}_{REPRESENTATION}_"
            f"{LABEL_MODE}_M{NODE_HIERARCHY_LEVEL}_{n_features}_features_"
            f"{direction_training_window}.npz",
        )

        save_dict = {
            "coef_z": coef_z,
            "coef_z_unit": coef_z_unit,
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "intercept": clf.intercept_.astype(np.float32),
            "n_features": np.array([n_features]),
            "direction_training_window": np.array([direction_training_window]),
            "train_start": np.array([str(TRAIN_START)]),
            "train_end": np.array([str(TRAIN_END)]),
            "test_start": np.array([str(TEST_START)]),
            "test_end": np.array([str(TEST_END)]),
        }

        if REPRESENTATION == "raw_activations":
            # If z = (x - mean) / scale,
            # then adding gamma * coef_z_unit in standardized space corresponds to
            # adding gamma * scale * coef_z_unit in raw activation space.
            direction_raw_delta = scaler.scale_.astype(np.float32) * coef_z_unit
            save_dict["direction_raw_delta"] = direction_raw_delta

        elif REPRESENTATION == "PCA":
            # This direction lives in PCA-score space, not raw GraphCast activation space.
            # Useful for analysis, but not directly insertable into GraphCast unless mapped back.
            save_dict["direction_pc_delta"] = coef_z_unit

        np.savez(direction_out, **save_dict)

        model_out = os.path.join(
            OUT_DIR,
            f"logistic_probe_model_{WEATHER_FEATURE}_{REPRESENTATION}_"
            f"{LABEL_MODE}_M{NODE_HIERARCHY_LEVEL}_{n_features}_features_"
            f"{direction_training_window}.joblib",
        )
        joblib.dump(model, model_out)

        print("Saved probe direction:", direction_out)
        print("Saved logistic model:", model_out)

        y_test_prob, test_metrics = evaluate_split(model, X_test, y_test, "test", threshold=0.5)

        event_dfs = []

        for threshold in THRESHOLDS:
            tmp = event_region_metrics(
                y_true=y_test,
                y_prob=y_test_prob,
                event_id=event_id[test_mask],
                threshold=threshold,
            )
            tmp["target"] = WEATHER_FEATURE
            tmp["representation"] = REPRESENTATION
            tmp["n_features"] = n_features
            tmp["label_mode"] = LABEL_MODE
            event_dfs.append(tmp)

        event_df = pd.concat(event_dfs, ignore_index=True)

        event_meta = matched_df.reset_index().rename(columns={"index": "event_id"})
        event_df = event_df.merge(event_meta, on="event_id", how="left")

        event_out = os.path.join(
            OUT_DIR,
            f"event_region_metrics_{LABEL_MODE}_M{NODE_HIERARCHY_LEVEL}_"
            f"{n_features}_features_2021_test_max_{MAX_TIME_DIFFERENCE_HOURS}hour.csv",
        )

        event_df.to_csv(event_out, index=False)

        summary = event_df.groupby("threshold")[[
            "event_found",
            "coverage_recall",
            "precision",
            "iou",
            "area_ratio",
        ]].mean()

        print("\nEvent-level summary:")
        print(summary)
        print("Saved event-level metrics:", event_out)

        row = {
            "target": WEATHER_FEATURE,
            "label_mode": LABEL_MODE,
            "n_features": n_features,
            "model": "logistic_l2_balanced",
            "direction_training_window": direction_training_window,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "train_positive_rate": float(np.mean(y_train)),
            "test_positive_rate": float(np.mean(y_test)),
        }

        row.update(test_metrics)
        results.append(row)

        print(
            f"{WEATHER_FEATURE} | Features={n_features:>3d} | "
            f"TEST AP={test_metrics['test_average_precision']:.3f} | "
            f"TEST AUC={test_metrics['test_roc_auc']:.3f} | "
            f"TEST F1={test_metrics['test_f1']:.3f} | "
            f"TEST P={test_metrics['test_precision']:.3f} | "
            f"TEST R={test_metrics['test_recall']:.3f}"
        )

    df = pd.DataFrame(results)

    out_csv = os.path.join(
        OUT_DIR,
        f"logistic_probe_2020train_2021test_{LABEL_MODE}_M{NODE_HIERARCHY_LEVEL}_max_{MAX_TIME_DIFFERENCE_HOURS}hour.csv",
    )
    df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print("Saved matched file table:", os.path.join(OUT_DIR, "matched_files.csv"))


if __name__ == "__main__":
    main()