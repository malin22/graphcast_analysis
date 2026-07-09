import os
import re
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr
import joblib

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)

from graphcast import icosahedral_mesh


# =====================
# CONFIG
# =====================

WEATHER_FEATURE = "AR"      # "TC" or "AR"
REPRESENTATION = "raw_activations"
NODE_HIERARCHY_LEVEL = 6
N_FEATURES = 512

LABEL_MODE = "intersection"
MAX_TIME_DIFFERENCE_HOURS = 3

TEST_START = pd.Timestamp("2021-01-01")
TEST_END = pd.Timestamp("2022-01-01")

ACTS_DIR = "/share/prj-4d/graphcast_shared/data/graphcast_activation_2021"

MASK_DIR = (
    f"/share/prj-4d/graphcast_shared/data/ClimateNetLarge/"
    f"{WEATHER_FEATURE}_labels_cleaned"
)

MODEL_PATH = (
    f"plots/malins_experiments/logistic_regression/"
    f"{WEATHER_FEATURE}/{REPRESENTATION}/"
    f"logistic_probe_model_{WEATHER_FEATURE}_{REPRESENTATION}_"
    f"{LABEL_MODE}_M{NODE_HIERARCHY_LEVEL}_{N_FEATURES}_features_"
    f"2020_train_only.joblib"
)

OUT_DIR = (
    f"plots/malins_experiments/logistic_regression/"
    f"{WEATHER_FEATURE}/{REPRESENTATION}/evaluation"
)
os.makedirs(OUT_DIR, exist_ok=True)


# =====================
# HELPERS
# =====================

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


def get_coarse_mesh_node_indices(fine_splits=6, coarse_splits=6, decimals=8):
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

    for v in coarse_vertices:
        key = tuple(np.round(v, decimals))
        if key not in fine_keys:
            raise ValueError("Could not match coarse vertex to fine mesh")
        coarse_indices.append(fine_keys[key])

    return np.array(coarse_indices, dtype=int)


def nearest_graphcast_index(mask_time, graphcast_times, max_hours=3):
    diffs = np.abs(graphcast_times - mask_time)
    idx = int(np.argmin(diffs))

    if diffs[idx] > pd.Timedelta(hours=max_hours):
        return None

    return idx


def load_mask_at_nodes(mask_path, lat, lon, node_indices, label_mode):
    ds = xr.open_dataset(mask_path)
    label = ds["label"]

    if label_mode == "intersection":
        mask = label.min("annotator")
    elif label_mode == "union":
        mask = label.max("annotator")
    elif label_mode == "soft":
        mask = label.mean("annotator")
    else:
        raise ValueError(f"Unknown label mode: {label_mode}")

    node_lat = xr.DataArray(lat[node_indices], dims="sample")
    node_lon = xr.DataArray(lon[node_indices], dims="sample")

    mask_nodes = mask.interp(
        latitude=node_lat,
        longitude=node_lon,
        method="nearest",
    ).values

    return (mask_nodes > 0).astype(np.int8)


def event_region_metrics(y_true, y_prob, event_id, threshold):
    rows = []

    for eid in np.unique(event_id):
        m = event_id == eid

        yt = y_true[m].astype(bool)
        yp = y_prob[m]
        pred = yp >= threshold

        if yt.sum() == 0:
            continue

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


# =====================
# MAIN
# =====================

def main():
    print("Loading model:")
    print(MODEL_PATH)

    model = joblib.load(MODEL_PATH)

    lat, lon = get_mesh_latlon(splits=6)

    node_indices = get_coarse_mesh_node_indices(
        fine_splits=6,
        coarse_splits=NODE_HIERARCHY_LEVEL,
    )

    samples_per_t = len(node_indices)

    act_files = sorted(glob(os.path.join(ACTS_DIR, "*.npy")))
    act_files = sorted(act_files, key=parse_activation_timestamp)

    graphcast_times = pd.to_datetime([
        parse_activation_timestamp(p) for p in act_files
    ])

    mask_files = sorted(glob(os.path.join(MASK_DIR, "*.nc")))

    X_parts = []
    y_parts = []
    event_parts = []
    matched_rows = []

    for mask_path in mask_files:
        mask_time = parse_mask_timestamp(mask_path)

        if not (TEST_START <= mask_time < TEST_END):
            continue

        t_idx = nearest_graphcast_index(
            mask_time,
            graphcast_times,
            max_hours=MAX_TIME_DIFFERENCE_HOURS,
        )

        if t_idx is None:
            continue

        X_t = load_activations(act_files[t_idx])
        X_t = X_t[node_indices, :N_FEATURES]

        y_t = load_mask_at_nodes(
            mask_path,
            lat,
            lon,
            node_indices,
            LABEL_MODE,
        )

        event_id = len(matched_rows)

        X_parts.append(X_t)
        y_parts.append(y_t)
        event_parts.append(np.full(samples_per_t, event_id, dtype=np.int32))

        matched_rows.append({
            "event_id": event_id,
            "mask_file": os.path.basename(mask_path),
            "mask_time": mask_time,
            "graphcast_time": graphcast_times[t_idx],
            "time_difference_hours": abs(
                graphcast_times[t_idx] - mask_time
            ).total_seconds() / 3600,
            "positive_nodes": int(y_t.sum()),
            "positive_fraction": float(y_t.mean()),
        })

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    event_id = np.concatenate(event_parts)
    matched_df = pd.DataFrame(matched_rows)

    print("X:", X.shape)
    print("y:", y.shape)
    print("positive rate:", y.mean())
    print("positives:", y.sum())

    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)

    X = X[valid]
    y = y[valid]
    event_id = event_id[valid]

    y_prob = model.predict_proba(X)[:, 1]

    # =====================
    # STANDARD RANKING METRICS
    # =====================

    ap = average_precision_score(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    print("\nRanking metrics:")
    print(f"Average precision: {ap:.6f}")
    print(f"ROC AUC:           {auc:.6f}")

    # =====================
    # THRESHOLD SWEEP
    # =====================

    precision, recall, thresholds = precision_recall_curve(y, y_prob)

    # precision/recall have length len(thresholds)+1
    precision_t = precision[:-1]
    recall_t = recall[:-1]

    f1_t = (
        2 * precision_t * recall_t
        / np.maximum(precision_t + recall_t, 1e-12)
    )

    best_idx = int(np.argmax(f1_t))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_t[best_idx])
    best_precision = float(precision_t[best_idx])
    best_recall = float(recall_t[best_idx])

    print("\nBest threshold by F1:")
    print(f"threshold: {best_threshold:.8f}")
    print(f"F1:        {best_f1:.6f}")
    print(f"precision: {best_precision:.6f}")
    print(f"recall:    {best_recall:.6f}")

    threshold_rows = []

    for thr in thresholds:
        pred = y_prob >= thr

        threshold_rows.append({
            "threshold": float(thr),
            "f1": f1_score(y, pred, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "pred_positive_rate": float(pred.mean()),
            "n_pred_positive": int(pred.sum()),
        })

    threshold_df = pd.DataFrame(threshold_rows)

    threshold_out = os.path.join(
        OUT_DIR,
        f"threshold_sweep_{WEATHER_FEATURE}_{LABEL_MODE}_"
        f"M{NODE_HIERARCHY_LEVEL}_{N_FEATURES}_features_2021.csv",
    )
    threshold_df.to_csv(threshold_out, index=False)

    # =====================
    # FIXED THRESHOLD COMPARISON
    # =====================

    fixed_thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, best_threshold]

    fixed_rows = []

    for thr in fixed_thresholds:
        pred = y_prob >= thr

        fixed_rows.append({
            "threshold": float(thr),
            "average_precision": ap,
            "roc_auc": auc,
            "f1": f1_score(y, pred, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "positive_rate": float(y.mean()),
            "pred_positive_rate": float(pred.mean()),
            "n_positive": int(y.sum()),
            "n_pred_positive": int(pred.sum()),
            "n_total": int(len(y)),
        })

    fixed_df = pd.DataFrame(fixed_rows)

    fixed_out = os.path.join(
        OUT_DIR,
        f"fixed_threshold_metrics_{WEATHER_FEATURE}_{LABEL_MODE}_"
        f"M{NODE_HIERARCHY_LEVEL}_{N_FEATURES}_features_2021.csv",
    )
    fixed_df.to_csv(fixed_out, index=False)

    print("\nFixed-threshold metrics:")
    print(fixed_df)

    # =====================
    # EVENT-LEVEL EVALUATION
    # =====================

    event_dfs = []

    for thr in fixed_thresholds:
        tmp = event_region_metrics(
            y_true=y,
            y_prob=y_prob,
            event_id=event_id,
            threshold=thr,
        )
        tmp["target"] = WEATHER_FEATURE
        tmp["representation"] = REPRESENTATION
        tmp["label_mode"] = LABEL_MODE
        tmp["n_features"] = N_FEATURES
        event_dfs.append(tmp)

    event_df = pd.concat(event_dfs, ignore_index=True)
    event_df = event_df.merge(matched_df, on="event_id", how="left")

    event_out = os.path.join(
        OUT_DIR,
        f"event_metrics_{WEATHER_FEATURE}_{LABEL_MODE}_"
        f"M{NODE_HIERARCHY_LEVEL}_{N_FEATURES}_features_2021.csv",
    )
    event_df.to_csv(event_out, index=False)

    event_summary = event_df.groupby("threshold")[[
        "event_found",
        "coverage_recall",
        "precision",
        "iou",
        "area_ratio",
        "mean_prob_inside",
        "mean_prob_outside",
        "max_prob_inside",
        "max_prob_outside",
    ]].mean()

    summary_out = os.path.join(
        OUT_DIR,
        f"event_summary_{WEATHER_FEATURE}_{LABEL_MODE}_"
        f"M{NODE_HIERARCHY_LEVEL}_{N_FEATURES}_features_2021.csv",
    )
    event_summary.to_csv(summary_out)

    print("\nEvent-level summary:")
    print(event_summary)

    print("\nSaved:")
    print(threshold_out)
    print(fixed_out)
    print(event_out)
    print(summary_out)


if __name__ == "__main__":
    main()