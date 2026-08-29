import os
from glob import glob

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from malins_helper_scripts.activation_preprocessing import load_activations

from malins_helper_scripts.climatenet_preprocessing import (
    load_mask_at_nodes,
    nearest_graphcast_row,
    parse_mask_timestamp,
)


def event_region_metrics(y_true, y_prob, event_id, threshold=0.5):
    """Compute event-level spatial detection metrics at one threshold."""
    rows = []

    for eid in np.unique(event_id):
        mask = event_id == eid

        yt = y_true[mask].astype(bool)
        yp = y_prob[mask]

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


def metrics_at_best_f1_threshold(y_true, y_prob):
    """Select the threshold that maximizes F1 on the supplied data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return {
            "best_threshold": 0.5,
            "f1": f1_score(y_true, y_prob >= 0.5, zero_division=0),
            "precision": precision_score(
                y_true, y_prob >= 0.5, zero_division=0
            ),
            "recall": recall_score(
                y_true, y_prob >= 0.5, zero_division=0
            ),
        }

    precision_for_thresholds = precision[:-1]
    recall_for_thresholds = recall[:-1]

    denominator = precision_for_thresholds + recall_for_thresholds

    f1_scores = np.divide(
        2 * precision_for_thresholds * recall_for_thresholds,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )

    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    y_pred = y_prob >= best_threshold

    return {
        "best_threshold": best_threshold,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def match_climatenet_events(
    graphcast_df,
    mask_dir,
    lat,
    lon,
    all_nodes,
    *,
    label_mode="intersection",
    max_time_difference_hours=3,
    include_activation_file=False,
):
    """
    Match ClimateNet masks to the nearest GraphCast timestep.

    Returns
    -------
    matched_df : pandas.DataFrame
        One row per matched ClimateNet event.
    y : np.ndarray
        Node-level labels concatenated event by event.
    event_id : np.ndarray
        Event id for every node-level label.
    """
    mask_files = sorted(glob(os.path.join(mask_dir, "*.nc")))

    y_parts = []
    event_parts = []
    matched_rows = []

    samples_per_t = len(all_nodes)

    for i, mask_path in enumerate(mask_files):
        mask_time = parse_mask_timestamp(mask_path)

        row = nearest_graphcast_row(
            mask_time,
            graphcast_df,
            max_hours=max_time_difference_hours,
        )

        if row is None:
            continue

        graphcast_time = row["time"]

        y_nodes = load_mask_at_nodes(
            mask_path,
            lat,
            lon,
            all_nodes,
            label_mode=label_mode,
        )

        if label_mode != "soft":
            y_nodes = (y_nodes > 0).astype(np.int8)

        event_idx = len(matched_rows)

        y_parts.append(y_nodes)
        event_parts.append(
            np.full(samples_per_t, event_idx, dtype=np.int32)
        )

        matched_row = {
            "mask_file": os.path.basename(mask_path),
            "mask_time": mask_time,
            "graphcast_time": graphcast_time,
            "year": int(row["year"]),
            "t_idx": int(row["t_idx"]),
            "time_difference_hours": abs(
                graphcast_time - mask_time
            ).total_seconds() / 3600,
            "positive_nodes": int(np.sum(y_nodes > 0)),
            "positive_fraction": float(np.mean(y_nodes > 0)),
        }

        if include_activation_file:
            matched_row["activation_file"] = row["activation_file"]

        matched_rows.append(matched_row)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(mask_files)} mask files")

    if not y_parts:
        raise ValueError("No ClimateNet mask files matched GraphCast timestamps.")

    matched_df = pd.DataFrame(matched_rows)
    y = np.concatenate(y_parts).astype(np.int8)
    event_id = np.concatenate(event_parts)

    return matched_df, y, event_id


def build_split_masks(
    matched_df,
    samples_per_t,
    *,
    train_start,
    train_end,
    val_start,
    val_end,
    test_start=None,
    test_end=None,
):
    """
    Build event-level and repeated node-level train/val/test masks.
    """
    matched_times = pd.to_datetime(matched_df["graphcast_time"].values)

    event_train_mask = (
        (matched_times >= train_start)
        & (matched_times < train_end)
    )

    event_val_mask = (
        (matched_times >= val_start)
        & (matched_times < val_end)
    )

    event_test_mask = None
    if test_start is not None and test_end is not None:
        event_test_mask = (
            (matched_times >= test_start)
            & (matched_times < test_end)
        )

    train_mask = np.repeat(event_train_mask, samples_per_t)
    val_mask = np.repeat(event_val_mask, samples_per_t)

    test_mask = None
    if event_test_mask is not None:
        test_mask = np.repeat(event_test_mask, samples_per_t)

    return {
        "event_train": event_train_mask,
        "event_val": event_val_mask,
        "event_test": event_test_mask,
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    }


def build_pca_X_for_split(
    matched_df,
    split_mask_events,
    pc_scores_by_year,
    all_nodes,
    selected_pcs,
):
    """
    Build a node-level feature matrix for arbitrary PCA components.

    selected_pcs may be, for example:
        np.arange(k)       -> first k PCs
        pc_ranking[:k]     -> ranked/selected k PCs
    """
    selected_pcs = np.asarray(selected_pcs, dtype=int)
    X_parts = []

    for _, row in matched_df.loc[split_mask_events].iterrows():
        year = int(row["year"])
        t_idx = int(row["t_idx"])

        X_t = pc_scores_by_year[year][
            t_idx,
            all_nodes,
            :,
        ]

        X_t = X_t[:, selected_pcs]

        X_parts.append(
            np.asarray(X_t, dtype=np.float32)
        )

    if not X_parts:
        return np.empty(
            (0, len(selected_pcs)),
            dtype=np.float32,
        )

    return np.concatenate(X_parts, axis=0)


def build_raw_X_for_split(
    matched_df,
    split_mask_events,
    all_nodes,
    selected_features,
):
    """
    Build a node-level matrix from raw activations.

    selected_features can be any integer feature indices. For the full raw
    representation use np.arange(n_features).
    """
    selected_features = np.asarray(selected_features, dtype=int)
    X_parts = []

    selected_rows = matched_df.loc[split_mask_events]

    for i, (_, row) in enumerate(selected_rows.iterrows(), start=1):
        activation_path = row["activation_file"]
        X_t = load_activations(activation_path)

        if selected_features.size:
            max_requested = int(selected_features.max())
            if max_requested >= X_t.shape[1]:
                raise ValueError(
                    f"{activation_path} has only {X_t.shape[1]} features, "
                    f"but feature index {max_requested} was requested."
                )

        X_t = X_t[
            all_nodes,
            :,
        ][:, selected_features]

        X_parts.append(
            np.asarray(X_t, dtype=np.float32)
        )

        if i % 100 == 0:
            print(
                f"Loaded {i}/{len(selected_rows)} raw activation timesteps"
            )

    if not X_parts:
        return np.empty(
            (0, len(selected_features)),
            dtype=np.float32,
        )

    return np.concatenate(X_parts, axis=0)


def filter_finite_rows(X, y, event_id=None):
    """Remove rows containing non-finite features or labels."""
    valid = (
        np.all(np.isfinite(X), axis=1)
        & np.isfinite(y)
    )

    X = X[valid]
    y = y[valid]

    if event_id is None:
        return X, y, valid

    return X, y, event_id[valid], valid


def fit_logistic_probe(X_train, y_train):
    """Fit the common standardized L2-balanced logistic probe."""
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training set contains only one class.")

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
    return model


def evaluate_validation(model, X_val, y_val):
    """
    Evaluate validation data and select the F1-maximizing threshold.
    """
    y_val_prob = model.predict_proba(X_val)[:, 1]

    metrics = {
        "val_average_precision": average_precision_score(
            y_val,
            y_val_prob,
        ),
        "val_positive_rate": float(np.mean(y_val)),
        "val_n_positive": int(np.sum(y_val)),
        "val_n_total": int(len(y_val)),
    }

    if len(np.unique(y_val)) == 2:
        metrics["val_roc_auc"] = roc_auc_score(
            y_val,
            y_val_prob,
        )
    else:
        metrics["val_roc_auc"] = np.nan

    best = metrics_at_best_f1_threshold(
        y_true=y_val,
        y_prob=y_val_prob,
    )

    metrics.update({
        "val_best_threshold": best["best_threshold"],
        "val_f1": best["f1"],
        "val_precision": best["precision"],
        "val_recall": best["recall"],
    })

    return metrics, y_val_prob


def evaluate_test(model, X_test, y_test, threshold):
    """
    Evaluate test data at a threshold selected outside the test set.
    """
    y_test_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_average_precision": average_precision_score(
            y_test,
            y_test_prob,
        ),
        "test_positive_rate": float(np.mean(y_test)),
        "test_n_positive": int(np.sum(y_test)),
        "test_n_total": int(len(y_test)),
    }

    if len(np.unique(y_test)) == 2:
        metrics["test_roc_auc"] = roc_auc_score(
            y_test,
            y_test_prob,
        )
    else:
        metrics["test_roc_auc"] = np.nan

    y_test_pred = y_test_prob >= threshold

    metrics.update({
        "test_threshold": float(threshold),
        "test_f1": f1_score(
            y_test,
            y_test_pred,
            zero_division=0,
        ),
        "test_precision": precision_score(
            y_test,
            y_test_pred,
            zero_division=0,
        ),
        "test_recall": recall_score(
            y_test,
            y_test_pred,
            zero_division=0,
        ),
    })

    return metrics, y_test_prob


def extract_probe_direction(model):
    """Extract coefficients in standardized feature coordinates."""
    scaler = model.named_steps["standardscaler"]
    clf = model.named_steps["logisticregression"]

    coef_z = clf.coef_[0].astype(np.float32)
    norm = np.linalg.norm(coef_z)

    if norm > 0:
        coef_z_unit = coef_z / norm
    else:
        coef_z_unit = coef_z.copy()

    return {
        "coef_z": coef_z,
        "coef_z_unit": coef_z_unit,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "intercept": clf.intercept_.astype(np.float32),
    }


def save_probe(
    model,
    *,
    direction_path,
    model_path,
    metadata=None,
):
    """Save the fitted model and its standardized probe direction."""
    direction = extract_probe_direction(model)

    save_dict = dict(direction)

    if metadata:
        for key, value in metadata.items():
            save_dict[key] = np.asarray(
                value if isinstance(value, (list, tuple, np.ndarray)) else [value]
            )

    np.savez(direction_path, **save_dict)
    joblib.dump(model, model_path, compress=3)


def evaluate_event_regions(
    y_test,
    y_test_prob,
    event_id_test,
    matched_df,
    thresholds,
    *,
    metadata=None,
):
    """
    Compute event-level metrics across fixed reporting thresholds.
    """
    event_dfs = []

    for threshold in thresholds:
        tmp = event_region_metrics(
            y_true=y_test,
            y_prob=y_test_prob,
            event_id=event_id_test,
            threshold=threshold,
        )

        if metadata:
            for key, value in metadata.items():
                tmp[key] = value

        event_dfs.append(tmp)

    event_df = pd.concat(event_dfs, ignore_index=True)

    event_meta = (
        matched_df
        .reset_index()
        .rename(columns={"index": "event_id"})
    )

    return event_df.merge(
        event_meta,
        on="event_id",
        how="left",
    )
