import os

import numpy as np
import pandas as pd

from malins_helper_scripts.activation_preprocessing import (
    build_graphcast_time_table,
    load_pca_metadata,
    load_raw_activation_years,
)
from malins_regression.logistic_probe_pipeline import (
    build_pca_X_for_split,
    build_raw_X_for_split,
    build_split_masks,
    evaluate_event_regions,
    evaluate_test,
    evaluate_validation,
    filter_finite_rows,
    fit_logistic_probe,
    match_climatenet_events,
    save_probe,
)
from malins_helper_scripts.mesh_context import (
    get_coarse_mesh_node_indices,
    get_mesh_latlon,
)


def run_logistic_experiment(
    *,
    experiment_name,
    weather_feature,
    feature_source,
    feature_counts,
    selected_features_fn,
    fine_mesh_level,
    node_hierarchy_level,
    label_mode,
    max_time_difference_hours,
    thresholds,
    train_start,
    train_end,
    val_start,
    val_end,
    test_start,
    test_end,
    mask_dir,
    out_dir,
    acts_dirs=None,
    pc_scores_paths=None,
    timestep_files_txts=None,
    extra_metadata=None,
):
    """
    Run one complete ClimateNet logistic-probe experiment.

    Parameters
    ----------
    experiment_name : str
        Name used in outputs, e.g. "raw_activations", "first_k_pcs",
        or "selected_pcs".

    weather_feature : str
        ClimateNet target, e.g. "AR" or "TC".

    feature_source : {"raw", "pca"}
        Which GraphCast representation to load.

    feature_counts : iterable of int
        Numbers of features/components to test.

    selected_features_fn : callable
        Function taking k and returning integer feature indices.

        Examples
        --------
        Raw / first-k:
            lambda k: np.arange(k)

        Ranked PCs:
            lambda k: pc_ranking[:k]

    Remaining arguments define mesh level, split dates, paths, and output
    settings.
    """
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("LOGISTIC PROBE EXPERIMENT")
    print("=" * 80)
    print("Experiment:", experiment_name)
    print("Target:", weather_feature)
    print("Feature source:", feature_source)
    print("Feature counts:", list(feature_counts))
    print()

    # ========================================================
    # 1. Mesh
    # ========================================================

    lat, lon = get_mesh_latlon(
        splits=fine_mesh_level,
    )

    # ClimateNet longitude coordinates use [0, 360).
    lon = lon % 360

    all_nodes = get_coarse_mesh_node_indices(
        fine_splits=fine_mesh_level,
        coarse_splits=node_hierarchy_level,
    )

    samples_per_t = len(all_nodes)

    print("Nodes per timestep:", samples_per_t)
    print(
        f"Using M{node_hierarchy_level} nodes "
        f"from M{fine_mesh_level} GraphCast mesh"
    )

    # ========================================================
    # 2. Load GraphCast representation + timestamps
    # ========================================================

    pc_scores_by_year = None

    if feature_source == "pca":
        if pc_scores_paths is None or timestep_files_txts is None:
            raise ValueError(
                "pc_scores_paths and timestep_files_txts are required "
                "when feature_source='pca'."
            )

        (
            pc_scores_by_year,
            timestamps_by_year,
            max_features,
        ) = load_pca_metadata(
            pc_scores_paths,
            timestep_files_txts,
        )

        graphcast_df = build_graphcast_time_table(
            timestamps_by_year
        )

    elif feature_source == "raw":
        if acts_dirs is None:
            raise ValueError(
                "acts_dirs is required when feature_source='raw'."
            )

        (
            activation_files,
            graphcast_times,
            max_features,
        ) = load_raw_activation_years(
            acts_dirs
        )

        graphcast_df = pd.DataFrame({
            "year": graphcast_times.year.astype(int),
            "t_idx": np.arange(
                len(activation_files),
                dtype=int,
            ),
            "time": graphcast_times,
            "activation_file": activation_files,
        })

        graphcast_df = (
            graphcast_df
            .sort_values("time")
            .reset_index(drop=True)
        )

    else:
        raise ValueError(
            f"Unknown feature_source: {feature_source!r}. "
            "Expected 'raw' or 'pca'."
        )

    # ========================================================
    # 3. ClimateNet matching
    # ========================================================

    matched_df, y, event_id = match_climatenet_events(
        graphcast_df,
        mask_dir,
        lat,
        lon,
        all_nodes,
        label_mode=label_mode,
        max_time_difference_hours=max_time_difference_hours,
        include_activation_file=(feature_source == "raw"),
    )

    matched_path = os.path.join(
        out_dir,
        "matched_files.csv",
    )

    matched_df.to_csv(
        matched_path,
        index=False,
    )

    # ========================================================
    # 4. Train / validation / test split
    # ========================================================

    split_masks = build_split_masks(
        matched_df,
        samples_per_t,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
    )

    y_train_all = y[
        split_masks["train"]
    ]

    y_val_all = y[
        split_masks["val"]
    ]

    y_test_all = y[
        split_masks["test"]
    ]

    print()
    print(
        f"Matched {weather_feature} files:",
        len(matched_df),
    )
    print("y shape:", y.shape)
    print(
        "Overall positive rate:",
        float(np.mean(y)),
    )

    print(
        f"Train window: "
        f"{train_start.date()} to {train_end.date()} exclusive"
    )
    print(
        f"Validation window: "
        f"{val_start.date()} to {val_end.date()} exclusive"
    )
    print(
        f"Test window: "
        f"{test_start.date()} to {test_end.date()} exclusive"
    )

    print("Train samples:", int(split_masks["train"].sum()))
    print("Validation samples:", int(split_masks["val"].sum()))
    print("Test samples:", int(split_masks["test"].sum()))

    print("Train positives:", int(y_train_all.sum()))
    print("Validation positives:", int(y_val_all.sum()))
    print("Test positives:", int(y_test_all.sum()))

    # ========================================================
    # 5. Run requested feature subsets
    # ========================================================

    results = []

    for n_features in feature_counts:
        selected_features = np.asarray(
            selected_features_fn(n_features),
            dtype=int,
        )

        if len(selected_features) != n_features:
            raise ValueError(
                f"selected_features_fn({n_features}) returned "
                f"{len(selected_features)} indices."
            )

        if selected_features.size == 0:
            print(
                f"Skipping {n_features}: no features selected."
            )
            continue

        if selected_features.min() < 0:
            raise ValueError(
                "Feature indices must be non-negative."
            )

        if selected_features.max() >= max_features:
            print(
                f"Skipping {n_features}: selected feature index "
                f"{selected_features.max()} exceeds available "
                f"range 0..{max_features - 1}"
            )
            continue

        print()
        print("=" * 80)
        print(
            f"{experiment_name} | "
            f"{n_features} features"
        )
        print("=" * 80)

        # ----------------------------------------------------
        # 5a. Build feature matrices
        # ----------------------------------------------------

        if feature_source == "pca":
            X_train = build_pca_X_for_split(
                matched_df,
                split_masks["event_train"],
                pc_scores_by_year,
                all_nodes,
                selected_features,
            )

            X_val = build_pca_X_for_split(
                matched_df,
                split_masks["event_val"],
                pc_scores_by_year,
                all_nodes,
                selected_features,
            )

            X_test = build_pca_X_for_split(
                matched_df,
                split_masks["event_test"],
                pc_scores_by_year,
                all_nodes,
                selected_features,
            )

        else:
            X_train = build_raw_X_for_split(
                matched_df,
                split_masks["event_train"],
                all_nodes,
                selected_features,
            )

            X_val = build_raw_X_for_split(
                matched_df,
                split_masks["event_val"],
                all_nodes,
                selected_features,
            )

            X_test = build_raw_X_for_split(
                matched_df,
                split_masks["event_test"],
                all_nodes,
                selected_features,
            )

        y_train = y_train_all
        y_val = y_val_all
        y_test = y_test_all

        # ----------------------------------------------------
        # 5b. Remove non-finite rows
        # ----------------------------------------------------

        X_train, y_train, _ = filter_finite_rows(
            X_train,
            y_train,
        )

        X_val, y_val, _ = filter_finite_rows(
            X_val,
            y_val,
        )

        event_id_test = event_id[
            split_masks["test"]
        ]

        (
            X_test,
            y_test,
            event_id_test,
            _,
        ) = filter_finite_rows(
            X_test,
            y_test,
            event_id=event_id_test,
        )

        if len(np.unique(y_train)) < 2:
            print(
                f"Skipping {n_features}: "
                "training set has only one class."
            )
            continue

        if len(np.unique(y_val)) < 2:
            print(
                f"Skipping {n_features}: "
                "validation set has only one class."
            )
            continue

        if len(np.unique(y_test)) < 2:
            print(
                f"Skipping {n_features}: "
                "test set has only one class."
            )
            continue

        # ----------------------------------------------------
        # 5c. Fit shared logistic probe
        # ----------------------------------------------------

        model = fit_logistic_probe(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # 5d. Validation: select threshold
        # ----------------------------------------------------

        val_metrics, _ = evaluate_validation(
            model,
            X_val,
            y_val,
        )

        selected_threshold = val_metrics[
            "val_best_threshold"
        ]

        # ----------------------------------------------------
        # 5e. Held-out test evaluation
        # ----------------------------------------------------

        test_metrics, y_test_prob = evaluate_test(
            model,
            X_test,
            y_test,
            threshold=selected_threshold,
        )

        # ----------------------------------------------------
        # 5f. Save probe + model
        # ----------------------------------------------------

        direction_training_window = "2019_2020_train_only"

        direction_out = os.path.join(
            out_dir,
            f"probe_direction_{weather_feature}_"
            f"{experiment_name}_{label_mode}_"
            f"M{node_hierarchy_level}_"
            f"{n_features}_features_"
            f"{direction_training_window}.npz",
        )

        model_out = os.path.join(
            out_dir,
            f"logistic_probe_model_{weather_feature}_"
            f"{experiment_name}_{label_mode}_"
            f"M{node_hierarchy_level}_"
            f"{n_features}_features_"
            f"{direction_training_window}.joblib",
        )

        metadata = {
            "n_features": n_features,
            "selected_features": selected_features,
            "feature_source": feature_source,
            "experiment_name": experiment_name,
            "weather_feature": weather_feature,
            "label_mode": label_mode,
            "fine_mesh_level": fine_mesh_level,
            "node_hierarchy_level": node_hierarchy_level,
            "direction_training_window": direction_training_window,
            "train_start": str(train_start),
            "train_end": str(train_end),
            "val_start": str(val_start),
            "val_end": str(val_end),
            "test_start": str(test_start),
            "test_end": str(test_end),
        }

        if extra_metadata:
            metadata.update(
                extra_metadata
            )

        save_probe(
            model,
            direction_path=direction_out,
            model_path=model_out,
            metadata=metadata,
        )

        print("Saved probe direction:", direction_out)
        print("Saved logistic model:", model_out)

        # ----------------------------------------------------
        # 5g. Event-level test metrics
        # ----------------------------------------------------

        event_metadata = {
            "target": weather_feature,
            "experiment": experiment_name,
            "feature_source": feature_source,
            "n_features": n_features,
            "label_mode": label_mode,
        }

        event_df = evaluate_event_regions(
            y_test=y_test,
            y_test_prob=y_test_prob,
            event_id_test=event_id_test,
            matched_df=matched_df,
            thresholds=thresholds,
            metadata=event_metadata,
        )

        event_out = os.path.join(
            out_dir,
            f"event_region_metrics_{weather_feature}_"
            f"{experiment_name}_{label_mode}_"
            f"M{node_hierarchy_level}_"
            f"{n_features}_features_"
            f"max_{max_time_difference_hours}hour.csv",
        )

        event_df.to_csv(
            event_out,
            index=False,
        )

        summary = (
            event_df
            .groupby("threshold")[
                [
                    "event_found",
                    "coverage_recall",
                    "precision",
                    "iou",
                    "area_ratio",
                ]
            ]
            .mean()
        )

        print()
        print("Event-level summary:")
        print(summary)
        print("Saved event-level metrics:", event_out)

        # ----------------------------------------------------
        # 5h. Collect summary row
        # ----------------------------------------------------

        result = {
            "target": weather_feature,
            "experiment": experiment_name,
            "feature_source": feature_source,
            "label_mode": label_mode,
            "n_features": n_features,
            "selected_features": ",".join(
                map(str, selected_features.tolist())
            ),
            "model": "logistic_l2_balanced",
            "direction_training_window": direction_training_window,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
            "train_positive_rate": float(
                np.mean(y_train)
            ),
        }

        result.update(
            val_metrics
        )

        result.update(
            test_metrics
        )

        results.append(
            result
        )

        print(
            f"{weather_feature} | "
            f"{experiment_name} | "
            f"features={n_features:>3d} | "
            f"TEST AP="
            f"{test_metrics['test_average_precision']:.3f} | "
            f"TEST AUC="
            f"{test_metrics['test_roc_auc']:.3f} | "
            f"TEST F1@VAL_THRESHOLD="
            f"{test_metrics['test_f1']:.3f} | "
            f"threshold="
            f"{test_metrics['test_threshold']:.6f} | "
            f"P={test_metrics['test_precision']:.3f} | "
            f"R={test_metrics['test_recall']:.3f}"
        )

        del X_train, X_val, X_test

    # ========================================================
    # 6. Save experiment summary
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    results_path = os.path.join(
        out_dir,
        f"logistic_probe_{weather_feature}_"
        f"{experiment_name}_{label_mode}_"
        f"M{node_hierarchy_level}_"
        f"max_{max_time_difference_hours}hour.csv",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("EXPERIMENT FINISHED")
    print("=" * 80)
    print("Saved results:", results_path)
    print("Saved matched files:", matched_path)

    return results_df
