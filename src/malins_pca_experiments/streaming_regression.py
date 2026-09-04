import os

import numpy as np
import pandas as pd

from malins_helper_scripts.activation_preprocessing import (
    load_timestamps,
)

from malins_pca_experiments.config import (
    PC_SCORES_PATHS,
    TIMESTEP_FILES_TXTS,
    ERA5_MESH_BASE_DIR,
    NODE_HIERARCHY_LEVEL,
    LEVEL_TO_LEV,
    TRAIN_YEARS,
    TEST_YEARS,
)



def mesh_target_filename(target):
    if target["level"] is None:
        return f"{target['var']}.npy"
    lev = LEVEL_TO_LEV[target["level"]]
    return f"{target['var']}_{lev}.npy"



def iter_pca_target_chunks(
    target,
    all_nodes,
    n_features,
    years,
    chunk_timesteps=1,
):
    for pc_path, txt_path in zip(
        PC_SCORES_PATHS,
        TIMESTEP_FILES_TXTS,
    ):
        _, timestamps = load_timestamps(txt_path)
        timestamps = pd.DatetimeIndex(timestamps)

        file_year = timestamps[0].year

        if file_year not in years:
            continue

        print(f"Streaming year {file_year}")

        # PCA scores stay memory-mapped
        pcs = np.load(
            pc_path,
            mmap_mode="r",
        )

        era5_mesh_dir = os.path.join(
            ERA5_MESH_BASE_DIR,
            str(file_year),
            "mesh_l6",
        )

        era5_mesh_ts_dir = os.path.join(
            era5_mesh_dir,
            "time_series",
        )

        era5_mesh_time_values = os.path.join(
            era5_mesh_dir,
            "time_values.npy",
        )

        if not os.path.exists(era5_mesh_time_values):
            raise FileNotFoundError(
                f"Missing ERA5 time values: {era5_mesh_time_values}"
            )

        mesh_times = pd.to_datetime(
            np.load(
                era5_mesh_time_values,
                allow_pickle=True,
            )
        )

        target_path = os.path.join(
            era5_mesh_ts_dir,
            mesh_target_filename(target),
        )

        if not os.path.exists(target_path):
            raise FileNotFoundError(
                f"Missing ERA5 target file: {target_path}"
            )

        y_memmap = np.load(
            target_path,
            mmap_mode="r",
        )

        time_to_idx = {
            pd.Timestamp(t): i
            for i, t in enumerate(mesh_times)
        }

        valid_time_mask = np.array([
            pd.Timestamp(t) in time_to_idx
            for t in timestamps
        ])

        if not np.all(valid_time_mask):
            missing = timestamps[~valid_time_mask]

            print(
                f"Warning: {len(missing)} PCA timestamps are missing "
                f"from ERA5 for {file_year}"
            )

            print(
                "First missing timestamps:",
                list(missing[:10]),
            )

        timestamps = timestamps[valid_time_mask]

        pca_time_indices = np.nonzero(valid_time_mask)[0]

        target_indices = np.array([
            time_to_idx[pd.Timestamp(t)]
            for t in timestamps
        ])

        T = len(timestamps)
        for start in range(0, T, chunk_timesteps):
            stop = min(start + chunk_timesteps, T)

            # PCA indices corresponding to the valid timestamps
            pca_idx = pca_time_indices[start:stop]

            if NODE_HIERARCHY_LEVEL == 6:
                X_chunk = np.asarray(
                    pcs[
                        pca_idx,
                        :,
                        :n_features,
                    ],
                    dtype=np.float32,
                )

                y_chunk = np.asarray(
                    y_memmap[
                        target_indices[start:stop],
                        :
                    ],
                    dtype=np.float32,
                )

            else:
                X_chunk = np.asarray(
                    pcs[
                        pca_idx,
                        all_nodes,
                        :n_features,
                    ],
                    dtype=np.float32,
                )

                y_chunk = np.asarray(
                    y_memmap[
                        target_indices[start:stop]
                    ][:, all_nodes],
                    dtype=np.float32,
                )
            X_chunk = X_chunk.reshape(
                -1,
                n_features,
            )

            y_chunk = y_chunk.reshape(-1)

            valid = (
                np.all(
                    np.isfinite(X_chunk),
                    axis=1,
                )
                &
                np.isfinite(y_chunk)
            )

            yield (
                X_chunk[valid],
                y_chunk[valid],
            )
def accumulate_training_statistics(
    target,
    all_nodes,
    max_features,
):
    XtX = np.zeros(
        (max_features, max_features),
        dtype=np.float64,
    )

    Xty = np.zeros(
        max_features,
        dtype=np.float64,
    )

    sum_x = np.zeros(
        max_features,
        dtype=np.float64,
    )

    sum_x2 = np.zeros(
        max_features,
        dtype=np.float64,
    )

    sum_y = 0.0
    n = 0

    for X_chunk, y_chunk in iter_pca_target_chunks(
        target=target,
        all_nodes=all_nodes,
        n_features=max_features,
        years=TRAIN_YEARS,
        chunk_timesteps=1,
    ):
        X64 = X_chunk.astype(
            np.float64,
            copy=False,
        )

        y64 = y_chunk.astype(
            np.float64,
            copy=False,
        )

        XtX += X64.T @ X64
        Xty += X64.T @ y64

        sum_x += X64.sum(axis=0)
        sum_x2 += np.sum(
            X64 * X64,
            axis=0,
        )

        sum_y += y64.sum()
        n += len(y64)

    if n == 0:
        raise RuntimeError(
            f"No valid training samples for {target['name']}"
        )

    return {
        "XtX": XtX,
        "Xty": Xty,
        "sum_x": sum_x,
        "sum_x2": sum_x2,
        "sum_y": sum_y,
        "n": n,
    }


def fit_regression_from_statistics(
    stats,
    n_features,
    regression_type,
    alpha=1.0,
):
    n = stats["n"]

    # Only use the first k PCs.
    XtX = stats["XtX"][:n_features, :n_features]
    Xty = stats["Xty"][:n_features]

    sum_x = stats["sum_x"][:n_features]
    sum_x2 = stats["sum_x2"][:n_features]

    sum_y = stats["sum_y"]

    mean_x = sum_x / n
    mean_y = sum_y / n

    XtX_centered = (
        XtX
        - n * np.outer(
            mean_x,
            mean_x,
        )
    )

    Xty_centered = (
        Xty
        - n * mean_x * mean_y
    )

    if regression_type == "linear":
        coef = np.linalg.solve(
            XtX_centered,
            Xty_centered,
        )

    elif regression_type == "ridge":
        var_x = (
            sum_x2 / n
            - mean_x ** 2
        )

        scale_x = np.sqrt(
            np.maximum(
                var_x,
                0.0,
            )
        )

        scale_x[scale_x == 0] = 1.0

        XtX_scaled = (
            XtX_centered
            / scale_x[:, None]
            / scale_x[None, :]
        )

        Xty_scaled = (
            Xty_centered
            / scale_x
        )

        A = (
            XtX_scaled
            + alpha * np.eye(
                n_features,
                dtype=np.float64,
            )
        )

        coef_scaled = np.linalg.solve(
            A,
            Xty_scaled,
        )

        coef = (
            coef_scaled
            / scale_x
        )

    else:
        raise ValueError(
            f"Streaming regression supports only "
            f"'linear' and 'ridge', got {regression_type}"
        )

    intercept = (
        mean_y
        - mean_x @ coef
    )

    return coef, intercept

def evaluate_streaming_regressions(
    target,
    all_nodes,
    models,
):
    """
    Evaluate multiple PCA regression models in a single pass
    through the test data.

    models:
        dict mapping n_features -> {
            "coef": coef,
            "intercept": intercept,
        }

    Returns:
        dict mapping n_features -> metrics
    """

    max_features = max(models)

    stats = {}

    for n_features in models:
        stats[n_features] = {
            "n": 0,
            "sum_y": 0.0,
            "sum_pred": 0.0,
            "sum_y2": 0.0,
            "sum_pred2": 0.0,
            "sum_ypred": 0.0,
            "sse": 0.0,
        }

    for X_chunk, y_chunk in iter_pca_target_chunks(
        target=target,
        all_nodes=all_nodes,
        n_features=max_features,
        years=TEST_YEARS,
        chunk_timesteps=1,
    ):
        # Convert once.
        X_chunk = X_chunk.astype(
            np.float64,
            copy=False,
        )

        y_chunk = y_chunk.astype(
            np.float64,
            copy=False,
        )

        for n_features, model in models.items():

            coef = model["coef"]
            intercept = model["intercept"]

            pred = (
                X_chunk[:, :n_features] @ coef
                + intercept
            )

            diff = y_chunk - pred

            s = stats[n_features]

            s["sse"] += np.sum(
                diff * diff
            )

            s["sum_y"] += np.sum(
                y_chunk
            )

            s["sum_pred"] += np.sum(
                pred
            )

            s["sum_y2"] += np.sum(
                y_chunk * y_chunk
            )

            s["sum_pred2"] += np.sum(
                pred * pred
            )

            s["sum_ypred"] += np.sum(
                y_chunk * pred
            )

            s["n"] += len(y_chunk)

    results = {}

    for n_features, s in stats.items():

        n = s["n"]

        if n == 0:
            raise RuntimeError(
                f"No valid test samples for "
                f"{target['name']}"
            )

        sst = (
            s["sum_y2"]
            - s["sum_y"] ** 2 / n
        )

        r2 = (
            1.0
            - s["sse"] / sst
        )

        rmse = np.sqrt(
            s["sse"] / n
        )

        corr_num = (
            s["sum_ypred"]
            - s["sum_y"]
            * s["sum_pred"]
            / n
        )

        corr_den = np.sqrt(
            (
                s["sum_y2"]
                - s["sum_y"] ** 2 / n
            )
            *
            (
                s["sum_pred2"]
                - s["sum_pred"] ** 2 / n
            )
        )

        corr_test = (
            corr_num / corr_den
        )

        results[n_features] = {
            "r2_test": r2,
            "rmse_test": rmse,
            "corr_test": corr_test,
            "n_test": n,
        }

    return results