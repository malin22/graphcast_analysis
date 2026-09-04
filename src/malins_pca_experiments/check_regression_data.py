import os

import numpy as np
import pandas as pd

from malins_helper_scripts.activation_preprocessing import (
    load_timestamps,
)

from malins_helper_scripts.mesh_context import (
    get_coarse_mesh_node_indices,
)

from malins_pca_experiments.config import (
    PC_SCORES_PATHS,
    TIMESTEP_FILES_TXTS,
    ERA5_MESH_BASE_DIR,
    NODE_HIERARCHY_LEVEL,
    LEVEL_TO_LEV,
    TARGETS,
)


def mesh_target_filename(target):
    if target["level"] is None:
        return f"{target['var']}.npy"

    lev = LEVEL_TO_LEV[target["level"]]
    return f"{target['var']}_{lev}.npy"


def main():

    target = TARGETS[0]

    all_nodes = get_coarse_mesh_node_indices(
        fine_splits=6,
        coarse_splits=NODE_HIERARCHY_LEVEL,
    )

    print("NODE_HIERARCHY_LEVEL:", NODE_HIERARCHY_LEVEL)
    print("Number of selected nodes:", len(all_nodes))
    print("Target:", target)

    # Just inspect the first training year.
    pc_path = PC_SCORES_PATHS[0]
    txt_path = TIMESTEP_FILES_TXTS[0]

    _, timestamps = load_timestamps(txt_path)
    timestamps = pd.DatetimeIndex(timestamps)

    print()
    print("PCA timestamps:")
    print("  count:", len(timestamps))
    print("  first:", timestamps[0])
    print("  last :", timestamps[-1])

    pcs = np.load(
        pc_path,
        mmap_mode="r",
    )

    print()
    print("PCA array:")
    print("  shape:", pcs.shape)
    print("  dtype:", pcs.dtype)

    file_year = timestamps[0].year

    era5_mesh_dir = os.path.join(
        ERA5_MESH_BASE_DIR,
        str(file_year),
        "mesh_l6",
    )

    time_path = os.path.join(
        era5_mesh_dir,
        "time_values.npy",
    )

    mesh_times = pd.to_datetime(
        np.load(
            time_path,
            allow_pickle=True,
        )
    )

    print()
    print("ERA5 timestamps:")
    print("  count:", len(mesh_times))
    print("  first:", mesh_times[0])
    print("  last :", mesh_times[-1])

    target_path = os.path.join(
        era5_mesh_dir,
        "time_series",
        mesh_target_filename(target),
    )

    y = np.load(
        target_path,
        mmap_mode="r",
    )

    print()
    print("Target array:")
    print("  path :", target_path)
    print("  shape:", y.shape)
    print("  dtype:", y.dtype)

    time_to_idx = {
        pd.Timestamp(t): i
        for i, t in enumerate(mesh_times)
    }

    valid_mask = np.array([
        pd.Timestamp(t) in time_to_idx
        for t in timestamps
    ])

    print()
    print("Timestamp alignment:")
    print("  matched:", valid_mask.sum())
    print("  missing:", (~valid_mask).sum())

    if not np.all(valid_mask):
        print(
            "  first missing:",
            list(timestamps[~valid_mask][:10]),
        )

   # ---------------------------------------------------------
    # Representative PCA orthogonality check
    # across BOTH training years.
    # ---------------------------------------------------------

    rng = np.random.default_rng(42)

    n_timesteps_per_year = 64
    n_nodes_per_timestep = 2000
    n_features = 100

    X_samples = []

    for pc_path, txt_path in zip(
        PC_SCORES_PATHS[:2],
        TIMESTEP_FILES_TXTS[:2],
    ):
        _, year_timestamps = load_timestamps(txt_path)
        year_timestamps = pd.DatetimeIndex(year_timestamps)

        year_pcs = np.load(
            pc_path,
            mmap_mode="r",
        )

        T = len(year_timestamps)

        sampled_times = np.linspace(
            0,
            T - 1,
            n_timesteps_per_year,
            dtype=int,
        )

        if NODE_HIERARCHY_LEVEL == 6:
            available_nodes = np.arange(
                year_pcs.shape[1]
            )
        else:
            available_nodes = np.asarray(all_nodes)

        sampled_nodes = rng.choice(
            available_nodes,
            size=min(
                n_nodes_per_timestep,
                len(available_nodes),
            ),
            replace=False,
        )

        print()
        print(
            f"Sampling year {year_timestamps[0].year}:"
        )
        print(
            "  timesteps:",
            len(sampled_times),
        )
        print(
            "  nodes/timestep:",
            len(sampled_nodes),
        )

        # Loading timestep-by-timestep avoids NumPy's
        # large advanced-indexing temporary arrays.
        for t_idx in sampled_times:

            if NODE_HIERARCHY_LEVEL == 6:
                X_t = np.asarray(
                    year_pcs[
                        t_idx,
                        sampled_nodes,
                        :n_features,
                    ],
                    dtype=np.float32,
                )
            else:
                X_t = np.asarray(
                    year_pcs[
                        t_idx,
                        sampled_nodes,
                        :n_features,
                    ],
                    dtype=np.float32,
                )

            finite = np.all(
                np.isfinite(X_t),
                axis=1,
            )

            X_samples.append(
                X_t[finite]
            )


    X = np.concatenate(
        X_samples,
        axis=0,
    )

    print()
    print("Representative PCA sample:")
    print("  shape:", X.shape)
    print(
        "  memory MB:",
        X.nbytes / 1024**2,
    )

    X64 = X.astype(
        np.float64,
    )

    mean_x = X64.mean(
        axis=0,
    )

    X64 -= mean_x

    gram = X64.T @ X64

    std = np.sqrt(
        np.diag(gram)
    )

    corr = (
        gram
        / std[:, None]
        / std[None, :]
    )

    np.fill_diagonal(
        corr,
        np.nan,
    )

    abs_corr = np.abs(
        corr[np.isfinite(corr)]
    )

    print()
    print(
        f"PCA correlation diagnostics "
        f"(first {n_features} PCs):"
    )

    print(
        "  median abs corr:",
        np.median(abs_corr),
    )

    print(
        "  90th percentile:",
        np.quantile(
            abs_corr,
            0.90,
        ),
    )

    print(
        "  99th percentile:",
        np.quantile(
            abs_corr,
            0.99,
        ),
    )

    print(
        "  max abs corr:",
        np.max(abs_corr),
    )


if __name__ == "__main__":
    main()