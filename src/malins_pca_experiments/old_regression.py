import os
import re
from glob import glob

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from graphcast import icosahedral_mesh


# =====================
# CONFIG
# =====================

YEARS = [2019, 2020, 2021]

PC_SCORES_PATHS = [
    (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2019_from_2019_2020_pca_per_timestep.npy"
    ),
    (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2020_from_2019_2020_pca_per_timestep.npy"
    ),
    (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2021_from_2019_2020_pca_per_timestep.npy"
    ),
]

TIMESTEP_FILES_TXTS = [
    (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2019_from_2019_2020_pca_per_timestep_files.txt"
    ),
    (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2020_from_2019_2020_pca_per_timestep_files.txt"
    ),
    (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2021_from_2019_2020_pca_per_timestep_files.txt"
    ),
]

ACTS_DIRS = [
    f"/share/prj-4d/graphcast_shared/data/graphcast_activation_{y}"
    for y in YEARS
]

ERA5_MESH_BASE_DIR = "/share/prj-4d/graphcast_shared/data/era5_daily_mesh"

node_hierarchy_level = 5

PC_COUNTS = [5, 10, 25, 50, 100, 200, 400, 512]
MAX_PCA = 512

regression_type = "linear"   # "linear", "ridge", or "lasso"
score_values = "PCA"           # "PCA" or "raw_activations"

OUT_DIR = (
    f"plots/malins_experiments/regression_test_2021_train_2019_2020/"
    f"{score_values}/{regression_type}/l{node_hierarchy_level}_nodes"
)
os.makedirs(OUT_DIR, exist_ok=True)


TARGETS = [
    {"name": "2t", "var": "2m_temperature", "level": None},
    {"name": "10u", "var": "10m_u_component_of_wind", "level": None},
    {"name": "10v", "var": "10m_v_component_of_wind", "level": None},
    {"name": "msl", "var": "mean_sea_level_pressure", "level": None},
    {"name": "tp", "var": "total_precipitation_6hr", "level": None},

    {"name": "t50", "var": "temperature", "level": 50},
    {"name": "t250", "var": "temperature", "level": 250},
    {"name": "t500", "var": "temperature", "level": 500},
    {"name": "t600", "var": "temperature", "level": 600},
    {"name": "t700", "var": "temperature", "level": 700},
    {"name": "t850", "var": "temperature", "level": 850},
    {"name": "t1000", "var": "temperature", "level": 1000},

    {"name": "u50", "var": "u_component_of_wind", "level": 50},
    {"name": "u250", "var": "u_component_of_wind", "level": 250},
    {"name": "u500", "var": "u_component_of_wind", "level": 500},
    {"name": "u600", "var": "u_component_of_wind", "level": 600},
    {"name": "u700", "var": "u_component_of_wind", "level": 700},
    {"name": "u850", "var": "u_component_of_wind", "level": 850},
    {"name": "u1000", "var": "u_component_of_wind", "level": 1000},

    {"name": "v50", "var": "v_component_of_wind", "level": 50},
    {"name": "v250", "var": "v_component_of_wind", "level": 250},
    {"name": "v500", "var": "v_component_of_wind", "level": 500},
    {"name": "v600", "var": "v_component_of_wind", "level": 600},
    {"name": "v700", "var": "v_component_of_wind", "level": 700},
    {"name": "v850", "var": "v_component_of_wind", "level": 850},
    {"name": "v1000", "var": "v_component_of_wind", "level": 1000},

    {"name": "z50", "var": "geopotential", "level": 50},
    {"name": "z250", "var": "geopotential", "level": 250},
    {"name": "z500", "var": "geopotential", "level": 500},
    {"name": "z600", "var": "geopotential", "level": 600},
    {"name": "z700", "var": "geopotential", "level": 700},
    {"name": "z850", "var": "geopotential", "level": 850},
    {"name": "z1000", "var": "geopotential", "level": 1000},

    {"name": "q50", "var": "specific_humidity", "level": 50},
    {"name": "q250", "var": "specific_humidity", "level": 250},
    {"name": "q500", "var": "specific_humidity", "level": 500},
    {"name": "q600", "var": "specific_humidity", "level": 600},
    {"name": "q700", "var": "specific_humidity", "level": 700},
    {"name": "q850", "var": "specific_humidity", "level": 850},
    {"name": "q1000", "var": "specific_humidity", "level": 1000},

    {"name": "w50", "var": "vertical_velocity", "level": 50},
    {"name": "w250", "var": "vertical_velocity", "level": 250},
    {"name": "w500", "var": "vertical_velocity", "level": 500},
    {"name": "w600", "var": "vertical_velocity", "level": 600},
    {"name": "w700", "var": "vertical_velocity", "level": 700},
    {"name": "w850", "var": "vertical_velocity", "level": 850},
    {"name": "w1000", "var": "vertical_velocity", "level": 1000},
]


PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700,
    750, 775, 800, 825, 850, 875, 900, 925,
    950, 975, 1000,
]

LEVEL_TO_LEV = {
    level: f"lev{i:02d}"
    for i, level in enumerate(PRESSURE_LEVELS)
}


# =====================
# HELPERS
# =====================

def parse_timestamp_from_path(path):
    fname = os.path.basename(path)
    m = re.search(r"t(\d{4})-(\d{2})-(\d{2})T(\d{2})", fname)
    if not m:
        raise ValueError(f"Could not parse timestamp from {fname}")
    y, mo, d, h = map(int, m.groups())
    return pd.Timestamp(y, mo, d, h)


def load_timestamps(files_txt):
    with open(files_txt, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    timestamps = pd.to_datetime([parse_timestamp_from_path(p) for p in files])
    return files, timestamps


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


def mesh_target_filename(target):
    if target["level"] is None:
        return f"{target['var']}.npy"
    lev = LEVEL_TO_LEV[target["level"]]
    return f"{target['var']}_{lev}.npy"


def load_mesh_target(target, timestamps, node_indices):
    timestamps = pd.DatetimeIndex(timestamps)
    ys = []

    for year in sorted(timestamps.year.unique()):
        year_mask = timestamps.year == year
        year_timestamps = timestamps[year_mask]

        era5_mesh_dir = os.path.join(
            ERA5_MESH_BASE_DIR,
            str(year),
            "mesh_l6",
        )
        era5_mesh_ts_dir = os.path.join(era5_mesh_dir, "time_series")
        era5_mesh_time_values = os.path.join(era5_mesh_dir, "time_values.npy")

        mesh_times = pd.to_datetime(
            np.load(era5_mesh_time_values, allow_pickle=True)
        )

        path = os.path.join(era5_mesh_ts_dir, mesh_target_filename(target))

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing mesh target file: {path}")

        arr = np.load(path, mmap_mode="r")

        time_to_idx = {pd.Timestamp(t): i for i, t in enumerate(mesh_times)}
        idx = [time_to_idx[pd.Timestamp(t)] for t in year_timestamps]

        y_year = np.asarray(arr[idx][:, node_indices], dtype=np.float32)
        ys.append(y_year)

    return np.concatenate(ys, axis=0).reshape(-1)


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

    for v in coarse_vertices:
        key = tuple(np.round(v, decimals))
        if key not in fine_keys:
            raise ValueError("Could not match coarse vertex to fine mesh")
        coarse_indices.append(fine_keys[key])

    return np.array(coarse_indices, dtype=int)


def load_activations(path):
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    return x.astype(np.float32)


def corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return np.corrcoef(a[mask], b[mask])[0, 1]

def load_pca_features(all_nodes, samples_per_t):
    if regression_type in ["ridge", "linear"]:
        max_needed = max(PC_COUNTS)
        feature_counts = PC_COUNTS
    elif regression_type == "lasso":
        max_needed = MAX_PCA
        feature_counts = [MAX_PCA]
    else:
        raise ValueError(f"Unknown regression_type: {regression_type}")

    all_timestamps = []
    all_pc_scores = []

    for pc_path, txt_path in zip(PC_SCORES_PATHS, TIMESTEP_FILES_TXTS):
        print("Loading PCA:", pc_path)

        _, ts = load_timestamps(txt_path)
        pcs = np.load(pc_path, mmap_mode="r")

        max_features = min(max_needed, pcs.shape[2])

        all_timestamps.append(ts)
        all_pc_scores.append(
            np.asarray(pcs[:, all_nodes, :max_features], dtype=np.float32)
        )

    timestamps = pd.DatetimeIndex(np.concatenate(all_timestamps))
    pc_scores = np.concatenate(all_pc_scores, axis=0)

    T, n_nodes, K = pc_scores.shape
    feature_counts = [n for n in feature_counts if n <= K]

    X = pc_scores.reshape(T * samples_per_t, K)

    return X, timestamps, feature_counts

def load_raw_activation_features(all_nodes, samples_per_t):
    act_files = []

    for acts_dir in ACTS_DIRS:
        print("Scanning activations:", acts_dir)
        act_files.extend(sorted(glob(os.path.join(acts_dir, "*.npy"))))

    valid_files = []

    for f in act_files:
        X_t = load_activations(f)

        if np.isnan(X_t).any():
            continue

        valid_files.append(f)

    act_files = valid_files
    timestamps = pd.to_datetime([parse_timestamp_from_path(p) for p in act_files])
    T = len(act_files)

    X_parts = []

    for i, f in enumerate(act_files):
        X_t = load_activations(f)
        X_parts.append(X_t[all_nodes, :])

        if (i + 1) % 100 == 0:
            print(f"Loaded raw activations for {i + 1}/{T}")

    X = np.stack(X_parts, axis=0).astype(np.float32)
    max_features = X.shape[2]
    feature_counts = [max_features]

    X = X.reshape(T * samples_per_t, max_features)

    return X, pd.DatetimeIndex(timestamps), feature_counts

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

            if node_hierarchy_level == 6:
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
def fit_streaming_regression(
    target,
    all_nodes,
    n_features,
    regression_type,
    alpha=1.0,
):
    XtX = np.zeros(
        (n_features, n_features),
        dtype=np.float64,
    )

    Xty = np.zeros(
        n_features,
        dtype=np.float64,
    )

    sum_x = np.zeros(
        n_features,
        dtype=np.float64,
    )

    sum_x2 = np.zeros(
        n_features,
        dtype=np.float64,
    )

    sum_y = 0.0
    n = 0

    for X_chunk, y_chunk in iter_pca_target_chunks(
        target=target,
        all_nodes=all_nodes,
        n_features=n_features,
        years=[2019, 2020],
        chunk_timesteps=1,
    ):
        # Accumulate in float64
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

        del X_chunk
        del y_chunk
        del X64
        del y64

    if n == 0:
        raise RuntimeError(
            f"No valid training samples for {target['name']}"
        )

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

    return coef, intercept, n


def evaluate_streaming_regression(
    target,
    all_nodes,
    n_features,
    coef,
    intercept,
):
    n = 0

    sum_y = 0.0
    sum_pred = 0.0

    sum_y2 = 0.0
    sum_pred2 = 0.0
    sum_ypred = 0.0

    sse = 0.0

    for X_chunk, y_chunk in iter_pca_target_chunks(
        target=target,
        all_nodes=all_nodes,
        n_features=n_features,
        years=[2021],
        chunk_timesteps=1,
    ):
        pred = (
            X_chunk @ coef
            + intercept
        )

        diff = y_chunk - pred

        sse += np.sum(
            diff * diff
        )

        sum_y += np.sum(y_chunk)
        sum_pred += np.sum(pred)

        sum_y2 += np.sum(
            y_chunk * y_chunk
        )

        sum_pred2 += np.sum(
            pred * pred
        )

        sum_ypred += np.sum(
            y_chunk * pred
        )

        n += len(y_chunk)

        del X_chunk
        del y_chunk
        del pred
        del diff

    if n == 0:
        raise RuntimeError(
            f"No valid test samples for {target['name']}"
        )

    sst = (
        sum_y2
        - sum_y ** 2 / n
    )

    r2 = 1.0 - sse / sst

    rmse = np.sqrt(
        sse / n
    )

    corr_num = (
        sum_ypred
        - sum_y * sum_pred / n
    )

    corr_den = np.sqrt(
        (
            sum_y2
            - sum_y ** 2 / n
        )
        *
        (
            sum_pred2
            - sum_pred ** 2 / n
        )
    )

    corr_test = corr_num / corr_den

    return (
        r2,
        rmse,
        corr_test,
        n,
    )

# =====================
# MAIN
# =====================

def main():
    lat, lon = get_mesh_latlon(splits=6)

    coarse_nodes = get_coarse_mesh_node_indices(
        fine_splits=6,
        coarse_splits=node_hierarchy_level,
    )

    all_nodes = coarse_nodes
    samples_per_t = len(all_nodes)

    print("Nodes per timestep:", samples_per_t)
    print(f"Using M{node_hierarchy_level} coarse mesh nodes: {samples_per_t}")

    if score_values != "PCA":
        raise NotImplementedError(
            "Streaming implementation currently supports PCA only."
        )

    print(
        f"Running streaming {regression_type} regression with PCA "
        f"to predict {len(TARGETS)} targets"
    )

    results = []

    for target in TARGETS:
        print(f"\nTarget: {target['name']}")

        for n_features in PC_COUNTS:

            if regression_type == "linear":
                alpha = 0.0

            elif regression_type == "ridge":
                alpha = 1.0

            else:
                raise ValueError(
                    f"Streaming version currently supports "
                    f"'linear' and 'ridge', got {regression_type}"
                )

            print(
                f"Fitting {target['name']} "
                f"with {n_features} PCs"
            )

            coef, intercept, n_train = (
                fit_streaming_regression(
                    target=target,
                    all_nodes=all_nodes,
                    n_features=n_features,
                    regression_type=regression_type,
                    alpha=alpha,
                )
            )

            (
                r2_test,
                rmse_test,
                corr_test,
                n_test,
            ) = evaluate_streaming_regression(
                target=target,
                all_nodes=all_nodes,
                n_features=n_features,
                coef=coef,
                intercept=intercept,
            )

            results.append({
                "target": target["name"],
                "n_features": n_features,
                "alpha": alpha,
                "r2_test": r2_test,
                "rmse_test": rmse_test,
                "corr_test": corr_test,
                "n_train": n_train,
                "n_test": n_test,
                "n_selected": n_features,
            })

            print(
                f"{target['name']:>6s} | "
                f"features={n_features:>3d} | "
                f"alpha={alpha:.4g} | "
                f"test R2={r2_test:.3f} | "
                f"test r={corr_test:.3f} | "
                f"test RMSE={rmse_test:.3f}"
            )

            del coef

    df = pd.DataFrame(results)

    out_csv = os.path.join(
        OUT_DIR,
        "pc_regression_physical_variables_2019_2020train_2021test.csv",
    )

    df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()