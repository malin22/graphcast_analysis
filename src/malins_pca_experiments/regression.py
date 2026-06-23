import os
import re
from glob import glob

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error

from graphcast import icosahedral_mesh


# =====================
# CONFIG
# =====================

YEARS = [2020, 2021]

PC_SCORES_PATHS = [
    f"/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/pc_scores_{y}_per_timestep.npy"
    for y in YEARS
]

TIMESTEP_FILES_TXTS = [
    f"/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/pc_scores_{y}_per_timestep_files.txt"
    for y in YEARS
]

ACTS_DIRS = [
    f"/share/prj-4d/graphcast_shared/data/graphcast_activation_{y}"
    for y in YEARS
]

ERA5_MESH_BASE_DIR = "/share/prj-4d/graphcast_shared/data/era5_daily_mesh"

node_hierarchy_level = 5

PC_COUNTS = [5, 10, 25, 50, 100, 200, 400]
MAX_PCA = 400

regression_type = "ridge"      # "ridge" or "lasso"
score_values = "PCA"           # "PCA" or "raw_activations"

OUT_DIR = (
    f"plots/malins_experiments/2020_2021_regression/"
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
    all_timestamps = []
    all_pc_scores = []

    for pc_path, txt_path in zip(PC_SCORES_PATHS, TIMESTEP_FILES_TXTS):
        print("Loading PCA:", pc_path)

        _, ts = load_timestamps(txt_path)
        pcs = np.load(pc_path, mmap_mode="r")

        all_timestamps.append(ts)
        all_pc_scores.append(
            np.asarray(pcs[:, all_nodes, :], dtype=np.float32)
        )

    timestamps = pd.DatetimeIndex(np.concatenate(all_timestamps))
    pc_scores = np.concatenate(all_pc_scores, axis=0)

    T, n_nodes, K = pc_scores.shape

    if n_nodes != samples_per_t:
        raise ValueError(f"Expected {samples_per_t} nodes, got {n_nodes}")

    if regression_type == "ridge":
        max_features = min(max(PC_COUNTS), K)
        feature_counts = [n for n in PC_COUNTS if n <= max_features]

    elif regression_type == "lasso":
        max_features = min(MAX_PCA, K)
        feature_counts = [max_features]

    else:
        raise ValueError(f"Unknown regression_type: {regression_type}")

    X = pc_scores[:, :, :max_features]
    X = X.reshape(T * samples_per_t, max_features)

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

    if score_values == "PCA":
        X, timestamps, feature_counts = load_pca_features(
            all_nodes,
            samples_per_t,
        )

    elif score_values == "raw_activations":
        X, timestamps, feature_counts = load_raw_activation_features(
            all_nodes,
            samples_per_t,
        )

    else:
        raise ValueError(f"Wrong score_values: {score_values}")

    time_index = np.repeat(timestamps.values, samples_per_t)
    time_index = pd.to_datetime(time_index)

    train_mask = (
        (time_index >= "2020-01-01") &
        (time_index < "2021-01-01")
    )

    test_mask = (
        (time_index >= "2021-01-01") &
        (time_index < "2022-01-01")
    )

    print("X shape:", X.shape)
    print("Train samples:", train_mask.sum())
    print("Test samples:", test_mask.sum())

    print(
        f"Running {regression_type} regression with {score_values} "
        f"to predict {len(TARGETS)} targets"
    )

    y_by_target = {}

    for target in TARGETS:
        print(f"Loading meshed ERA5 target: {target['name']}")
        y_by_target[target["name"]] = load_mesh_target(
            target,
            timestamps=timestamps,
            node_indices=all_nodes,
        )

    valid_X = np.all(np.isfinite(X), axis=1)

    results = []

    for target in TARGETS:
        print(f"\nTarget: {target['name']}")

        y = y_by_target[target["name"]]
        valid = np.isfinite(y) & valid_X

        valid_train = valid & train_mask
        valid_test = valid & test_mask

        for n_features in feature_counts:
            X_train = X[valid_train, :n_features]
            X_test = X[valid_test, :n_features]

            y_train = y[valid_train]
            y_test = y[valid_test]

            if regression_type == "ridge":
                regs = [
                    ("ridge", Ridge(alpha=1.0), 1.0)
                ]

            elif regression_type == "lasso":
                regs = [
                    (f"lasso_alpha_{alpha}", Lasso(alpha=alpha, max_iter=3000), alpha)
                    for alpha in [0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 1.0]
                ]

            else:
                raise ValueError(f"Unknown regression_type: {regression_type}")

            for reg_name, reg, alpha in regs:
                model = make_pipeline(
                    StandardScaler(),
                    reg,
                )

                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)

                fitted_reg = model.named_steps[
                    reg.__class__.__name__.lower()
                ]

                n_selected = np.sum(fitted_reg.coef_ != 0)

                r2_test = r2_score(y_test, y_test_pred)
                rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
                corr_test = corr(y_test, y_test_pred)

                results.append({
                    "target": target["name"],
                    "n_features": n_features,
                    "alpha": alpha,
                    "r2_test": r2_test,
                    "rmse_test": rmse_test,
                    "corr_test": corr_test,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "n_selected": int(n_selected),
                })

                print(
                    f"{target['name']:>6s} | features={n_features:>3d} | "
                    f"alpha={alpha:.4g} | "
                    f"test R2={r2_test:.3f} | "
                    f"test r={corr_test:.3f} | "
                    f"test RMSE={rmse_test:.3f} | "
                    f"selected={n_selected:>3d}"
                )

    df = pd.DataFrame(results)

    out_csv = os.path.join(
        OUT_DIR,
        "pc_regression_physical_variables_2020train_2021test.csv",
    )

    df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()