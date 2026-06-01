#!/usr/bin/env python3
import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
import re

import numpy as np
from graphcast import icosahedral_mesh

from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================
ACTIVATIONS_DIR = Path("/share/prj-4d/graphcast_shared/data/graphcast_activation_2021")

COARSE_ERA5_ROOT = Path("/share/prj-4d/graphcast_shared/data/era5_daily_course_grid/2021")
COARSE_TIME_VALUES = COARSE_ERA5_ROOT / "time_values.npy"
COARSE_TIME_SERIES_DIR = COARSE_ERA5_ROOT / "time_series"
COARSE_STATIC_DIR = COARSE_ERA5_ROOT / "static"

PCA_COMPONENTS_PATH = "/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021.npy"
PCA_MEAN_PATH = "/share/prj-4d/graphcast_shared/data/pca_components/pca_mean_2021.npy"

RESULTS_DIR = Path("plots/sabines_experiments")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

YEARLY_SUMMARY_PATH = RESULTS_DIR / "pc_era5_yearly_top_variables.json"
SCREENING_CACHE_PATH = RESULTS_DIR / "pc_era5_screening_cache.json"
MODEL_RESULTS_PATH = RESULTS_DIR / "pc_era5_model_results.json"

# User controls
N_PCS_TO_CHECK = 20
TOP_K = 5
LAT_RES = 2.5
LON_RES = 2.5
MIN_POINTS = 10
VERBOSE_TIMESTEPS = True

lat_bins = np.arange(-90, 90.0 + LAT_RES, LAT_RES)
lon_bins = np.arange(0, 360.0 + LON_RES, LON_RES)


# ============================================================
# NUMERIC HELPERS
# ============================================================
def to_float32(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == np.dtype("|V2"):
        arr = arr.view(np.float16)
    return np.asarray(arr, dtype=np.float32)


def load_activation_matrix(path: Path) -> np.ndarray:
    """
    Load saved GraphCast activations and coerce to a numeric float32 matrix.
    Expected output shape: [n_nodes, n_features]
    """
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x, dtype=np.float32)

    # Common saved shapes from your earlier scripts
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"Expected activation matrix [nodes, features], got shape {x.shape}")

    return x


# ============================================================
# MESH / GRID MAPPING
# ============================================================
def get_mesh_latlon(splits: int = 6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    vertices = meshes[6].vertices
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    return lat.astype(np.float32), lon.astype(np.float32)


def build_node_cell_index(lat: np.ndarray, lon: np.ndarray, lat_edges: np.ndarray, lon_edges: np.ndarray):
    """
    Build a node -> coarse-cell mapping for the mesh nodes.
    """
    lon_mod = np.mod(lon, 360.0)

    lat_idx = np.digitize(lat, lat_edges) - 1
    lon_idx = np.digitize(lon_mod, lon_edges) - 1

    n_lat = len(lat_edges) - 1
    n_lon = len(lon_edges) - 1

    valid = (
        (lat_idx >= 0) & (lat_idx < n_lat) &
        (lon_idx >= 0) & (lon_idx < n_lon)
    )

    cell_idx = lat_idx * n_lon + lon_idx
    return cell_idx, valid, n_lat, n_lon


def coarsen_1d_field(values: np.ndarray, cell_idx: np.ndarray, valid: np.ndarray, n_lat: int, n_lon: int) -> np.ndarray:
    """
    Vectorized binning of node values onto the coarse grid using np.add.at.
    """
    values = to_float32(values).ravel()
    values = values[valid]
    idx = cell_idx[valid]

    finite = np.isfinite(values)
    values = values[finite]
    idx = idx[finite]

    sums = np.zeros(n_lat * n_lon, dtype=np.float64)
    counts = np.zeros(n_lat * n_lon, dtype=np.int32)

    np.add.at(sums, idx, values)
    np.add.at(counts, idx, 1)

    out = np.full(n_lat * n_lon, np.nan, dtype=np.float32)
    mask = counts > 0
    out[mask] = (sums[mask] / counts[mask]).astype(np.float32)
    return out.reshape(n_lat, n_lon)


def project_activations_to_pc_grids(
    activations: np.ndarray,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
    n_pcs: int,
    cell_idx: np.ndarray,
    valid: np.ndarray,
    n_lat: int,
    n_lon: int,
) -> tuple[list[str], np.ndarray]:
    """
    Project activation vectors onto PCA components, then coarse-bin each PC map.
    Returns:
      pc_names: list of PC labels
      pc_grids: array with shape [n_pcs, n_lat, n_lon]
    """
    if activations.shape[1] != pca_mean.shape[0]:
        raise ValueError(
            f"Activation feature dimension {activations.shape[1]} does not match pca_mean {pca_mean.shape[0]}"
        )

    centered = activations - pca_mean
    pcs = pca_components[:n_pcs]
    pc_scores = centered @ pcs.T  # [n_nodes, n_pcs]

    pc_grids = np.empty((n_pcs, n_lat, n_lon), dtype=np.float32)
    for pc_idx in range(n_pcs):
        pc_grids[pc_idx] = coarsen_1d_field(pc_scores[:, pc_idx], cell_idx, valid, n_lat, n_lon)

    pc_names = [f"PC_{i + 1}" for i in range(n_pcs)]
    return pc_names, pc_grids


# ============================================================
# COARSE ERA5 LOADING
# ============================================================
def load_coarse_catalog(root: Path):
    """
    Load the precomputed coarse ERA5 dataset as memmaps.
    Time-series fields live in time_series/*.npy.
    Static fields live in static/*.npy.
    """
    if not COARSE_TIME_VALUES.exists():
        raise FileNotFoundError(f"Missing coarse time index file: {COARSE_TIME_VALUES}")

    time_values = np.load(COARSE_TIME_VALUES, allow_pickle=False)
    time_index = {
        np.datetime_as_string(np.datetime64(t), unit="h"): i
        for i, t in enumerate(time_values)
    }

    time_series = {}
    for path in sorted(COARSE_TIME_SERIES_DIR.glob("*.npy")):
        time_series[path.stem] = np.load(path, mmap_mode="r")

    static_fields = {}
    for path in sorted(COARSE_STATIC_DIR.glob("*.npy")):
        static_fields[path.stem] = np.load(path, mmap_mode="r")

    if not time_series and not static_fields:
        raise FileNotFoundError(f"No coarse ERA5 files found under {root}")

    return time_index, time_series, static_fields


def load_coarse_feature_grids(
    time_index: dict,
    time_series: dict,
    static_fields: dict,
    center_str: str,
) -> tuple[list[str], np.ndarray]:
    """
    Load one timestep's coarse ERA5 fields into:
      feature_names: list[str]
      grids: [n_features, n_lat, n_lon]
    """
    if center_str not in time_index:
        return [], np.empty((0, 0, 0), dtype=np.float32)

    t_idx = time_index[center_str]
    feature_names = []
    grids = []

    for name in sorted(time_series.keys()):
        arr = time_series[name]
        grid = to_float32(arr[t_idx])
        grids.append(grid)
        feature_names.append(name)

    for name in sorted(static_fields.keys()):
        arr = static_fields[name]
        grid = to_float32(arr)
        grids.append(grid)
        feature_names.append(name)

    if not grids:
        return [], np.empty((0, 0, 0), dtype=np.float32)

    grids = np.stack(grids, axis=0)
    return feature_names, grids


# ============================================================
# MATRIX CORRELATION
# ============================================================
def batch_correlation_matrix(P: np.ndarray, E: np.ndarray, min_points: int = 10):
    """
    Compute correlations between rows of P and rows of E using one matrix multiply.

    P: [n_pcs, n_cells]
    E: [n_features, n_cells]

    Returns:
      corr_matrix: [n_pcs, n_features]
      n_used: number of spatial cells used
    """
    if P.ndim != 2 or E.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got P={P.shape}, E={E.shape}")

    mask = np.isfinite(P).all(axis=0) & np.isfinite(E).all(axis=0)
    n_used = int(mask.sum())
    if n_used < min_points:
        return None, n_used

    P = P[:, mask].astype(np.float64, copy=False)
    E = E[:, mask].astype(np.float64, copy=False)

    P0 = P - P.mean(axis=1, keepdims=True)
    E0 = E - E.mean(axis=1, keepdims=True)

    Pstd = P0.std(axis=1, ddof=0, keepdims=True)
    Estd = E0.std(axis=1, ddof=0, keepdims=True)

    valid_p = (Pstd[:, 0] > 0)
    valid_e = (Estd[:, 0] > 0)

    corr = np.full((P.shape[0], E.shape[0]), np.nan, dtype=np.float32)

    if valid_p.any() and valid_e.any():
        Pz = P0[valid_p] / Pstd[valid_p]
        Ez = E0[valid_e] / Estd[valid_e]
        corr[np.ix_(valid_p, valid_e)] = (Pz @ Ez.T) / n_used

    return corr, n_used


# ============================================================
# SPARSE REGRESSION FITTING
# ============================================================

def _atomic_write_json(path: Path, payload: dict):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w") as f:
        json.dump(payload, f, indent=2)
    temp_path.replace(path)


def _load_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _rank_feature_coefficients(feature_names, coef_values):
    rows = []
    for name, coef in zip(feature_names, coef_values):
        coef = float(coef)
        rows.append(
            {
                "feature": name,
                "coefficient": coef,
                "abs_coefficient": abs(coef),
            }
        )
    return sorted(rows, key=lambda row: (-row["abs_coefficient"], row["feature"]))


def _feature_field_name(feature_name: str) -> str:
    match = re.match(r"^(.*)_lev(\d+)$", feature_name)
    if match:
        return match.group(1)
    return feature_name


def _feature_level_index(feature_name: str):
    match = re.match(r"^.*_lev(\d+)$", feature_name)
    if match:
        return int(match.group(1))
    return None


def _level_band_from_index(level_idx: int, level_values: list[int]) -> str:
    if level_idx is None:
        return "static"

    if not level_values:
        return "all_levels"

    sorted_levels = sorted(level_values)
    n = len(sorted_levels)
    low_cut = sorted_levels[max(0, n // 3 - 1)]
    high_cut = sorted_levels[min(n - 1, (2 * n) // 3)]

    if level_idx <= low_cut:
        return "lower"
    if level_idx <= high_cut:
        return "mid"
    return "upper"


def _summarize_grouped_importance(feature_names, coef_values):
    field_scores = defaultdict(float)
    band_scores = defaultdict(float)

    level_values = []
    for name in feature_names:
        level_idx = _feature_level_index(name)
        if level_idx is not None:
            level_values.append(level_idx)

    for name, coef in zip(feature_names, coef_values):
        weight = abs(float(coef))
        field_scores[_feature_field_name(name)] += weight

        level_idx = _feature_level_index(name)
        band_scores[_level_band_from_index(level_idx, level_values)] += weight

    field_summary = [
        {"field": field, "abs_weight_sum": score}
        for field, score in sorted(field_scores.items(), key=lambda kv: -kv[1])
    ]
    band_summary = [
        {"band": band, "abs_weight_sum": score}
        for band, score in sorted(band_scores.items(), key=lambda kv: -kv[1])
    ]
    return field_summary, band_summary

def _rescale_linear_model(coef_std, intercept_std, x_scaler, y_scaler):
    x_scale = np.asarray(x_scaler.scale_, dtype=np.float64)
    x_mean = np.asarray(x_scaler.mean_, dtype=np.float64)
    y_scale = float(np.asarray(y_scaler.scale_).ravel()[0])
    y_mean = float(np.asarray(y_scaler.mean_).ravel()[0])

    coef_std = np.asarray(coef_std, dtype=np.float64)
    coef_raw = coef_std * (y_scale / x_scale)
    intercept_raw = y_mean + y_scale * float(intercept_std) - float(np.dot(coef_raw, x_mean))
    return coef_raw.astype(np.float32), float(intercept_raw)


def _fit_linear_model_with_alpha_grid(
    X_train_z: np.ndarray,
    y_train_z: np.ndarray,
    X_val_z: np.ndarray,
    y_val_z: np.ndarray,
    model_type: str,
    alpha_grid=None,
    l1_ratio: float = 0.7,
):
    if alpha_grid is None:
        alpha_grid = np.logspace(-4, -1, 12)

    model_type = model_type.lower()
    best_model = None
    best_score = -np.inf
    best_alpha = None

    for alpha in alpha_grid:
        if model_type == "ridge":
            model = Ridge(alpha=float(alpha), fit_intercept=True)
        elif model_type == "elasticnet":
            model = ElasticNet(
                alpha=float(alpha),
                l1_ratio=float(l1_ratio),
                fit_intercept=True,
                max_iter=20000,
                tol=1e-4,
                random_state=0,
            )
        else:
            raise ValueError("model_type must be Ridge or ElasticNet")

        model.fit(X_train_z, y_train_z)
        score = model.score(X_val_z, y_val_z)

        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_model = model

    return best_model, best_score, best_alpha


def run_single_variable_analysis(
    n_pcs_to_check: int = N_PCS_TO_CHECK,
    activation_files=None,
    cache_path: Path = SCREENING_CACHE_PATH,
    force_recompute: bool = False,
    max_timesteps: int | None = None,
):
    """
    Compute the full single-variable PC-to-ERA5 ranking once and cache it.
    The cache keeps the full ranked list per PC, not just the top display rows.
    """
    if cache_path.exists() and not force_recompute:
        with open(cache_path, "r") as f:
            return json.load(f)

    print("Loading PCA components and mean...")
    pca_components = np.load(PCA_COMPONENTS_PATH)
    pca_mean = np.load(PCA_MEAN_PATH)

    print(f" pca_components.shape: {pca_components.shape}")
    print(f" pca_mean.shape: {pca_mean.shape}")

    if pca_components.ndim != 2:
        raise ValueError(f"Expected pca_components to be 2D, got {pca_components.shape}")

    n_pcs_available = pca_components.shape[0]
    n_pcs = min(n_pcs_to_check, n_pcs_available)
    print(f"\nAnalyzing {n_pcs} PCs (out of {n_pcs_available} available)\n")

    print("Loading coarse ERA5 catalog...")
    time_index, time_series, static_fields = load_coarse_catalog(COARSE_ERA5_ROOT)
    print(f" Loaded {len(time_series)} time-series coarse files and {len(static_fields)} static coarse files")
    print(f" Loaded {len(time_index)} coarse timesteps")

    mesh_lat, mesh_lon = get_mesh_latlon(splits=6)
    n_mesh_nodes = mesh_lat.size
    print(f"Mesh nodes: {n_mesh_nodes}")

    cell_idx, valid, n_lat, n_lon = build_node_cell_index(mesh_lat, mesh_lon, lat_bins, lon_bins)
    print(f"Coarse grid shape: {n_lat} x {n_lon}")

    if activation_files is None:
        activation_files = sorted(ACTIVATIONS_DIR.glob("*.npy"))

    centers = [p.stem.split("_t")[-1] for p in activation_files]
    print(f"Found {len(activation_files)} activation files")

    pc_stats = defaultdict(lambda: defaultdict(lambda: {"sum_abs_r": 0.0, "count": 0, "wins": 0}))
    processed_timesteps = 0

    for act_file, center_str in zip(activation_files, centers):
        print(f"\nProcessing {center_str}")

        if center_str not in time_index:
            print("  Skipping: no matching coarse ERA5 timestep")
            continue

        activations = load_activation_matrix(act_file)
        print(f"  activations.shape: {activations.shape}")

        if activations.shape[0] != n_mesh_nodes:
            print(f"  Warning: activation nodes {activations.shape[0]} != mesh nodes {n_mesh_nodes}")

        feature_names, era5_grids = load_coarse_feature_grids(
            time_index=time_index,
            time_series=time_series,
            static_fields=static_fields,
            center_str=center_str,
        )

        if era5_grids.size == 0:
            print("  Skipping: no coarse ERA5 features loaded")
            del activations
            gc.collect()
            continue

        print(f"  Loaded coarse ERA5 features: {len(feature_names)}")

        pc_names, pc_grids = project_activations_to_pc_grids(
            activations=activations,
            pca_mean=pca_mean,
            pca_components=pca_components,
            n_pcs=n_pcs,
            cell_idx=cell_idx,
            valid=valid,
            n_lat=n_lat,
            n_lon=n_lon,
        )

        P = pc_grids.reshape(n_pcs, -1)
        E = era5_grids.reshape(era5_grids.shape[0], -1)

        corr_matrix, n_used = batch_correlation_matrix(P, E, min_points=MIN_POINTS)
        if corr_matrix is None:
            print(f"  Skipping: not enough common valid cells ({n_used})")
            del activations, era5_grids, pc_grids
            gc.collect()
            continue

        if VERBOSE_TIMESTEPS:
            print(f"  Correlation matrix computed using {n_used} common grid cells")

        for pc_idx, pc_name in enumerate(pc_names):
            row = corr_matrix[pc_idx]
            valid_row = np.isfinite(row)
            if not valid_row.any():
                print(f"  {pc_name}: no valid correlations")
                continue

            best_j = int(np.nanargmax(np.abs(row)))
            best_name = feature_names[best_j]
            best_r = float(row[best_j])

            if VERBOSE_TIMESTEPS:
                print(f"  For timestep {center_str} and {pc_name} - top variable = {best_name} with r={best_r:.3f}")

            for j, feature_name in enumerate(feature_names):
                r = row[j]
                if not np.isfinite(r):
                    continue
                s = pc_stats[pc_idx][feature_name]
                s["sum_abs_r"] += abs(float(r))
                s["count"] += 1

            pc_stats[pc_idx][best_name]["wins"] += 1

        processed_timesteps += 1

        del activations
        del era5_grids
        del pc_grids
        del corr_matrix
        gc.collect()

    cache = {}
    display_summary = {}
    for pc_idx in range(n_pcs):
        rows = []
        for feature_name, stats in pc_stats[pc_idx].items():
            if stats["count"] == 0:
                continue
            rows.append(
                {
                    "variable": feature_name,
                    "mean_abs_r": stats["sum_abs_r"] / stats["count"],
                    "wins": stats["wins"],
                    "count": stats["count"],
                }
            )

        rows_sorted = sorted(rows, key=lambda x: (-x["mean_abs_r"], -x["wins"]))
        pc_key = f"PC_{pc_idx + 1}"
        cache[pc_key] = {
            "ranked_variables": rows_sorted,
            "top_variables": rows_sorted[:TOP_K],
            "total_timesteps": processed_timesteps,
        }
        display_summary[pc_key] = {
            "top_variables": rows_sorted[:TOP_K],
            "total_timesteps": processed_timesteps,
        }

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    with open(YEARLY_SUMMARY_PATH, "w") as f:
        json.dump(display_summary, f, indent=2)

    print(f"Saved screening cache to {cache_path}")
    print(f"Saved yearly summary to {YEARLY_SUMMARY_PATH}")

    return cache


def fit_pc_era5_sparse_model(
    pc_idx: int,
    screening_rows: list[dict],
    activation_files=None,
    top_screen_k: int = 30,
    train_fraction: float = 0.8,
    model_type: str = "Ridge",
    l1_ratio: float = 0.7,
    alpha_grid=None,
    max_timesteps: int | None = None,
):
    """
    Fit either Ridge or Elastic Net for one PC using precomputed screening rows.
    Stage 1 is skipped if screening_rows are supplied.
    """
    if activation_files is None:
        activation_files = sorted(ACTIVATIONS_DIR.glob("*.npy"))

    if alpha_grid is None:
        alpha_grid = np.logspace(-4, -1, 12) if model_type.lower() == "elasticnet" else np.logspace(-3, 3, 25)

    print("Loading PCA components and mean...")
    pca_components = np.load(PCA_COMPONENTS_PATH)
    pca_mean = np.load(PCA_MEAN_PATH)

    time_index, time_series, static_fields = load_coarse_catalog(COARSE_ERA5_ROOT)
    mesh_lat, mesh_lon = get_mesh_latlon(splits=6)
    cell_idx, valid, n_lat, n_lon = build_node_cell_index(mesh_lat, mesh_lon, lat_bins, lon_bins)

    centers = [p.stem.split("_t")[-1] for p in activation_files]
    usable_steps = []
    for act_file, center_str in zip(activation_files, centers):
        if center_str in time_index:
            usable_steps.append((act_file, center_str))

    if max_timesteps is not None:
        usable_steps = usable_steps[:max_timesteps]

    if not usable_steps:
        raise ValueError("No activation files matched the coarse ERA5 time index")

    n_train_steps = max(1, int(len(usable_steps) * train_fraction))
    train_steps = usable_steps[:n_train_steps]
    val_steps = usable_steps[n_train_steps:] if n_train_steps < len(usable_steps) else usable_steps[-1:]

    if pca_components.ndim != 2:
        raise ValueError(f"Expected pca_components to be 2D, got {pca_components.shape}")

    n_pcs_available = pca_components.shape[0]
    if pc_idx < 0 or pc_idx >= n_pcs_available:
        raise ValueError(f"pc_idx={pc_idx} out of range for {n_pcs_available} PCs")

    selected_features = [row["variable"] for row in screening_rows[:top_screen_k]]

    if not selected_features:
        raise ValueError("No screened features found for sparse fit")

    print(f"Selected features for {model_type} fit: {selected_features}")

    feature_to_index = None
    X_train_blocks = []
    y_train_blocks = []
    X_val_blocks = []
    y_val_blocks = []

    for split_idx, (act_file, center_str) in enumerate(train_steps + val_steps):
        activations = load_activation_matrix(act_file)
        feature_names, era5_grids = load_coarse_feature_grids(
            time_index=time_index,
            time_series=time_series,
            static_fields=static_fields,
            center_str=center_str,
        )

        if era5_grids.size == 0:
            continue

        if feature_to_index is None:
            feature_to_index = {name: i for i, name in enumerate(feature_names)}

        missing = [name for name in selected_features if name not in feature_to_index]
        if missing:
            raise KeyError(f"Selected features not found in ERA5 grid files: {missing}")

        pc_names, pc_grids = project_activations_to_pc_grids(
            activations=activations,
            pca_mean=pca_mean,
            pca_components=pca_components,
            n_pcs=pc_idx + 1,
            cell_idx=cell_idx,
            valid=valid,
            n_lat=n_lat,
            n_lon=n_lon,
        )

        y = pc_grids[pc_idx].reshape(-1).astype(np.float32)

        feature_idx = [feature_to_index[name] for name in selected_features]
        X = era5_grids[feature_idx].reshape(len(selected_features), -1).T.astype(np.float32)

        mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if mask.sum() < MIN_POINTS:
            continue

        X = X[mask]
        y = y[mask]

        if split_idx < len(train_steps):
            X_train_blocks.append(X)
            y_train_blocks.append(y)
        else:
            X_val_blocks.append(X)
            y_val_blocks.append(y)

        del activations, era5_grids, pc_grids, X, y
        gc.collect()

    if not X_train_blocks or not X_val_blocks:
        raise ValueError("Not enough train/validation data to fit the model")

    X_train = np.concatenate(X_train_blocks, axis=0)
    y_train = np.concatenate(y_train_blocks, axis=0)
    X_val = np.concatenate(X_val_blocks, axis=0)
    y_val = np.concatenate(y_val_blocks, axis=0)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_z = x_scaler.fit_transform(X_train)
    y_train_z = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    X_val_z = x_scaler.transform(X_val)
    y_val_z = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    best_model, best_score, best_alpha = _fit_linear_model_with_alpha_grid(
        X_train_z=X_train_z,
        y_train_z=y_train_z,
        X_val_z=X_val_z,
        y_val_z=y_val_z,
        model_type=model_type,
        alpha_grid=alpha_grid,
        l1_ratio=l1_ratio,
    )

    coef_std = np.asarray(best_model.coef_, dtype=np.float32)
    coef_raw, intercept_raw = _rescale_linear_model(
        coef_std=coef_std,
        intercept_std=best_model.intercept_,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )

    ranked_features_standardized = _rank_feature_coefficients(selected_features, coef_std)
    ranked_features_raw = _rank_feature_coefficients(selected_features, coef_raw)

    field_summary_standardized, band_summary_standardized = _summarize_grouped_importance(
        selected_features,
        coef_std,
    )
    field_summary_raw, band_summary_raw = _summarize_grouped_importance(
        selected_features,
        coef_raw,
    )

    result = {
        "pc_name": f"PC_{pc_idx + 1}",
        "selected_features": selected_features,
        "model_type": model_type,
        "alpha": best_alpha,
        "l1_ratio": float(l1_ratio) if model_type.lower() == "elasticnet" else None,
        "val_r2": float(best_score),
        "ranked_features_standardized": ranked_features_standardized,
        "ranked_features_raw": ranked_features_raw,
        "coef_standardized": {name: float(coef) for name, coef in zip(selected_features, coef_std)},
        "coef_raw": {name: float(coef) for name, coef in zip(selected_features, coef_raw)},
        "intercept_raw": float(intercept_raw),
        "field_importance": field_summary_standardized,
        "field_importance_raw": field_summary_raw,
        "vertical_band_importance": band_summary_standardized,
        "vertical_band_importance_raw": band_summary_raw,
        "n_train_samples": int(X_train.shape[0]),
        "n_val_samples": int(X_val.shape[0]),
    }

    return result

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="PC to ERA5 screening and regression")
    parser.add_argument(
        "--model-type",
        choices=["Ridge", "ElasticNet"],
        default="ElasticNet",
        help="Regression model to fit after screening",
    )
    parser.add_argument(
        "--top-screen-k",
        type=int,
        default=30,
        help="How many screened variables to use in the regression stage",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=0.7,
        help="Elastic Net l1_ratio, ignored for Ridge",
    )
    parser.add_argument(
        "--force-recompute-screening",
        action="store_true",
        help="Recompute and overwrite the screening cache",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Optional cap on timesteps to process",
    )
    parser.add_argument(
        "--pc-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of zero-based PC indices to fit. If omitted, fit all PCs in the screening cache.",
    )
    args = parser.parse_args()

    screening_cache = run_single_variable_analysis(
        n_pcs_to_check=N_PCS_TO_CHECK,
        cache_path=SCREENING_CACHE_PATH,
        force_recompute=args.force_recompute_screening,
        max_timesteps=args.max_timesteps,
    )

    if args.pc_indices is None or len(args.pc_indices) == 0:
        pc_indices = sorted(
            [int(key.split("_")[1]) - 1 for key in screening_cache.keys()],
            key=int,
        )
    else:
        pc_indices = args.pc_indices

    all_results = _load_json_dict(MODEL_RESULTS_PATH)

    for pc_idx in pc_indices:
        pc_key = f"PC_{pc_idx + 1}"
        if pc_key not in screening_cache:
            print(f"Skipping {pc_key}: not present in screening cache")
            continue

        if pc_key in all_results:
            print(f"Skipping {pc_key}: already present in {MODEL_RESULTS_PATH}")
            continue

        result = fit_pc_era5_sparse_model(
            pc_idx=pc_idx,
            screening_rows=screening_cache[pc_key]["ranked_variables"],
            top_screen_k=args.top_screen_k,
            train_fraction=0.8,
            model_type=args.model_type,
            l1_ratio=args.l1_ratio,
            max_timesteps=args.max_timesteps,
        )

        all_results[pc_key] = result
        _atomic_write_json(MODEL_RESULTS_PATH, all_results)

        print(f"\n{pc_key} results")
        print("top standardized features:")
        for row in result["ranked_features_standardized"][:10]:
            print(row)
        print("field_importance (standardized):")
        print(result["field_importance"])
        print("vertical_band_importance (standardized):")
        print(result["vertical_band_importance"])

    _atomic_write_json(MODEL_RESULTS_PATH, all_results)
    print(f"\nSaved model results to {MODEL_RESULTS_PATH}")

if __name__ == "__main__":
    main()