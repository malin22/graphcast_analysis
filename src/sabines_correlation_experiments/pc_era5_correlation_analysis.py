#!/usr/bin/env python3
import gc
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from graphcast import icosahedral_mesh

# ============================================================
# CONFIGURATION
# ============================================================
ACTIVATIONS_DIR = Path("/share/prj-4d/graphcast_shared/data/graphcast_activation_2021")

# Precomputed coarse ERA5 dataset produced by the coarse-grid export script
# Expected layout:
#   /share/prj-4d/graphcast_shared/data/era5_daily_course_grid/2021/
#       time_values.npy
#       metadata.json
#       time_series/*.npy
#       static/*.npy
COARSE_ERA5_ROOT = Path("/share/prj-4d/graphcast_shared/data/era5_daily_course_grid/2021")
COARSE_TIME_VALUES = COARSE_ERA5_ROOT / "time_values.npy"
COARSE_TIME_SERIES_DIR = COARSE_ERA5_ROOT / "time_series"
COARSE_STATIC_DIR = COARSE_ERA5_ROOT / "static"

PCA_COMPONENTS_PATH = "/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021.npy"
PCA_MEAN_PATH = "/share/prj-4d/graphcast_shared/data/pca_components/pca_mean_2021.npy"

RESULTS_DIR = Path("plots/sabines_experiments")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
YEARLY_SUMMARY_PATH = RESULTS_DIR / "pc_era5_yearly_top_variables.json"

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
# MAIN
# ============================================================
def main():
    print("Loading PCA components and mean...")
    pca_components = np.load(PCA_COMPONENTS_PATH)
    pca_mean = np.load(PCA_MEAN_PATH)

    print(f" pca_components.shape: {pca_components.shape}")
    print(f" pca_mean.shape: {pca_mean.shape}")

    if pca_components.ndim != 2:
        raise ValueError(f"Expected pca_components to be 2D, got {pca_components.shape}")

    n_pcs_available = pca_components.shape[0]
    n_pcs = min(N_PCS_TO_CHECK, n_pcs_available)
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

    activation_files = sorted(ACTIVATIONS_DIR.glob("*.npy"))
    centers = [p.stem.split("_t")[-1] for p in activation_files]
    print(f"Found {len(activation_files)} activation files")

    # Stats per PC per ERA5 feature
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

        # Project activations onto PCs and bin to the same coarse grid
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

        # Batched correlations: PC rows vs ERA5 rows
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

            # Update yearly stats
            for j, feature_name in enumerate(feature_names):
                r = row[j]
                if not np.isfinite(r):
                    continue
                s = pc_stats[pc_idx][feature_name]
                s["sum_abs_r"] += abs(float(r))
                s["count"] += 1

            pc_stats[pc_idx][best_name]["wins"] += 1

        processed_timesteps += 1

        # Free per-timestep memory
        del activations
        del era5_grids
        del pc_grids
        del corr_matrix
        gc.collect()

    # ========================================================
    # YEARLY SUMMARY
    # ========================================================
    summary = {}
    for pc_idx in range(n_pcs):
        rows = []
        for feature_name, stats in pc_stats[pc_idx].items():
            if stats["count"] == 0:
                continue
            rows.append({
                "variable": feature_name,
                "mean_abs_r": stats["sum_abs_r"] / stats["count"],
                "wins": stats["wins"],
                "count": stats["count"],
            })

        rows_sorted = sorted(rows, key=lambda x: (-x["mean_abs_r"], -x["wins"]))
        summary[f"PC_{pc_idx + 1}"] = {
            "top_variables": rows_sorted[:TOP_K],
            "total_timesteps": processed_timesteps,
        }

    with open(YEARLY_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Yearly PC -> ERA5 Summary ===\n")
    for pc_key, info in sorted(summary.items(), key=lambda kv: int(kv[0].split("_")[1])):
        print(f"{pc_key} (timesteps: {info['total_timesteps']})")
        if not info["top_variables"]:
            print("  no valid correlations")
            continue
        for i, item in enumerate(info["top_variables"], 1):
            print(
                f"  {i}. {item['variable']}: "
                f"mean_|r|={item['mean_abs_r']:.3f}, "
                f"wins={item['wins']}, "
                f"count={item['count']}"
            )
        print()

    print(f"Saved summary to {YEARLY_SUMMARY_PATH}")


if __name__ == "__main__":
    main()