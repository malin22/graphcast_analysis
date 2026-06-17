#!/usr/bin/env python3
import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from graphcast import icosahedral_mesh

# ============================================================
# DEFAULT CONFIGURATION
# ============================================================
DEFAULT_ACTIVATIONS_DIR = Path("/share/prj-4d/graphcast_shared/data/graphcast_activation_2021")
DEFAULT_ERA5_ROOT = Path("/share/prj-4d/graphcast_shared/data/era5_daily_mesh/2021/mesh_l6")

PCA_COMPONENTS_PATH = "/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021.npy"
PCA_MEAN_PATH = "/share/prj-4d/graphcast_shared/data/pca_components/pca_mean_2021.npy"

RESULTS_DIR = Path("plots/sabines_experiments")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_PCS_TO_CHECK = 20
TOP_K = 5
MIN_POINTS = 10
VERBOSE_TIMESTEPS = True


# ============================================================
# NUMERIC HELPERS
# ============================================================
def to_float32(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == np.dtype("|V2"):
        arr = arr.view(np.float16)
    return np.asarray(arr, dtype=np.float32)


def load_activation_matrix(path: Path) -> np.ndarray:
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"Expected activation matrix [nodes, features], got {x.shape}")

    return x


def parse_center_time(path: Path) -> str:
    return path.stem.split("_t")[-1]


# ============================================================
# MESH LEVEL SELECTION
# ============================================================
def get_graphcast_mesh_vertices(level: int, splits: int = 6) -> np.ndarray:
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    return np.asarray(meshes[level].vertices, dtype=np.float32)


def vertex_key(v: np.ndarray, decimals: int = 12):
    return tuple(np.round(v.astype(np.float64), decimals))


def selected_m6_indices_for_mesh_level(
    mesh_level: int,
    era5_m6_vertices: np.ndarray,
    splits: int = 6,
) -> np.ndarray:
    """
    Return indices into stored m6 ERA5 arrays corresponding to the requested
    GraphCast mesh level.

    mesh_level=6 returns all m6 indices.
    mesh_level=5 returns the m6 indices whose coordinates are m5 vertices.
    """
    era5_m6_vertices = np.asarray(era5_m6_vertices, dtype=np.float32)
    n_m6 = era5_m6_vertices.shape[0]

    if mesh_level == 6:
        return np.arange(n_m6, dtype=np.int64)

    if mesh_level < 0 or mesh_level > 6:
        raise ValueError("mesh_level must be between 0 and 6")

    target_vertices = get_graphcast_mesh_vertices(mesh_level, splits=splits)

    m6_lookup = {
        vertex_key(v): i
        for i, v in enumerate(era5_m6_vertices)
    }

    selected = []
    missing = []

    for j, v in enumerate(target_vertices):
        key = vertex_key(v)
        if key in m6_lookup:
            selected.append(m6_lookup[key])
        else:
            missing.append(j)

    if missing:
        raise ValueError(
            f"Could not match {len(missing)} level-{mesh_level} vertices into ERA5 m6 vertices. "
            "This suggests mesh ordering or vertex precision is not the same."
        )

    selected = np.asarray(selected, dtype=np.int64)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Duplicate m6 indices found while selecting mesh-level nodes")

    return selected


def select_activation_nodes(
    activations: np.ndarray,
    selected_m6_indices: np.ndarray,
    n_m6_nodes: int,
) -> np.ndarray:
    """
    Accept either:
      - full m6 activations [n_m6, features], then slice to selected nodes
      - already-selected activations [n_selected, features], then return as-is
    """
    n_selected = len(selected_m6_indices)

    if activations.shape[0] == n_selected:
        return activations

    if activations.shape[0] == n_m6_nodes:
        return activations[selected_m6_indices]

    raise ValueError(
        f"Activation node count {activations.shape[0]} does not match either "
        f"selected node count {n_selected} or full m6 node count {n_m6_nodes}"
    )


# ============================================================
# PCA PROJECTION
# ============================================================
def project_activations_to_pc_nodes(
    activations: np.ndarray,
    pca_mean: np.ndarray,
    pca_components: np.ndarray,
    n_pcs: int,
) -> tuple[list[str], np.ndarray]:
    """
    activations: [n_nodes, n_activation_features]

    Returns:
      pc_names: list[str]
      pc_nodes: [n_pcs, n_nodes]
    """
    if activations.shape[1] != pca_mean.shape[0]:
        raise ValueError(
            f"Activation feature dimension {activations.shape[1]} does not match "
            f"pca_mean {pca_mean.shape[0]}"
        )

    centered = activations - pca_mean
    pc_scores = centered @ pca_components[:n_pcs].T  # [n_nodes, n_pcs]
    pc_nodes = pc_scores.T.astype(np.float32)        # [n_pcs, n_nodes]

    pc_names = [f"PC_{i + 1}" for i in range(n_pcs)]
    return pc_names, pc_nodes


# ============================================================
# ERA5 MESH LOADING
# ============================================================
def load_mesh_catalog(era5_root: Path):
    time_values_path = era5_root / "time_values.npy"
    time_series_dir = era5_root / "time_series"
    static_dir = era5_root / "static"
    vertices_path = era5_root / "mesh_vertices.npy"

    if not time_values_path.exists():
        raise FileNotFoundError(f"Missing ERA5 mesh time file: {time_values_path}")

    if not vertices_path.exists():
        raise FileNotFoundError(f"Missing ERA5 mesh vertices file: {vertices_path}")

    time_values = np.load(time_values_path, allow_pickle=False)
    time_index = {
        np.datetime_as_string(np.datetime64(t), unit="h"): i
        for i, t in enumerate(time_values)
    }

    time_series = {
        path.stem: np.load(path, mmap_mode="r")
        for path in sorted(time_series_dir.glob("*.npy"))
    }

    static_fields = {
        path.stem: np.load(path, mmap_mode="r")
        for path in sorted(static_dir.glob("*.npy"))
    }

    if not time_series and not static_fields:
        raise FileNotFoundError(f"No ERA5 mesh fields found under {era5_root}")

    vertices = np.load(vertices_path, mmap_mode="r")

    return time_index, time_series, static_fields, vertices


def load_mesh_feature_nodes(
    time_index: dict,
    time_series: dict,
    static_fields: dict,
    center_str: str,
    selected_indices: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """
    Load one timestep's ERA5 fields and slice to selected mesh nodes.

    Returns:
      feature_names: list[str]
      era5_nodes: [n_features, n_selected_nodes]
    """
    if center_str not in time_index:
        return [], np.empty((0, 0), dtype=np.float32)

    t_idx = time_index[center_str]
    feature_names = []
    node_fields = []

    for name in sorted(time_series.keys()):
        arr = time_series[name]          # [time, m6_node]
        nodes = to_float32(arr[t_idx])   # [m6_node]
        node_fields.append(nodes[selected_indices])
        feature_names.append(name)

    for name in sorted(static_fields.keys()):
        arr = static_fields[name]        # [m6_node]
        nodes = to_float32(arr)
        node_fields.append(nodes[selected_indices])
        feature_names.append(name)

    if not node_fields:
        return [], np.empty((0, 0), dtype=np.float32)

    return feature_names, np.stack(node_fields, axis=0)


# ============================================================
# CORRELATION
# ============================================================
def batch_correlation_matrix(P: np.ndarray, E: np.ndarray, min_points: int = 10):
    """
    P: [n_pcs, n_nodes]
    E: [n_features, n_nodes]
    """
    if P.ndim != 2 or E.ndim != 2:
        raise ValueError(f"Expected 2D matrices, got P={P.shape}, E={E.shape}")

    if P.shape[1] != E.shape[1]:
        raise ValueError(f"PC nodes {P.shape[1]} != ERA5 nodes {E.shape[1]}")

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

    valid_p = Pstd[:, 0] > 0
    valid_e = Estd[:, 0] > 0

    corr = np.full((P.shape[0], E.shape[0]), np.nan, dtype=np.float32)

    if valid_p.any() and valid_e.any():
        Pz = P0[valid_p] / Pstd[valid_p]
        Ez = E0[valid_e] / Estd[valid_e]
        corr[np.ix_(valid_p, valid_e)] = (Pz @ Ez.T) / n_used

    return corr, n_used


# ============================================================
# ANALYSIS
# ============================================================
def run_single_variable_analysis(
    activations_dir: Path,
    era5_root: Path,
    mesh_level: int,
    n_pcs_to_check: int,
    cache_path: Path,
    summary_path: Path,
    force_recompute: bool = False,
    max_timesteps: int | None = None,
):
    if cache_path.exists() and not force_recompute:
        with open(cache_path, "r") as f:
            return json.load(f)

    print("Loading PCA components and mean...")
    pca_components = np.load(PCA_COMPONENTS_PATH)
    pca_mean = np.load(PCA_MEAN_PATH)

    n_pcs = min(n_pcs_to_check, pca_components.shape[0])
    print(f"Analyzing {n_pcs} PCs")

    print("Loading ERA5 mesh catalog...")
    time_index, time_series, static_fields, era5_m6_vertices = load_mesh_catalog(era5_root)

    n_m6_nodes = int(era5_m6_vertices.shape[0])
    selected_indices = selected_m6_indices_for_mesh_level(
        mesh_level=mesh_level,
        era5_m6_vertices=era5_m6_vertices,
        splits=6,
    )
    n_selected_nodes = len(selected_indices)

    print(f"Stored ERA5 mesh nodes: {n_m6_nodes}")
    print(f"Requested mesh level: m{mesh_level}")
    print(f"Selected nodes for comparison: {n_selected_nodes}")
    print(f"Loaded {len(time_series)} time-series ERA5 fields")
    print(f"Loaded {len(static_fields)} static ERA5 fields")
    print(f"Loaded {len(time_index)} ERA5 timesteps")

    activation_files = sorted(activations_dir.glob("*.npy"))
    if max_timesteps is not None:
        activation_files = activation_files[:max_timesteps]

    centers = [parse_center_time(p) for p in activation_files]
    print(f"Found {len(activation_files)} activation files")

    pc_stats = defaultdict(lambda: defaultdict(lambda: {"sum_abs_r": 0.0, "count": 0, "wins": 0}))
    processed_timesteps = 0

    for act_file, center_str in zip(activation_files, centers):
        print(f"\nProcessing {center_str}")

        if center_str not in time_index:
            print("  Skipping: no matching ERA5 mesh timestep")
            continue

        activations = load_activation_matrix(act_file)
        activations = select_activation_nodes(
            activations=activations,
            selected_m6_indices=selected_indices,
            n_m6_nodes=n_m6_nodes,
        )

        feature_names, era5_nodes = load_mesh_feature_nodes(
            time_index=time_index,
            time_series=time_series,
            static_fields=static_fields,
            center_str=center_str,
            selected_indices=selected_indices,
        )

        if era5_nodes.size == 0:
            print("  Skipping: no ERA5 fields loaded")
            del activations
            gc.collect()
            continue

        pc_names, pc_nodes = project_activations_to_pc_nodes(
            activations=activations,
            pca_mean=pca_mean,
            pca_components=pca_components,
            n_pcs=n_pcs,
        )

        corr_matrix, n_used = batch_correlation_matrix(
            P=pc_nodes,
            E=era5_nodes,
            min_points=MIN_POINTS,
        )

        if corr_matrix is None:
            print(f"  Skipping: not enough valid nodes ({n_used})")
            del activations, era5_nodes, pc_nodes
            gc.collect()
            continue

        if VERBOSE_TIMESTEPS:
            print(f"  Correlation matrix computed using {n_used} nodes")

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
                print(
                    f"  {pc_name}: top variable = {best_name}, r={best_r:.3f}"
                )

            for j, feature_name in enumerate(feature_names):
                r = row[j]
                if not np.isfinite(r):
                    continue

                stats = pc_stats[pc_idx][feature_name]
                stats["sum_abs_r"] += abs(float(r))
                stats["count"] += 1

            pc_stats[pc_idx][best_name]["wins"] += 1

        processed_timesteps += 1

        del activations, era5_nodes, pc_nodes, corr_matrix
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
            "space": "mesh_nodes",
            "mesh_level": mesh_level,
            "n_selected_nodes": n_selected_nodes,
            "era5_root": str(era5_root),
            "activations_dir": str(activations_dir),
        }

        display_summary[pc_key] = {
            "top_variables": rows_sorted[:TOP_K],
            "total_timesteps": processed_timesteps,
            "mesh_level": mesh_level,
            "n_selected_nodes": n_selected_nodes,
        }

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    with open(summary_path, "w") as f:
        json.dump(display_summary, f, indent=2)

    print(f"\nSaved screening cache to {cache_path}")
    print(f"Saved yearly summary to {summary_path}")

    return cache


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="PC-to-ERA5 mesh-node correlation screening"
    )
    parser.add_argument(
        "--mesh-level",
        type=int,
        default=6,
        choices=[0, 1, 2, 3, 4, 5, 6],
        help="GraphCast mesh hierarchy level to compare on. Use 5 for quick m5 tests.",
    )
    parser.add_argument(
        "--activations-dir",
        type=Path,
        default=DEFAULT_ACTIVATIONS_DIR,
        help="Directory containing activation .npy files.",
    )
    parser.add_argument(
        "--era5-root",
        type=Path,
        default=DEFAULT_ERA5_ROOT,
        help="ERA5 mesh_l6 root directory.",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=N_PCS_TO_CHECK,
        help="Number of PCs to check.",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Optional cap on timesteps for quick tests.",
    )
    parser.add_argument(
        "--force-recompute-screening",
        action="store_true",
        help="Recompute and overwrite existing cache.",
    )

    args = parser.parse_args()

    cache_path = RESULTS_DIR / f"pc_era5_mesh_m{args.mesh_level}_screening_cache.json"
    summary_path = RESULTS_DIR / f"pc_era5_mesh_m{args.mesh_level}_yearly_top_variables.json"

    run_single_variable_analysis(
        activations_dir=args.activations_dir,
        era5_root=args.era5_root,
        mesh_level=args.mesh_level,
        n_pcs_to_check=args.n_pcs,
        cache_path=cache_path,
        summary_path=summary_path,
        force_recompute=args.force_recompute_screening,
        max_timesteps=args.max_timesteps,
    )


if __name__ == "__main__":
    main()