#!/usr/bin/env python3
import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from graphcast import icosahedral_mesh
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

'''
Compute a colocated spatial Pearson REGRESSION over mesh nodes.

The ERA5 values used:
DEFAULT_VARS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
    "toa_incident_solar_radiation",
    "geopotential_at_surface",
    "land_sea_mask",

    #new:
    latitude
    longitude_sin
    longitude_cos
    local_time_sin
    local_time_cos

'''

DEFAULT_ACTIVATIONS_DIR = Path("/share/prj-4d/graphcast_shared/data/graphcast_activation_2021") #val set 2021
DEFAULT_ERA5_ROOT = Path("/share/prj-4d/graphcast_shared/data/era5_daily_mesh/2021/mesh_l6") #val set 2021

PCA_COMPONENTS_PATH = "/share/prj-4d/graphcast_shared/data/pca_components/512_PCs/layer8_only/pca_components_2019_2020_layer8.npy" #train set coordinates 2019/2020
PCA_MEAN_PATH = "/share/prj-4d/graphcast_shared/data/pca_components/512_PCs/layer8_only/pca_mean_2019_2020_layer8.npy"  #train set coordinates 2019/2020

RESULTS_DIR = Path("plots/sabines_experiments/mapping_experiments/top_50_pcs")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PC_SCORES_PATH = Path(
    "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
    "pc_scores_2021_from_2019_2020_pca_per_timestep.npy"
)

MIN_POINTS = 10


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

def vertices_to_latlon(vertices: np.ndarray):
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    return lat.astype(np.float32), lon.astype(np.float32)


def cyclic_time_features(center_str: str, lon_deg: np.ndarray):
    """
    Build GraphCast-like clock/context fields on mesh nodes.

    local_time_* varies over longitude and UTC time.
    year_progress_* is constant over nodes for a timestep, so it is not useful
    for per-timestep spatial correlation, but useful for node-time regression.
    """
    t = np.datetime64(center_str, "h")

    # UTC hour of day as fraction [0, 1).
    day = t.astype("datetime64[D]")
    hours_since_midnight = (t - day) / np.timedelta64(1, "h")
    utc_day_fraction = float(hours_since_midnight) / 24.0

    # Local solar time: longitude shifts UTC time by lon / 360 of a day.
    lon_fraction = lon_deg.astype(np.float32) / 360.0
    local_day_fraction = (utc_day_fraction + lon_fraction) % 1.0

    local_time_angle = 2.0 * np.pi * local_day_fraction
    local_time_sin = np.sin(local_time_angle).astype(np.float32)
    local_time_cos = np.cos(local_time_angle).astype(np.float32)

    # Year progress. This is global/constant over nodes at one timestep.
    year = int(str(t)[:4])
    year_start = np.datetime64(f"{year}-01-01T00", "h")
    year_end = np.datetime64(f"{year + 1}-01-01T00", "h")

    year_fraction = float((t - year_start) / (year_end - year_start))
    year_angle = 2.0 * np.pi * year_fraction

    return {
        "local_time_sin": local_time_sin,
        "local_time_cos": local_time_cos,
        "year_progress_sin": np.full_like(lon_deg, np.sin(year_angle), dtype=np.float32),
        "year_progress_cos": np.full_like(lon_deg, np.cos(year_angle), dtype=np.float32),
    }

def build_pc_scores_row_mapping(
    pc_scores_all: np.ndarray,
    pc_scores_files_list: Path,
) -> dict[str, int]:
    """
    Build center_str -> pc_scores_all row index mapping from the
    authoritative files-list manifest (one activation file path per line,
    in the same row order as pc_scores_all).
    """
    with open(pc_scores_files_list) as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) != pc_scores_all.shape[0]:
        raise ValueError(
            f"{pc_scores_files_list} lists {len(lines)} files but "
            f"pc_scores_all has {pc_scores_all.shape[0]} rows -- these must "
            "match exactly for row indexing to be trustworthy."
        )

    mapping = {}
    for i, line in enumerate(lines):
        # Each line may be "path" or "path\tsize" -- take the path only.
        path_str = line.split("\t")[0]
        center_str = parse_center_time(Path(path_str))
        if center_str in mapping:
            raise ValueError(
                f"Duplicate center_str {center_str!r} found in "
                f"{pc_scores_files_list} at row {i} (already mapped to row "
                f"{mapping[center_str]})"
            )
        mapping[center_str] = i

    return mapping


# ============================================================
# MESH LEVEL SELECTION
# ============================================================
def get_graphcast_mesh_vertices(level: int, splits: int = 6) -> np.ndarray:
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    return np.asarray(meshes[level].vertices, dtype=np.float32)


def vertex_key(v: np.ndarray, decimals: int = 12):
    return tuple(np.round(v.astype(np.float64), decimals))


def selected_m6_indices_for_mesh_level(mesh_level: int, era5_m6_vertices: np.ndarray) -> np.ndarray:
    era5_m6_vertices = np.asarray(era5_m6_vertices, dtype=np.float32)
    n_m6 = era5_m6_vertices.shape[0]

    if mesh_level == 6:
        return np.arange(n_m6, dtype=np.int64)

    if mesh_level < 0 or mesh_level > 6:
        raise ValueError("mesh_level must be between 0 and 6")

    target_vertices = get_graphcast_mesh_vertices(mesh_level, splits=6)
    m6_lookup = {vertex_key(v): i for i, v in enumerate(era5_m6_vertices)}

    selected = []
    for v in target_vertices:
        key = vertex_key(v)
        if key not in m6_lookup:
            raise ValueError(f"Could not match an m{mesh_level} vertex inside stored m6 vertices")
        selected.append(m6_lookup[key])

    selected = np.asarray(selected, dtype=np.int64)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Duplicate selected m6 indices found")

    return selected


def select_activation_nodes(activations: np.ndarray, selected_m6_indices: np.ndarray, n_m6_nodes: int):
    n_selected = len(selected_m6_indices)

    if activations.shape[0] == n_selected:
        return activations

    if activations.shape[0] == n_m6_nodes:
        return activations[selected_m6_indices]

    raise ValueError(
        f"Activation node count {activations.shape[0]} does not match "
        f"selected nodes {n_selected} or full m6 nodes {n_m6_nodes}"
    )

# ============================================================
# ERA5 MESH LOADING
# ============================================================
def load_mesh_catalog(era5_root: Path):
    time_values = np.load(era5_root / "time_values.npy", allow_pickle=False)
    time_index = {
        np.datetime_as_string(np.datetime64(t), unit="h"): i
        for i, t in enumerate(time_values)
    }

    time_series = {
        p.stem: np.load(p, mmap_mode="r")
        for p in sorted((era5_root / "time_series").glob("*.npy"))
    }
    static_fields = {
        p.stem: np.load(p, mmap_mode="r")
        for p in sorted((era5_root / "static").glob("*.npy"))
    }
    vertices = np.load(era5_root / "mesh_vertices.npy", mmap_mode="r")

    if not time_series and not static_fields:
        raise FileNotFoundError(f"No ERA5 mesh fields found under {era5_root}")

    return time_index, time_series, static_fields, vertices

def load_era5_X_for_timestep(
    time_index: dict,
    time_series: dict,
    static_fields: dict,
    center_str: str,
    selected_indices: np.ndarray,
    vertices: np.ndarray,
    include_context: bool = True,
    include_year_progress: bool = False,
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
        arr = time_series[name]
        nodes = to_float32(arr[t_idx])
        node_fields.append(nodes[selected_indices])
        feature_names.append(name)

    for name in sorted(static_fields.keys()):
        arr = static_fields[name]
        nodes = to_float32(arr)
        node_fields.append(nodes[selected_indices])
        feature_names.append(name)

    if include_context:
        lat, lon = vertices_to_latlon(np.asarray(vertices))
        lat = lat[selected_indices]
        lon = lon[selected_indices]

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)

        context_fields = {
            "latitude": lat.astype(np.float32),
            "latitude_sin": np.sin(lat_rad).astype(np.float32),
            "longitude_sin": np.sin(lon_rad).astype(np.float32),
            "longitude_cos": np.cos(lon_rad).astype(np.float32),
        }

        context_fields.update(cyclic_time_features(center_str, lon))

        for name, nodes in context_fields.items():
            if name.startswith("year_progress") and not include_year_progress:
                continue

            node_fields.append(nodes.astype(np.float32))
            feature_names.append(name)

    if not node_fields:
        return [], np.empty((0, 0), dtype=np.float32)

    return feature_names, np.stack(node_fields, axis=0)

# ============================================================
# PCA PROJECTION
# ============================================================

def project_pc(activations, pca_mean, pca_components, pc_idx: int):
    if activations.shape[1] != pca_mean.shape[0]:
        raise ValueError(
            f"Activation feature dim {activations.shape[1]} != PCA mean dim {pca_mean.shape[0]}"
        )
    centered = activations - pca_mean
    return (centered @ pca_components[pc_idx]).astype(np.float32)  # [nodes]


def finite_and_optional_sample(X, y, max_nodes=None, rng=None):
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    idx = np.flatnonzero(mask)

    if len(idx) < MIN_POINTS:
        return None, None

    if max_nodes is not None and len(idx) > max_nodes:
        idx = rng.choice(idx, size=max_nodes, replace=False)

    return X[idx], y[idx]




# ============================================================
# REGRESSION
# ============================================================
def fit_full_grid(X_train_z, y_train_z, X_val_z, y_val_z, model_type, alpha_grid, l1_ratio_grid):
    """
    Sweep alpha x l1_ratio and return every fitted result (not just the best),
    so downstream analysis can trade off val_r2 against sparsity.

    Returns:
      grid_results: list[dict] with l1_ratio, alpha, val_r2, n_nonzero, coef
      best_model: the ElasticNet/LinearRegression model object for the
                   highest-val_r2 point (kept for convenience/back-compat;
                   selection logic for "best" should generally happen
                   downstream using grid_results, not this).
    """
    model_type = model_type.lower()

    if model_type in ("linear", "linearregression", "ols"):
        print("  fitting LinearRegression", flush=True)
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train_z, y_train_z)
        val_r2 = float(model.score(X_val_z, y_val_z))
        coef = np.asarray(model.coef_, dtype=np.float32)
        grid_results = [{
            "l1_ratio": None,
            "alpha": None,
            "val_r2": val_r2,
            "n_nonzero": int(np.sum(np.abs(coef) > 1e-8)),
            "n_features": int(len(coef)),
        }]
        return grid_results, model

    grid_results = []
    best_model = None
    best_val_r2 = -np.inf

    total = len(l1_ratio_grid) * len(alpha_grid)
    done = 0

    for l1_ratio in l1_ratio_grid:
        for alpha in alpha_grid:
            done += 1
            print(
                f"  fitting {done}/{total}: l1_ratio={l1_ratio}, alpha={alpha}",
                flush=True,
            )

            if float(l1_ratio) == 0.0:
                # sklearn's ElasticNet disallows l1_ratio=0; Ridge is the
                # equivalent limiting case.
                model = Ridge(alpha=float(alpha), fit_intercept=True)
            else:
                model = ElasticNet(
                    alpha=float(alpha),
                    l1_ratio=float(l1_ratio),
                    fit_intercept=True,
                    max_iter=5000,
                    tol=1e-3,
                    random_state=0,
                    selection="random",
                    precompute=True,
                )

            model.fit(X_train_z, y_train_z)
            val_r2 = float(model.score(X_val_z, y_val_z))
            coef = np.asarray(model.coef_, dtype=np.float32)
            n_nonzero = int(np.sum(np.abs(coef) > 1e-8))

            grid_results.append({
                "l1_ratio": float(l1_ratio),
                "alpha": float(alpha),
                "val_r2": val_r2,
                "n_nonzero": n_nonzero,
                "n_features": int(len(coef)),
            })

            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_model = model

    return grid_results, best_model


def select_by_tolerance(grid_results, r2_tolerance=0.02):
    """
    Among grid points within r2_tolerance of the best val_r2, pick the
    sparsest (fewest nonzero coefficients). Falls back to best-by-r2 if
    grid_results has only one entry (e.g. Linear).
    """
    if len(grid_results) == 1:
        return grid_results[0]

    best_r2 = max(r["val_r2"] for r in grid_results)
    threshold = best_r2 - r2_tolerance
    candidates = [r for r in grid_results if r["val_r2"] >= threshold]
    return min(candidates, key=lambda r: r["n_nonzero"])

def rank_coefficients(feature_names, coefs):
    rows = [
        {"feature": name, "coefficient": float(c), "abs_coefficient": abs(float(c))}
        for name, c in zip(feature_names, coefs)
    ]
    return sorted(rows, key=lambda r: (-r["abs_coefficient"], r["feature"]))


def field_name(feature_name: str) -> str:
    match = re.match(r"^(.*)_lev\d+$", feature_name)
    return match.group(1) if match else feature_name


def level_index(feature_name: str):
    match = re.match(r"^.*_lev(\d+)$", feature_name)
    return int(match.group(1)) if match else None


def summarize_grouped_importance(feature_names, coefs):
    field_scores = defaultdict(float)
    level_scores = defaultdict(float)

    for name, coef in zip(feature_names, coefs):
        weight = abs(float(coef))
        field_scores[field_name(name)] += weight
        lev = level_index(name)
        level_scores["static" if lev is None else f"lev{lev:02d}"] += weight

    return {
        "field_importance": [
            {"field": k, "abs_weight_sum": v}
            for k, v in sorted(field_scores.items(), key=lambda kv: -kv[1])
        ],
        "level_importance": [
            {"level": k, "abs_weight_sum": v}
            for k, v in sorted(level_scores.items(), key=lambda kv: -kv[1])
        ],
    }


def atomic_write_json(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)

# ============================================================
# ANALYSIS
# ============================================================
def run_regression(args):
    rng = np.random.default_rng(args.random_seed)

    pca_components = np.load(args.pca_components)
    pca_mean = np.load(args.pca_mean)

    time_index, time_series, static_fields, era5_m6_vertices = load_mesh_catalog(args.era5_root)
    n_m6_nodes = int(era5_m6_vertices.shape[0])

    selected_indices = selected_m6_indices_for_mesh_level(args.mesh_level, era5_m6_vertices)
    n_selected_nodes = len(selected_indices)

    pattern = "layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t*.npy"
    activation_files = sorted(args.activations_dir.glob(pattern))
    #activation_files = sorted(args.activations_dir.glob("*.npy"))
    
    usable = [(p, parse_center_time(p)) for p in activation_files if parse_center_time(p) in time_index]

    # --- NEW: load precomputed PC scores + build center_str -> array-index map ---
    pc_scores_all = None
    center_to_pc_row = {}
    if not args.recompute_pc_scores:
        pc_scores_all = np.load(args.pc_scores_path, mmap_mode="r")

        if pc_scores_all.shape[1] != n_m6_nodes:
            raise ValueError(
                f"pc_scores_path has {pc_scores_all.shape[1]} nodes but "
                f"expected the full m6 mesh ({n_m6_nodes} nodes)."
            )

        center_to_pc_row = build_pc_scores_row_mapping(
            pc_scores_all=pc_scores_all,
            pc_scores_files_list=args.pc_scores_files_list,
        )
        print(
            f"Loaded PC-score row mapping for {len(center_to_pc_row)} "
            f"timesteps from {args.pc_scores_files_list}"
        )
    # --- end NEW ---

    if args.max_timesteps is not None:
        usable = usable[: args.max_timesteps]

    if not usable:
        raise ValueError("No activation files matched ERA5 time index")

    n_train = max(1, int(len(usable) * args.train_fraction))
    train_steps = usable[:n_train]
    val_steps = usable[n_train:] if n_train < len(usable) else usable[-1:]

    if args.pc_indices:
        pc_indices = args.pc_indices
    else:
        pc_indices = list(range(min(args.n_pcs, pca_components.shape[0])))

    if args.model_type.lower() in ("linear", "linearregression", "ols"):
        alpha_grid = None
        l1_ratio_grid = None
    elif args.alpha_grid:
        alpha_grid = np.array(args.alpha_grid, dtype=np.float64)
        l1_ratio_grid = np.array(args.l1_ratio_grid, dtype=np.float64)
    else:
        alpha_grid = np.logspace(-3, 1, 16)
        l1_ratio_grid = np.array(args.l1_ratio_grid, dtype=np.float64)

    results = {}

    print(f"ERA5 root: {args.era5_root}")
    print(f"Mesh level: m{args.mesh_level}")
    print(f"Selected nodes: {n_selected_nodes}")
    print(f"Usable timesteps: {len(usable)}")
    print(f"Train/val timesteps: {len(train_steps)}/{len(val_steps)}")
    print(f"Model: {args.model_type}")

    for pc_idx in pc_indices:
        print(f"\nFitting PC_{pc_idx + 1}")

        X_train_blocks, y_train_blocks = [], []
        X_val_blocks, y_val_blocks = [], []
        feature_names_ref = None

        for split_name, steps in [("train", train_steps), ("val", val_steps)]:
            for step_i, (act_file, center_str) in enumerate(steps, start=1):
                if step_i % 25 == 0 or step_i == len(steps):
                    print(
                        f"  {split_name}: loaded {step_i}/{len(steps)} timesteps",
                        flush=True,
                    )
                #activations = load_activation_matrix(act_file)
                #activations = select_activation_nodes(activations, selected_indices, n_m6_nodes)

                feature_names, era5_nodes = load_era5_X_for_timestep(
                    time_index=time_index,
                    time_series=time_series,
                    static_fields=static_fields,
                    center_str=center_str,
                    selected_indices=selected_indices,
                    vertices=era5_m6_vertices,
                    include_context=True,
                    include_year_progress=True, # for correlation keep false, for regression true!
                )

                if feature_names_ref is None:
                    feature_names_ref = feature_names
                elif feature_names != feature_names_ref:
                    raise ValueError("ERA5 feature ordering changed between timesteps")
                
                if era5_nodes.size == 0:
                    continue
                X = era5_nodes.T  # [nodes, features]

                if args.recompute_pc_scores:
                    activations = load_activation_matrix(act_file)
                    activations = select_activation_nodes(activations, selected_indices, n_m6_nodes)
                    y = project_pc(activations, pca_mean, pca_components, pc_idx)
                    del activations
                else:
                    pc_row = center_to_pc_row.get(center_str)
                    if pc_row is None:
                        print(f"  Skipping {center_str}: no precomputed PC score for this timestep")
                        continue
                    y = np.asarray(
                        pc_scores_all[pc_row, selected_indices, pc_idx], dtype=np.float32
                    )

                X_use, y_use = finite_and_optional_sample(
                    X,
                    y,
                    max_nodes=args.max_nodes_per_timestep,
                    rng=rng,
                )
                if X_use is None:
                    continue

                if split_name == "train":
                    X_train_blocks.append(X_use)
                    y_train_blocks.append(y_use)
                else:
                    X_val_blocks.append(X_use)
                    y_val_blocks.append(y_use)

                del X, y, X_use, y_use
                gc.collect()


        if not X_train_blocks or not X_val_blocks:
            print(f"Skipping PC_{pc_idx + 1}: not enough train/validation data")
            continue

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

        grid_results, best_model = fit_full_grid(
            X_train_z,
            y_train_z,
            X_val_z,
            y_val_z,
            args.model_type,
            alpha_grid,
            l1_ratio_grid,
        )

        chosen = select_by_tolerance(grid_results, r2_tolerance=args.r2_tolerance)

        # refit the chosen (l1_ratio, alpha) to get its coefficients
        # (grid search above didn't retain every model object to save memory)
        if chosen["l1_ratio"] is None:
            model = LinearRegression(fit_intercept=True)
        elif chosen["l1_ratio"] == 0.0:
            model = Ridge(alpha=chosen["alpha"], fit_intercept=True)
        else:
            model = ElasticNet(
                alpha=chosen["alpha"],
                l1_ratio=chosen["l1_ratio"],
                fit_intercept=True,
                max_iter=5000,
                tol=1e-3,
                random_state=0,
                selection="random",
                precompute=True,
            )
        model.fit(X_train_z, y_train_z)

        val_r2 = chosen["val_r2"]
        alpha = chosen["alpha"]
        l1_ratio_chosen = chosen["l1_ratio"]

        coef_std = np.asarray(model.coef_, dtype=np.float32)
        ranked = rank_coefficients(feature_names_ref, coef_std)
        grouped = summarize_grouped_importance(feature_names_ref, coef_std)

        result = {
            "pc_name": f"PC_{pc_idx + 1}",
            "pc_idx": int(pc_idx),
            "mesh_level": int(args.mesh_level),
            "n_selected_nodes": int(n_selected_nodes),
            "model_type": args.model_type,
            "alpha": alpha, 
            "l1_ratio": l1_ratio_chosen,
            "r2_tolerance": float(args.r2_tolerance),
            "val_r2": float(val_r2),
            "n_features": int(len(feature_names_ref)),
            "feature_names": feature_names_ref,
            "ranked_features_standardized": ranked,
            "coef_standardized": {
                name: float(coef)
                for name, coef in zip(feature_names_ref, coef_std)
            },
            "field_importance": grouped["field_importance"],
            "level_importance": grouped["level_importance"],
            "n_train_samples": int(X_train.shape[0]),
            "n_val_samples": int(X_val.shape[0]),
            "train_timesteps": int(len(train_steps)),
            "val_timesteps": int(len(val_steps)),
            "max_nodes_per_timestep": args.max_nodes_per_timestep,
            "grid_search": grid_results,   # NEW: full alpha x l1_ratio sweep
        }

        results[f"PC_{pc_idx + 1}"] = result
        atomic_write_json(args.output_path, results)

        print(f"PC_{pc_idx + 1}: val R2 = {val_r2:.4f}, alpha = {alpha}, l1_ratio = {l1_ratio_chosen}")
        print("Top standardized coefficients:")
        for row in ranked[:10]:
            print(row)

        del X_train, y_train, X_val, y_val, X_train_z, y_train_z, X_val_z, y_val_z
        gc.collect()

    atomic_write_json(args.output_path, results)
    print(f"\nSaved all-variable regression results to {args.output_path}")

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="All-variable ERA5 mesh-node regression for GraphCast PCs")
    parser.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    parser.add_argument("--era5-root", type=Path, default=DEFAULT_ERA5_ROOT)
    parser.add_argument("--mesh-level", type=int, choices=[0, 1, 2, 3, 4, 5, 6], default=6)
    parser.add_argument("--model-type", choices=["Linear", "Ridge", "Lasso", "ElasticNet"], default="ElasticNet")
    parser.add_argument(
        "--l1-ratio-grid",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        help="l1_ratio values to sweep (0=Ridge-like, 1=Lasso-like).",
    )
    parser.add_argument(
        "--r2-tolerance",
        type=float,
        default=0.02,
        help="Accept models within this much val_r2 of the best, then pick sparsest.",
    )
    parser.add_argument("--n-pcs", type=int, default=20)
    parser.add_argument("--pc-indices", type=int, nargs="*", default=None)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-timesteps", type=int, default=None)
    parser.add_argument("--max-nodes-per-timestep", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--alpha-grid", type=float, nargs="*", default=None)
    parser.add_argument("--pca-components", default=PCA_COMPONENTS_PATH)
    parser.add_argument("--pca-mean", default=PCA_MEAN_PATH)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Where to save regression JSON. Defaults to results dir by mesh/model.",
    )
    parser.add_argument("--pc-scores-path", type=Path, default=DEFAULT_PC_SCORES_PATH)
    parser.add_argument(
        "--pc-scores-files-list",
        type=Path,
        default=Path(
            "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
            "pc_scores_2021_per_timestep_files.txt"
        ),
    )
    parser.add_argument(
        "--recompute-pc-scores",
        action="store_true",
        help="Fall back to on-the-fly projection via project_pc instead of the precomputed array.",
    )

    args = parser.parse_args()

    if args.output_path is None:
        model = args.model_type.lower()
        args.output_path = RESULTS_DIR / f"regression_pc_era5_mesh_m{args.mesh_level}_allvars_{model}_results.json"

    run_regression(args)


if __name__ == "__main__":
    main()