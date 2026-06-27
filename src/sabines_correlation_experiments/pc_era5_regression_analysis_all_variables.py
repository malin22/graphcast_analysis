#!/usr/bin/env python3
import argparse
import gc
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from graphcast import icosahedral_mesh
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

DEFAULT_ACTIVATIONS_DIR = Path("/share/prj-4d/graphcast_shared/data/graphcast_activation_2021")
DEFAULT_ERA5_ROOT = Path("/share/prj-4d/graphcast_shared/data/era5_daily_mesh/2021/mesh_l6")

PCA_COMPONENTS_PATH = "/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021.npy"
PCA_MEAN_PATH = "/share/prj-4d/graphcast_shared/data/pca_components/pca_mean_2021.npy"

RESULTS_DIR = Path("plots/sabines_experiments")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_POINTS = 10


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


def load_era5_X_for_timestep(time_index, time_series, static_fields, center_str, selected_indices):
    if center_str not in time_index:
        return None, None

    t_idx = time_index[center_str]
    feature_names = []
    cols = []

    for name in sorted(time_series.keys()):
        vals = to_float32(time_series[name][t_idx])[selected_indices]
        cols.append(vals)
        feature_names.append(name)

    for name in sorted(static_fields.keys()):
        vals = to_float32(static_fields[name])[selected_indices]
        cols.append(vals)
        feature_names.append(name)

    X = np.stack(cols, axis=1).astype(np.float32)  # [nodes, features]
    return feature_names, X


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


def fit_with_alpha_grid(X_train_z, y_train_z, X_val_z, y_val_z, model_type, alpha_grid, l1_ratio):
    model_type = model_type.lower()
    best_model = None
    best_score = -np.inf
    best_alpha = None

    for alpha_i, alpha in enumerate(alpha_grid, start=1):
        print(
            f"  fitting alpha {alpha_i}/{len(alpha_grid)}: {alpha}",
            flush=True,
        )
        if model_type == "ridge":
            model = Ridge(alpha=float(alpha), fit_intercept=True)
        elif model_type == "lasso":
            model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=30000, tol=1e-4)
        elif model_type == "elasticnet":
            model = ElasticNet(
                alpha=float(alpha),
                l1_ratio=float(l1_ratio),
                fit_intercept=True,
                max_iter=5000,
                tol=1e-3,
                random_state=0,
                selection='random',
                precompute=True,
            )
        else:
            raise ValueError("model_type must be Ridge, Lasso, or ElasticNet")

        model.fit(X_train_z, y_train_z)
        score = model.score(X_val_z, y_val_z)

        if score > best_score:
            best_model = model
            best_score = float(score)
            best_alpha = float(alpha)

    return best_model, best_score, best_alpha


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


def run_regression(args):
    rng = np.random.default_rng(args.random_seed)

    pca_components = np.load(args.pca_components)
    pca_mean = np.load(args.pca_mean)

    time_index, time_series, static_fields, era5_m6_vertices = load_mesh_catalog(args.era5_root)
    n_m6_nodes = int(era5_m6_vertices.shape[0])

    selected_indices = selected_m6_indices_for_mesh_level(args.mesh_level, era5_m6_vertices)
    n_selected_nodes = len(selected_indices)

    activation_files = sorted(args.activations_dir.glob("*.npy"))
    usable = [(p, parse_center_time(p)) for p in activation_files if parse_center_time(p) in time_index]

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

    if args.alpha_grid:
        alpha_grid = np.array(args.alpha_grid, dtype=np.float64)
    elif args.model_type.lower() == "ridge":
        alpha_grid = np.logspace(-3, 4, 20)
    else:
        alpha_grid = np.logspace(-3, 1, 16)

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
                activations = load_activation_matrix(act_file)
                activations = select_activation_nodes(activations, selected_indices, n_m6_nodes)

                feature_names, X = load_era5_X_for_timestep(
                    time_index,
                    time_series,
                    static_fields,
                    center_str,
                    selected_indices,
                )

                if feature_names_ref is None:
                    feature_names_ref = feature_names
                elif feature_names != feature_names_ref:
                    raise ValueError("ERA5 feature ordering changed between timesteps")

                y = project_pc(activations, pca_mean, pca_components, pc_idx)

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

                del activations, X, y, X_use, y_use
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

        model, val_r2, alpha = fit_with_alpha_grid(
            X_train_z,
            y_train_z,
            X_val_z,
            y_val_z,
            args.model_type,
            alpha_grid,
            args.l1_ratio,
        )

        coef_std = np.asarray(model.coef_, dtype=np.float32)
        ranked = rank_coefficients(feature_names_ref, coef_std)
        grouped = summarize_grouped_importance(feature_names_ref, coef_std)

        result = {
            "pc_name": f"PC_{pc_idx + 1}",
            "pc_idx": int(pc_idx),
            "mesh_level": int(args.mesh_level),
            "n_selected_nodes": int(n_selected_nodes),
            "model_type": args.model_type,
            "alpha": float(alpha),
            "l1_ratio": float(args.l1_ratio) if args.model_type.lower() == "elasticnet" else None,
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
        }

        results[f"PC_{pc_idx + 1}"] = result
        atomic_write_json(args.output_path, results)

        print(f"PC_{pc_idx + 1}: val R2 = {val_r2:.4f}, alpha = {alpha}")
        print("Top standardized coefficients:")
        for row in ranked[:10]:
            print(row)

        del X_train, y_train, X_val, y_val, X_train_z, y_train_z, X_val_z, y_val_z
        gc.collect()

    atomic_write_json(args.output_path, results)
    print(f"\nSaved all-variable regression results to {args.output_path}")


def main():
    parser = argparse.ArgumentParser(description="All-variable ERA5 mesh-node regression for GraphCast PCs")
    parser.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    parser.add_argument("--era5-root", type=Path, default=DEFAULT_ERA5_ROOT)
    parser.add_argument("--mesh-level", type=int, choices=[0, 1, 2, 3, 4, 5, 6], default=6)
    parser.add_argument("--model-type", choices=["Ridge", "Lasso", "ElasticNet"], default="Ridge")
    parser.add_argument("--l1-ratio", type=float, default=0.5)
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

    args = parser.parse_args()

    if args.output_path is None:
        model = args.model_type.lower()
        args.output_path = RESULTS_DIR / f"pc_era5_mesh_m{args.mesh_level}_allvars_{model}_results.json"

    run_regression(args)


if __name__ == "__main__":
    main()