#!/usr/bin/env python3
import argparse
import json
import os
import tarfile
from glob import glob
from pathlib import Path

import numpy as np
from graphcast import icosahedral_mesh


DEFAULT_ACTS_DIR = Path("/share/prj-4d/graphcast_shared/data/graphcast_activation_2020")
DEFAULT_OUT_DIR = Path("/share/prj-4d/graphcast_shared/data/tensor_decomposition_subsets")
DEFAULT_PATTERN = "layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t*.npy"


def load_activations(path: Path) -> np.ndarray:
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    if x.ndim != 2:
        raise ValueError(f"Expected [nodes, features], got shape {x.shape} for {path}")

    return x.astype(np.float32)


def parse_time(path: Path) -> np.datetime64:
    center_str = path.stem.split("_t")[-1]
    return np.datetime64(center_str, "h")


def month_bounds(year: int, month: int):
    start = np.datetime64(f"{year:04d}-{month:02d}-01T00", "h")
    if month == 12:
        end = np.datetime64(f"{year + 1:04d}-01-01T00", "h")
    else:
        end = np.datetime64(f"{year:04d}-{month + 1:02d}-01T00", "h")
    return start, end


def mesh_hierarchy_indices(splits: int = 6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    verts = [m.vertices for m in meshes]
    finest = verts[splits]

    def to_key(v):
        return tuple(np.round(v, 12))

    finest_map = {to_key(v): i for i, v in enumerate(finest)}

    def level_idx_in_finest(level):
        return np.array([finest_map[to_key(v)] for v in verts[level]], dtype=np.int64)

    idx_by_level_cumulative = {}
    idx_by_level_only = {}

    prev = np.array([], dtype=np.int64)
    for level in range(splits + 1):
        idx = level_idx_in_finest(level)
        idx_by_level_cumulative[level] = idx
        idx_by_level_only[level] = np.setdiff1d(idx, prev)
        prev = idx

    return finest, idx_by_level_cumulative, idx_by_level_only


def vertices_to_latlon(vertices: np.ndarray):
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    return lat.astype(np.float32), lon.astype(np.float32)


def select_month_files(acts_dir: Path, pattern: str, year: int, month: int):
    start, end = month_bounds(year, month)

    selected = []
    for path in sorted(acts_dir.glob(pattern)):
        t = parse_time(path)
        if start <= t < end:
            selected.append((t, path))

    return selected


def save_mlevel_month_tensor(
    acts_dir: Path,
    out_dir: Path,
    year: int,
    month: int,
    level: int,
    splits: int,
    pattern: str,
    dtype: str,
    make_tar: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = select_month_files(acts_dir, pattern, year, month)
    if not rows:
        raise FileNotFoundError(
            f"No files found for {year}-{month:02d} in {acts_dir} with pattern {pattern}"
        )

    timestamps = np.array([t for t, _ in rows], dtype="datetime64[h]")
    files = [p for _, p in rows]

    finest_vertices, idx_by_level_cum, _ = mesh_hierarchy_indices(splits=splits)
    if level not in idx_by_level_cum:
        raise ValueError(f"Requested level {level} is not available for splits={splits}")

    selected_idx = idx_by_level_cum[level]
    selected_vertices = finest_vertices[selected_idx]
    selected_lat, selected_lon = vertices_to_latlon(selected_vertices)

    first = load_activations(files[0])
    if first.shape[0] <= int(selected_idx.max()):
        raise ValueError(
            f"First activation file has {first.shape[0]} nodes, "
            f"but M{level} requires index {int(selected_idx.max())}"
        )

    n_nodes = len(selected_idx)
    n_times = len(files)
    n_features = first.shape[1]
    out_dtype = np.dtype(dtype)

    base = f"graphcast_layer0008_m{level}_nodes_{year:04d}_{month:02d}"
    tensor_path = out_dir / f"{base}.npy"

    tensor = np.lib.format.open_memmap(
        tensor_path,
        mode="w+",
        dtype=out_dtype,
        shape=(n_nodes, n_times, n_features),
    )

    for t_idx, path in enumerate(files):
        acts = load_activations(path)

        if acts.shape != first.shape:
            raise ValueError(
                f"Shape mismatch in {path.name}: {acts.shape}, expected {first.shape}"
            )

        if np.isnan(acts).any():
            raise ValueError(f"NaNs found in {path.name}")

        tensor[:, t_idx, :] = acts[selected_idx].astype(out_dtype, copy=False)

        if (t_idx + 1) % 20 == 0 or t_idx == n_times - 1:
            print(f"Processed {t_idx + 1}/{n_times}: {path.name}", flush=True)

    tensor.flush()

    np.save(out_dir / f"{base}_timestamps.npy", timestamps)
    np.save(out_dir / f"{base}_m6_indices.npy", selected_idx)
    np.save(out_dir / f"{base}_vertices_xyz.npy", selected_vertices.astype(np.float32))
    np.save(out_dir / f"{base}_lat.npy", selected_lat)
    np.save(out_dir / f"{base}_lon.npy", selected_lon)

    metadata = {
        "year": int(year),
        "month": int(month),
        "acts_dir": str(acts_dir),
        "pattern": pattern,
        "mesh_level": int(level),
        "splits": int(splits),
        "tensor_shape": [int(n_nodes), int(n_times), int(n_features)],
        "tensor_layout": "[node, time, feature]",
        "dtype": str(out_dtype),
        "n_files": int(n_times),
        "first_timestamp": np.datetime_as_string(timestamps[0], unit="h"),
        "last_timestamp": np.datetime_as_string(timestamps[-1], unit="h"),
        "tensor_path": str(tensor_path),
        "notes": "Cumulative M-level nodes selected from full M6 GraphCast mesh-node activations.",
    }

    metadata_path = out_dir / f"{base}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved tensor: {tensor_path}")
    print(f"Shape: {tensor.shape}")
    print(f"Saved metadata: {metadata_path}")

    if make_tar:
        archive_path = out_dir / f"{base}.tar.gz"
        members = [
            tensor_path,
            out_dir / f"{base}_timestamps.npy",
            out_dir / f"{base}_m6_indices.npy",
            out_dir / f"{base}_vertices_xyz.npy",
            out_dir / f"{base}_lat.npy",
            out_dir / f"{base}_lon.npy",
            metadata_path,
        ]

        with tarfile.open(archive_path, "w:gz") as tar:
            for member in members:
                tar.add(member, arcname=member.name)

        print(f"Saved archive: {archive_path}")

    return tensor_path


def main():
    parser = argparse.ArgumentParser(
        description="Create a one-month tensor of cumulative M-level GraphCast activations."
    )
    parser.add_argument("--acts-dir", type=Path, default=DEFAULT_ACTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--splits", type=int, default=6)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--no-tar", action="store_true")

    args = parser.parse_args()

    save_mlevel_month_tensor(
        acts_dir=args.acts_dir,
        out_dir=args.out_dir,
        year=args.year,
        month=args.month,
        level=args.level,
        splits=args.splits,
        pattern=args.pattern,
        dtype=args.dtype,
        make_tar=not args.no_tar,
    )


if __name__ == "__main__":
    main()