#!/usr/bin/env python3
import gc
import json
from pathlib import Path

import numpy as np
import xarray as xr
from graphcast import icosahedral_mesh
from graphcast import grid_mesh_connectivity

INPUT_DIR = Path("/share/prj-4d/graphcast_shared/data/era5_daily_nc")
OUTPUT_DIR = Path("/share/prj-4d/graphcast_shared/data/era5_daily_mesh")
YEAR = 2021
MESH_LEVEL = 5

# This matches the GraphCast-style radius fraction.
# GraphCast docs describe reasonable values as 0.6 to 1.0.
RADIUS_QUERY_FRACTION_EDGE_LENGTH = 0.6

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
]


def list_year_files(input_dir: Path, year: int) -> list[Path]:
    return sorted(input_dir.glob(f"era5_{year:04d}-*.nc"))


def open_era5(path: Path) -> xr.Dataset:
    ds = xr.open_dataset(path)

    if "latitude" in ds.coords and "lat" not in ds.coords:
        ds = ds.rename({"latitude": "lat"})
    if "longitude" in ds.coords and "lon" not in ds.coords:
        ds = ds.rename({"longitude": "lon"})

    if ds["lat"][0] < ds["lat"][-1]:
        ds = ds.sortby("lat")

    return ds


def to_float32(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == np.dtype("|V2"):
        arr = arr.view(np.float16)
    return np.asarray(arr, dtype=np.float32)


def get_mesh(mesh_level: int):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=mesh_level
    )
    return meshes[-1]


def max_mesh_edge_distance(mesh) -> float:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)

    a = mesh.vertices[edges[:, 0]]
    b = mesh.vertices[edges[:, 1]]
    return float(np.linalg.norm(a - b, axis=1).max())


def build_era5_to_mesh_mapper(
    lat_1d,
    lon_1d,
    mesh_level: int,
    radius_query_fraction_edge_length: float = RADIUS_QUERY_FRACTION_EDGE_LENGTH,
):
    """
    Build a GraphCast-style ERA5 grid -> mesh mapper.

    This uses graphcast.grid_mesh_connectivity.radius_query_indices,
    the same connectivity helper GraphCast uses for its Grid2Mesh graph.
    """
    lat_1d = to_float32(lat_1d)
    lon_1d = to_float32(lon_1d)

    mesh = get_mesh(mesh_level)
    radius = max_mesh_edge_distance(mesh) * float(radius_query_fraction_edge_length)

    grid_indices, mesh_indices = grid_mesh_connectivity.radius_query_indices(
        grid_latitude=lat_1d,
        grid_longitude=lon_1d,
        mesh=mesh,
        radius=radius,
    )

    n_mesh_nodes = mesh.vertices.shape[0]
    counts = np.zeros(n_mesh_nodes, dtype=np.int32)
    np.add.at(counts, mesh_indices, 1)

    return {
        "mesh_level": int(mesh_level),
        "mesh_vertices": np.asarray(mesh.vertices, dtype=np.float32),
        "grid_indices": np.asarray(grid_indices, dtype=np.int64),
        "mesh_indices": np.asarray(mesh_indices, dtype=np.int64),
        "counts": counts,
        "input_shape": (len(lat_1d), len(lon_1d)),
        "radius_chord": float(radius),
        "radius_query_fraction_edge_length": float(radius_query_fraction_edge_length),
    }


def era5_to_mesh(variable_grid, mesh_level: int, mapper) -> np.ndarray:
    """
    Project one ERA5 lat/lon field to mesh nodes using GraphCast-style
    grid-to-mesh connectivity.

    This is a simple mean over all grid points connected to each mesh node.
    """
    field = to_float32(variable_grid)
    if field.ndim != 2:
        raise ValueError(f"Expected a 2D [lat, lon] field, got shape {field.shape}")

    if tuple(field.shape) != tuple(mapper["input_shape"]):
        raise ValueError(
            f"Field shape {field.shape} does not match mapper input shape "
            f"{mapper['input_shape']}"
        )

    flat = field.ravel()
    grid_indices = mapper["grid_indices"]
    mesh_indices = mapper["mesh_indices"]
    counts = mapper["counts"]

    mesh_values = np.zeros(len(counts), dtype=np.float64)

    values = flat[grid_indices].astype(np.float64, copy=False)
    finite = np.isfinite(values)

    np.add.at(mesh_values, mesh_indices[finite], values[finite])

    # No fallback logic: mirrors the simple implementation style.
    mesh_values = mesh_values / np.maximum(counts, 1)

    return mesh_values.astype(np.float32)


def feature_specs(ds: xr.Dataset):
    time_features = []
    static_features = []

    for var in DEFAULT_VARS:
        if var not in ds.data_vars:
            continue

        dims = ds[var].dims

        if "time" in dims:
            if "level" in dims:
                nlev = ds[var].sizes["level"]
                for lev in range(nlev):
                    time_features.append((f"{var}_lev{lev:02d}", var, lev))
            else:
                time_features.append((var, var, None))
        else:
            static_features.append((var, var, None))

    return time_features, static_features


def main():
    year_dir = OUTPUT_DIR / f"{YEAR}" / f"mesh_l{MESH_LEVEL}"
    time_dir = year_dir / "time_series"
    static_dir = year_dir / "static"

    time_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    files = list_year_files(INPUT_DIR, YEAR)
    if not files:
        raise FileNotFoundError(f"No ERA5 files found for {YEAR} in {INPUT_DIR}")

    print(f"Found {len(files)} daily files")

    first = open_era5(files[0])
    lat = to_float32(first["lat"].values)
    lon = to_float32(first["lon"].values)

    mapper = build_era5_to_mesh_mapper(
        lat,
        lon,
        mesh_level=MESH_LEVEL,
        radius_query_fraction_edge_length=RADIUS_QUERY_FRACTION_EDGE_LENGTH,
    )

    n_mesh_nodes = mapper["mesh_vertices"].shape[0]
    counts = mapper["counts"]

    print(f"Mesh level {MESH_LEVEL}: {n_mesh_nodes} nodes")
    print(
        "ERA5 grid points per mesh node: "
        f"min={counts.min()}, median={np.median(counts)}, max={counts.max()}"
    )
    print("Grid2Mesh radius chord distance:", mapper["radius_chord"])

    time_features, static_features = feature_specs(first)

    print("Time features:")
    for name, _, _ in time_features:
        print(" ", name)

    print("Static features:")
    for name, _, _ in static_features:
        print(" ", name)

    first.close()

    total_times = 0
    for path in files:
        ds = open_era5(path)
        total_times += int(ds.sizes.get("time", 0))
        ds.close()

    print(f"Total time steps in year: {total_times}")

    time_series = {}
    for name, _, _ in time_features:
        out_path = time_dir / f"{name}.npy"
        time_series[name] = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_times, n_mesh_nodes),
        )

    static_outputs = {}
    for name, _, _ in static_features:
        out_path = static_dir / f"{name}.npy"
        static_outputs[name] = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_mesh_nodes,),
        )

    time_values = []
    cursor = 0

    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path.name}")

        ds = open_era5(path)

        if "time" in ds.coords:
            time_values.extend([np.datetime64(t) for t in ds["time"].values])

        for out_name, var_name, lev_idx in time_features:
            arr = to_float32(ds[var_name].values)

            if arr.ndim == 3:
                for t in range(arr.shape[0]):
                    time_series[out_name][cursor + t] = era5_to_mesh(
                        arr[t],
                        mesh_level=MESH_LEVEL,
                        mapper=mapper,
                    )

            elif arr.ndim == 4:
                for t in range(arr.shape[0]):
                    time_series[out_name][cursor + t] = era5_to_mesh(
                        arr[t, lev_idx],
                        mesh_level=MESH_LEVEL,
                        mapper=mapper,
                    )

            else:
                raise ValueError(f"Unexpected shape for {var_name}: {arr.shape}")

        if cursor == 0:
            for out_name, var_name, _ in static_features:
                arr = to_float32(ds[var_name].values)

                static_outputs[out_name][:] = era5_to_mesh(
                    arr,
                    mesh_level=MESH_LEVEL,
                    mapper=mapper,
                )

        cursor += int(ds.sizes.get("time", 0))

        ds.close()
        del ds
        gc.collect()

    for mm in time_series.values():
        mm.flush()

    for mm in static_outputs.values():
        mm.flush()

    metadata = {
        "year": YEAR,
        "input_dir": str(INPUT_DIR),
        "output_dir": str(year_dir),
        "mesh_level": MESH_LEVEL,
        "n_mesh_nodes": int(n_mesh_nodes),
        "radius_query_fraction_edge_length": float(RADIUS_QUERY_FRACTION_EDGE_LENGTH),
        "radius_chord": float(mapper["radius_chord"]),
        "neighbor_count_min": int(counts.min()),
        "neighbor_count_median": float(np.median(counts)),
        "neighbor_count_max": int(counts.max()),
        "time_features": [name for name, _, _ in time_features],
        "static_features": [name for name, _, _ in static_features],
        "time_values_count": len(time_values),
    }

    with open(year_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    np.save(year_dir / "mesh_vertices.npy", mapper["mesh_vertices"])
    np.save(year_dir / "time_values.npy", np.array(time_values, dtype="datetime64[ns]"))

    print(f"Saved mesh ERA5 fields to: {year_dir}")


if __name__ == "__main__":
    main()