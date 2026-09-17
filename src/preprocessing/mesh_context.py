import numpy as np
import pandas as pd

from graphcast import icosahedral_mesh


def timestamp_key(value) -> str:
    """Return the common hourly key used for timestamp matching."""
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H")


def vertices_to_latlon(vertices: np.ndarray):
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    return lat.astype(np.float32), lon.astype(np.float32)


def cyclic_time_features(center_str: str, lon_deg: np.ndarray):
    """
    Build GraphCast-like clock/context fields on mesh nodes.

    local_time_* varies over longitude and UTC time.
    year_progress_* is constant over nodes for one timestep but varies over
    time, so it can be useful for pooled node-time correlation.
    """
    t = np.datetime64(center_str, "h")

    day = t.astype("datetime64[D]")
    hours_since_midnight = (t - day) / np.timedelta64(1, "h")
    utc_day_fraction = float(hours_since_midnight) / 24.0

    lon_fraction = lon_deg.astype(np.float32) / 360.0
    local_day_fraction = (utc_day_fraction + lon_fraction) % 1.0

    local_time_angle = 2.0 * np.pi * local_day_fraction
    local_time_sin = np.sin(local_time_angle).astype(np.float32)
    local_time_cos = np.cos(local_time_angle).astype(np.float32)

    year = int(str(t)[:4])
    year_start = np.datetime64(f"{year}-01-01T00", "h")
    year_end = np.datetime64(f"{year + 1}-01-01T00", "h")

    year_fraction = float((t - year_start) / (year_end - year_start))
    year_angle = 2.0 * np.pi * year_fraction

    return {
        "local_time_sin": local_time_sin,
        "local_time_cos": local_time_cos,
        "year_progress_sin": np.full_like(
            lon_deg, np.sin(year_angle), dtype=np.float32
        ),
        "year_progress_cos": np.full_like(
            lon_deg, np.cos(year_angle), dtype=np.float32
        ),
    }


def get_mesh_vertices(splits: int) -> np.ndarray:
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=splits
    )
    return np.asarray(meshes[splits].vertices)

def get_mesh_latlon(splits=6):
    vertices = get_mesh_vertices(splits)
    return vertices_to_latlon(vertices)


def get_coarse_mesh_node_indices(
    fine_splits: int,
    coarse_splits: int,
    decimals: int = 8,
) -> np.ndarray:
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=fine_splits
    )

    fine_vertices = np.asarray(meshes[fine_splits].vertices)
    coarse_vertices = np.asarray(meshes[coarse_splits].vertices)

    fine_lookup = {
        tuple(np.round(vertex, decimals)): index
        for index, vertex in enumerate(fine_vertices)
    }

    indices = []

    for vertex in coarse_vertices:
        key = tuple(np.round(vertex, decimals))

        if key not in fine_lookup:
            raise ValueError("Could not match coarse vertex to fine mesh")

        indices.append(fine_lookup[key])

    return np.asarray(indices, dtype=np.int64)


def build_context_features(
    timestamps: pd.DatetimeIndex,
    selected_indices: np.ndarray,
    vertices: np.ndarray,
    include_year_progress: bool,
) -> dict[str, np.ndarray]:
    """
    Construct context arrays in PCA ordering: time first, then selected node.

    Every returned array has shape (n_times, n_selected_nodes).
    """
    lat_all, lon_all = vertices_to_latlon(vertices)

    lat = lat_all[selected_indices]
    lon = lon_all[selected_indices]

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    n_times = len(timestamps)

    context = {
        "context__latitude": np.broadcast_to(
            lat[None, :], (n_times, len(lat))
        ).astype(np.float32, copy=True),

        "context__latitude_sin": np.broadcast_to(
            np.sin(lat_rad)[None, :], (n_times, len(lat))
        ).astype(np.float32, copy=True),

        "context__longitude_sin": np.broadcast_to(
            np.sin(lon_rad)[None, :], (n_times, len(lat))
        ).astype(np.float32, copy=True),

        "context__longitude_cos": np.broadcast_to(
            np.cos(lon_rad)[None, :], (n_times, len(lat))
        ).astype(np.float32, copy=True),
    }

    local_sin = np.empty((n_times, len(lat)), dtype=np.float32)
    local_cos = np.empty_like(local_sin)

    if include_year_progress:
        year_sin = np.empty_like(local_sin)
        year_cos = np.empty_like(local_sin)

    for time_index, timestamp in enumerate(timestamps):
        fields = cyclic_time_features(
            timestamp_key(timestamp),
            lon,
        )

        local_sin[time_index] = fields["local_time_sin"]
        local_cos[time_index] = fields["local_time_cos"]

        if include_year_progress:
            year_sin[time_index] = fields["year_progress_sin"]
            year_cos[time_index] = fields["year_progress_cos"]

    context["context__local_time_sin"] = local_sin
    context["context__local_time_cos"] = local_cos

    if include_year_progress:
        context["context__year_progress_sin"] = year_sin
        context["context__year_progress_cos"] = year_cos

    return context