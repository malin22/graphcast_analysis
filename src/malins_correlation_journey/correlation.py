import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from graphcast import icosahedral_mesh


# =====================
# CONFIG
# =====================

PC_SCORES_PATHS = [
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
        "pc_scores_2021_from_2019_2020_pca_per_timestep_files.txt"
    ),

]

PCA_TIME_CHUNK_SIZE = 4

ERA5_MESH_BASE_DIR = Path(
    "/share/prj-4d/graphcast_shared/data/era5_daily_mesh"
)

# PCA scores are sampled at these mesh nodes.
NODE_HIERARCHY_LEVEL = 6
FINE_MESH_LEVEL = 6
MAX_PCS = 512

# Include mesh position and cyclic clock fields as additional ERA5/context
# features. Year progress is constant over nodes within one timestep, but it
# varies across time and is therefore meaningful for pooled node-time
# correlation.
INCLUDE_CONTEXT = True
INCLUDE_YEAR_PROGRESS = True

# Some installations store static arrays in one of these directories. The
# script checks them in order. Arrays directly under mesh_l6/static_fields,
# mesh_l6/static, or mesh_l6 are supported.
STATIC_DIR_NAMES = ("static_fields", "static")

OUT_DIR = Path(
    "plots/malins_experiments/2021_correlation_on_2020_19/"
    f"PCA/l{NODE_HIERARCHY_LEVEL}_nodes"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================
# BASIC HELPERS
# =====================


def to_float32(array: np.ndarray) -> np.ndarray:
    """Convert normal and bfloat16-like arrays to float32."""
    array = np.asarray(array)

    if array.dtype == np.dtype("|V2"):
        array = array.view(np.float16)

    return array.astype(np.float32, copy=False)


def parse_timestamp_from_path(path: str) -> pd.Timestamp:
    """Extract YYYY-MM-DDTHH from an activation or timestep filename."""
    filename = os.path.basename(path)
    match = re.search(r"t(\d{4})-(\d{2})-(\d{2})T(\d{2})", filename)

    if not match:
        raise ValueError(f"Could not parse timestamp from {filename}")

    year, month, day, hour = map(int, match.groups())
    return pd.Timestamp(year, month, day, hour)


def timestamp_key(value) -> str:
    """Return the common hourly key used for timestamp matching."""
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H")


def load_timestamps(files_txt: str) -> tuple[list[str], pd.DatetimeIndex]:
    with open(files_txt, "r", encoding="utf-8") as file:
        files = [line.strip() for line in file if line.strip()]

    timestamps = pd.DatetimeIndex(
        [parse_timestamp_from_path(path) for path in files]
    )
    return files, timestamps


# =====================
# MESH AND CONTEXT
# =====================


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


def get_mesh_vertices(splits: int = FINE_MESH_LEVEL) -> np.ndarray:
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=splits
    )
    return np.asarray(meshes[splits].vertices)


def get_coarse_mesh_node_indices(
    fine_splits: int = FINE_MESH_LEVEL,
    coarse_splits: int = NODE_HIERARCHY_LEVEL,
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
        fields = cyclic_time_features(timestamp_key(timestamp), lon)
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


# =====================
# PCA LOADING
# =====================


def open_pca_sources(
    selected_indices: np.ndarray,
):
    """
    Open PCA files using memory mapping without loading their values.

    Returns:
        sources:
            List of dictionaries containing each memmap and timestamps.

        timestamps:
            Concatenated timestamps in the same order as the PCA sources.

        n_pcs:
            Number of PCs used.
    """
    sources = []
    timestamp_parts = []
    common_n_pcs = None

    for score_path, files_txt in zip(
        PC_SCORES_PATHS,
        TIMESTEP_FILES_TXTS,
    ):
        print(f"Opening PCA scores: {score_path}")

        _, timestamps = load_timestamps(files_txt)

        scores = np.load(
            score_path,
            mmap_mode="r",
        )

        if scores.ndim != 3:
            raise ValueError(
                f"Expected PCA shape (time,node,PC), "
                f"got {scores.shape}: {score_path}"
            )

        if scores.shape[0] != len(timestamps):
            raise ValueError(
                f"PCA/timestamp mismatch: "
                f"{scores.shape[0]} versus {len(timestamps)}"
            )

        if selected_indices.max() >= scores.shape[1]:
            raise IndexError(
                f"Selected mesh index exceeds PCA node dimension: "
                f"{score_path}"
            )

        source_n_pcs = min(
            MAX_PCS,
            scores.shape[2],
        )

        if common_n_pcs is None:
            common_n_pcs = source_n_pcs
        else:
            common_n_pcs = min(
                common_n_pcs,
                source_n_pcs,
            )

        sources.append(
            {
                "path": score_path,
                "scores": scores,
                "timestamps": timestamps,
            }
        )

        timestamp_parts.append(timestamps)

    timestamps = pd.DatetimeIndex(
        np.concatenate(
            [part.values for part in timestamp_parts]
        )
    )

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "PCA source timestamps are not globally ordered. "
            "Sort the input source files instead of sorting a huge PCA array."
        )

    return sources, timestamps, int(common_n_pcs)


# =====================
# ERA5 FILE DISCOVERY
# =====================


def mesh_dir_for_year(year: int) -> Path:
    return ERA5_MESH_BASE_DIR / str(year) / f"mesh_l{FINE_MESH_LEVEL}"


def discover_time_series_features(year: int) -> dict[str, Path]:
    directory = mesh_dir_for_year(year) / "time_series"
    if not directory.exists():
        raise FileNotFoundError(f"Missing ERA5 time-series directory: {directory}")

    return {
        path.stem: path
        for path in sorted(directory.glob("*.npy"))
    }


def discover_static_features(year: int) -> dict[str, Path]:
    mesh_dir = mesh_dir_for_year(year)
    found = {}

    for dirname in STATIC_DIR_NAMES:
        directory = mesh_dir / dirname
        if directory.exists():
            for path in sorted(directory.glob("*.npy")):
                found[path.stem] = path

    # Optionally accept node-length arrays stored directly in mesh_l6. Exclude
    # known metadata files. Shape validation happens when they are loaded.
    excluded = {"time_values", "vertices", "faces"}
    for path in sorted(mesh_dir.glob("*.npy")):
        if path.stem not in excluded and path.stem not in found:
            found[path.stem] = path

    return found


def common_feature_names(
    feature_maps: dict[int, dict[str, Path]],
    feature_kind: str,
) -> list[str]:
    if not feature_maps:
        return []

    name_sets = [set(mapping) for mapping in feature_maps.values()]
    common = set.intersection(*name_sets)
    union = set.union(*name_sets)

    missing = sorted(union - common)
    if missing:
        print(
            f"Warning: skipping {len(missing)} {feature_kind} features that "
            "are not present in every year"
        )

    return sorted(common)


def load_year_time_lookup(year: int) -> tuple[pd.DatetimeIndex, dict[str, int]]:
    path = mesh_dir_for_year(year) / "time_values.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing ERA5 time file: {path}")

    times = pd.DatetimeIndex(
        pd.to_datetime(np.load(path, allow_pickle=True))
    )
    lookup = {timestamp_key(value): index for index, value in enumerate(times)}
    return times, lookup


def requested_indices_by_year(
    timestamps: pd.DatetimeIndex,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Return year -> (positions in global PCA time axis, ERA5 time indices).
    """
    result = {}

    for year in sorted(timestamps.year.unique()):
        global_positions = np.flatnonzero(timestamps.year == year)
        _, lookup = load_year_time_lookup(int(year))

        era_indices = []
        missing = []
        for position in global_positions:
            key = timestamp_key(timestamps[position])
            if key not in lookup:
                missing.append(key)
            else:
                era_indices.append(lookup[key])

        if missing:
            raise KeyError(
                f"ERA5 year {year} is missing {len(missing)} PCA timestamps. "
                f"First missing: {missing[:5]}"
            )

        result[int(year)] = (
            global_positions,
            np.asarray(era_indices, dtype=np.int64),
        )

    return result


# =====================
# FEATURE LOADING
# =====================


def load_time_series_feature(
    feature_name: str,
    paths_by_year: dict[int, dict[str, Path]],
    indices_by_year: dict[int, tuple[np.ndarray, np.ndarray]],
    n_times: int,
    selected_indices: np.ndarray,
) -> np.ndarray:
    """Load one ERA5 time-series feature as (time, selected_node)."""
    output = np.empty((n_times, len(selected_indices)), dtype=np.float32)

    for year, (global_positions, era_indices) in indices_by_year.items():
        path = paths_by_year[year][feature_name]
        array = np.load(path, mmap_mode="r")

        if array.ndim != 2:
            raise ValueError(
                f"Expected time-series array (time, node), got {array.shape}: "
                f"{path}"
            )
        if selected_indices.max() >= array.shape[1]:
            raise IndexError(f"Selected node is outside array {path}")

        # Index time first and nodes second to avoid materializing the full
        # node grid. This only creates the selected time x selected node block.
        output[global_positions] = to_float32(
            array[era_indices][:, selected_indices]
        )

    return output


def load_static_feature(
    feature_name: str,
    path: Path,
    n_times: int,
    selected_indices: np.ndarray,
) -> np.ndarray:
    """Load one static node feature and broadcast it over requested times."""
    array = np.load(path, mmap_mode="r")
    array = np.squeeze(array)

    if array.ndim != 1:
        raise ValueError(f"Expected static node vector, got {array.shape}: {path}")
    if selected_indices.max() >= array.shape[0]:
        raise IndexError(f"Selected node is outside static array {path}")

    nodes = to_float32(array[selected_indices])
    return np.broadcast_to(
        nodes[None, :], (n_times, len(selected_indices))
    )


# =====================
# MATRIX-MULTIPLICATION CORRELATION
# =====================


def correlations_streaming(
    pca_sources,
    feature_grid: np.ndarray,
    selected_indices: np.ndarray,
    n_pcs: int,
    time_chunk_size: int = PCA_TIME_CHUNK_SIZE,
):
    """
    Calculate correlations between one ERA5 feature and all PCs without
    materializing the complete sample-by-PC matrix.

    feature_grid must have shape:
        (all_times, selected_nodes)

    PCA data are read from their memory-mapped files in small time chunks.
    """
    feature_grid = np.asarray(
        feature_grid,
        dtype=np.float32,
    )

    expected_times = sum(
        len(source["timestamps"])
        for source in pca_sources
    )

    if feature_grid.ndim != 2:
        raise ValueError(
            f"feature_grid must be 2D, got {feature_grid.shape}"
        )

    if feature_grid.shape[0] != expected_times:
        raise ValueError(
            f"Feature has {feature_grid.shape[0]} timesteps, "
            f"but PCA sources have {expected_times}"
        )

    if feature_grid.shape[1] != len(selected_indices):
        raise ValueError(
            f"Feature has {feature_grid.shape[1]} nodes, "
            f"but {len(selected_indices)} nodes were selected"
        )

    # Only these small PC-length arrays remain in memory.
    sum_x = np.zeros(
        n_pcs,
        dtype=np.float64,
    )

    sum_x2 = np.zeros(
        n_pcs,
        dtype=np.float64,
    )

    sum_xy = np.zeros(
        n_pcs,
        dtype=np.float64,
    )

    sum_y = 0.0
    sum_y2 = 0.0
    n_valid = 0

    global_time_offset = 0

    for source in pca_sources:
        scores = source["scores"]
        source_times = scores.shape[0]

        node_selector = get_node_selector(
            selected_indices,
            scores.shape[1],
        )

        for start in range(
            0,
            source_times,
            time_chunk_size,
        ):
            stop = min(
                start + time_chunk_size,
                source_times,
            )

            global_start = global_time_offset + start
            global_stop = global_time_offset + stop

            # This is the only substantial PCA allocation.
            x_chunk = to_float32(
                scores[
                    start:stop,
                    node_selector,
                    :n_pcs,
                ]
            )

            y_chunk = feature_grid[
                global_start:global_stop
            ]

            x_chunk = x_chunk.reshape(
                -1,
                n_pcs,
            )

            y_chunk = y_chunk.reshape(-1)

            # Most likely only y contains missing values.
            valid_y = np.isfinite(y_chunk)

            if not valid_y.any():
                del x_chunk
                continue

            # Check PC finiteness only within the small chunk.
            valid_x = np.all(
                np.isfinite(x_chunk),
                axis=1,
            )

            valid = valid_y & valid_x

            if not valid.any():
                del x_chunk
                continue

            # These copies are limited to one chunk.
            xv = x_chunk[valid]
            yv = y_chunk[valid]

            chunk_n = int(yv.size)

            n_valid += chunk_n

            sum_y += np.sum(
                yv,
                dtype=np.float64,
            )

            sum_y2 += np.einsum(
                "i,i->",
                yv,
                yv,
                dtype=np.float64,
            )

            sum_x += np.sum(
                xv,
                axis=0,
                dtype=np.float64,
            )

            sum_x2 += np.einsum(
                "ij,ij->j",
                xv,
                xv,
                dtype=np.float64,
            )

            sum_xy += np.einsum(
                "ij,i->j",
                xv,
                yv,
                dtype=np.float64,
            )

            del x_chunk
            del xv
            del yv

        global_time_offset += source_times

    correlations = np.full(
        n_pcs,
        np.nan,
        dtype=np.float64,
    )

    if n_valid < 3:
        return correlations, n_valid

    numerator = (
        sum_xy
        - sum_x * sum_y / n_valid
    )

    x_ss = (
        sum_x2
        - sum_x * sum_x / n_valid
    )

    y_ss = (
        sum_y2
        - sum_y * sum_y / n_valid
    )

    x_ss = np.maximum(
        x_ss,
        0.0,
    )

    y_ss = max(
        y_ss,
        0.0,
    )

    denominator = np.sqrt(
        x_ss * y_ss
    )

    usable = denominator > 0.0

    correlations[usable] = (
        numerator[usable]
        / denominator[usable]
    )

    correlations = np.clip(
        correlations,
        -1.0,
        1.0,
    )

    return correlations, n_valid


def append_feature_results(
    results: list[dict],
    feature_name: str,
    feature_kind: str,
    feature_grid: np.ndarray,
    pca_sources,
    selected_indices: np.ndarray,
    n_pcs: int,
) -> None:
    correlations, n_valid = correlations_streaming(
        pca_sources=pca_sources,
        feature_grid=feature_grid,
        selected_indices=selected_indices,
        n_pcs=n_pcs,
        time_chunk_size=PCA_TIME_CHUNK_SIZE,
    )

    for pc_number, correlation in enumerate(
        correlations,
        start=1,
    ):
        results.append(
            {
                "feature": feature_name,
                "feature_kind": feature_kind,
                "pc": pc_number,
                "correlation": correlation,
                "abs_correlation": (
                    abs(correlation)
                    if np.isfinite(correlation)
                    else np.nan
                ),
                "n_valid": n_valid,
            }
        )

    if np.isfinite(correlations).any():
        best = int(
            np.nanargmax(
                np.abs(correlations)
            )
        )

        print(
            f"  strongest: PC{best + 1}, "
            f"r={correlations[best]:.5f}, "
            f"n={n_valid}"
        )

    else:
        print(
            f"  no valid correlation, n={n_valid}"
        )

def get_node_selector(
    selected_indices: np.ndarray,
    total_nodes: int,
):
    """
    Return slice(None) when all nodes are selected in their original order.

    A slice is cheaper than NumPy advanced indexing.
    """
    expected = np.arange(
        total_nodes,
        dtype=selected_indices.dtype,
    )

    if (
        len(selected_indices) == total_nodes
        and np.array_equal(selected_indices, expected)
    ):
        return slice(None)

    return selected_indices
# =====================
# MAIN
# =====================


def main() -> None:
    selected_indices = get_coarse_mesh_node_indices()
    vertices = get_mesh_vertices()

    pca_sources, timestamps, n_pcs = open_pca_sources(
        selected_indices
    )

    n_times = len(timestamps)
    n_nodes = len(selected_indices)

    print(
        f"PCA sources: {len(pca_sources)}"
    )
    print(
        f"Logical PCA shape: "
        f"({n_times}, {n_nodes}, {n_pcs})"
    )
    print(
        f"Samples: {n_times * n_nodes:,}"
    )
    print(
        f"Time range: {timestamps.min()} to {timestamps.max()}"
    )

    years = sorted(int(year) for year in timestamps.year.unique())
    time_paths_by_year = {
        year: discover_time_series_features(year) for year in years
    }
    static_paths_by_year = {
        year: discover_static_features(year) for year in years
    }

    time_feature_names = common_feature_names(
        time_paths_by_year, "time-series"
    )
    static_feature_names = common_feature_names(
        static_paths_by_year, "static"
    )
    indices_by_year = requested_indices_by_year(timestamps)

    print(f"Time-series ERA5 features found: {len(time_feature_names)}")
    print(f"Static ERA5 features found: {len(static_feature_names)}")

    results: list[dict] = []

    # Load one ERA5 variable at a time. This is the main memory-saving step:
    # the script never constructs a samples x all_ERA5_features matrix.
    for number, feature_name in enumerate(time_feature_names, start=1):
        print(
            f"[{number}/{len(time_feature_names)}] "
            f"time-series: {feature_name}"
        )
        feature_grid = load_time_series_feature(
            feature_name=feature_name,
            paths_by_year=time_paths_by_year,
            indices_by_year=indices_by_year,
            n_times=n_times,
            selected_indices=selected_indices,
        )
        append_feature_results(
            results=results,
            feature_name=feature_name,
            feature_kind="era5_time_series",
            feature_grid=feature_grid,
            pca_sources=pca_sources,
            selected_indices=selected_indices,
            n_pcs=n_pcs,
        )
        del feature_grid

    # Static fields should normally be identical across years. Use the first
    # year's file and report a warning if file presence differs.
    first_year = years[0]
    for number, feature_name in enumerate(static_feature_names, start=1):
        print(
            f"[{number}/{len(static_feature_names)}] static: {feature_name}"
        )
        try:
            feature_grid = load_static_feature(
                feature_name=feature_name,
                path=static_paths_by_year[first_year][feature_name],
                n_times=n_times,
                selected_indices=selected_indices,
            )
        except (ValueError, IndexError) as error:
            print(f"  skipping incompatible static file: {error}")
            continue

        append_feature_results(
            results=results,
            feature_name=feature_name,
            feature_kind="era5_static",
            feature_grid=feature_grid,
            pca_sources=pca_sources,
            selected_indices=selected_indices,
            n_pcs=n_pcs,
        )
        del feature_grid

    if INCLUDE_CONTEXT:
        print("Building latitude/longitude/time context fields")
        context_features = build_context_features(
            timestamps=timestamps,
            selected_indices=selected_indices,
            vertices=vertices,
            include_year_progress=INCLUDE_YEAR_PROGRESS,
        )

        for feature_name, feature_grid in context_features.items():
            print(f"context: {feature_name}")
            append_feature_results(
                results=results,
                feature_name=feature_name,
                feature_kind="context",
                feature_grid=feature_grid,
                pca_sources=pca_sources,
                selected_indices=selected_indices,
                n_pcs=n_pcs,
            )

    results_df = pd.DataFrame(results)

    long_path = OUT_DIR / "all_era5_context_pc_correlations_long.csv"
    results_df.to_csv(long_path, index=False)

    matrix_df = results_df.pivot(
        index=["feature_kind", "feature"],
        columns="pc",
        values="correlation",
    )
    matrix_df.columns = [f"PC{number}" for number in matrix_df.columns]
    matrix_path = OUT_DIR / "all_era5_context_pc_correlation_matrix.csv"
    matrix_df.to_csv(matrix_path)

    valid_df = results_df.dropna(subset=["correlation"]).copy()
    if valid_df.empty:
        strongest_df = valid_df
    else:
        strongest_indices = valid_df.groupby(
            ["feature_kind", "feature"]
        )["abs_correlation"].idxmax()
        strongest_df = valid_df.loc[strongest_indices].sort_values(
            "abs_correlation", ascending=False
        )

    strongest_path = OUT_DIR / "all_era5_context_strongest_pc.csv"
    strongest_df.to_csv(strongest_path, index=False)

    print("\nSaved:")
    print(f"  {long_path}")
    print(f"  {matrix_path}")
    print(f"  {strongest_path}")


if __name__ == "__main__":
    main()