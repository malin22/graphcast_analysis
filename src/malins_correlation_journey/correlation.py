import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from malins_correlation_journey.mesh_context import (
    build_context_features,
    get_coarse_mesh_node_indices,
    get_mesh_vertices,
)

from malins_correlation_journey.correlation_maths import (
    correlations_streaming,
    to_float32,
)

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
    return ERA5_MESH_BASE_DIR / str(year) / f"mesh_l{NODE_HIERARCHY_LEVEL}"


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

# =====================
# MAIN
# =====================


def main() -> None:
    selected_indices = get_coarse_mesh_node_indices( fine_splits=NODE_HIERARCHY_LEVEL, coarse_splits=NODE_HIERARCHY_LEVEL,)

    vertices = get_mesh_vertices(splits=NODE_HIERARCHY_LEVEL,)

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