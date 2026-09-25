import csv
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from graphcast import icosahedral_mesh

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse



# =============================================================================
# Configuration
# =============================================================================

PC_SCORES_PATH = (
    "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
    "pc_scores_2021_from_2019_2020_pca_per_timestep.npy"
)

PC_SCORES_FILES_LIST = (
    "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
    "pc_scores_2021_from_2019_2020_pca_per_timestep_files.txt"
)

GLORYS_GLOB = (
    "/share/prj-4d/graphcast_shared/data/glorys/"
    "thetao_2021_monthly/glorys12_thetao_2021_*.nc"
)

OUT_DIR = (
    "/share/prj-4d/graphcast_shared/data/glorys/outdata/"
    "pc2_glorys_thetao_metrics_2021"
)

CACHE_FILENAME = "cached_pc2_glorys_2021_plot_data.npz"


YEAR = 2021
PC_INDEX = 1              # Zero-based index: 1 = PC2
PC_SIGN = -1.0            # PCA signs are arbitrary. Flip PC2 so its temperature relationship is positive.
MESH_SPLITS = 6
REQUIRE_ALL_GLORYS_MONTHS = True


# =============================================================================
# PC-score row auditing
# =============================================================================

TIMESTAMP_RE = re.compile(
    r"_t(\d{4}-\d{2}-\d{2}T\d{2}(?::\d{2})?(?::\d{2})?)"
)


def parse_timestamp_from_filename(path):
    """Extract a GraphCast activation timestamp from an activation filename."""
    match = TIMESTAMP_RE.search(os.path.basename(path))
    if match is None:
        raise ValueError(f"Could not parse timestamp from: {path}")

    return datetime.fromisoformat(match.group(1))


def load_pc_score_metadata(pc_scores_path, files_list_path):
    """
    Read PC-score array metadata and verify each row has a source activation file.
    PC-score row i corresponds exactly to non-empty line i of files_list_path.
    """
    scores = np.load(pc_scores_path, mmap_mode="r")

    if scores.ndim != 3:
        raise ValueError(
            f"Expected PC scores [timesteps, nodes, PCs], got {scores.shape}"
        )

    with open(files_list_path) as f:
        source_files = [line.strip() for line in f if line.strip()]

    if len(source_files) != scores.shape[0]:
        raise ValueError(
            f"Mismatch: PC scores has {scores.shape[0]} rows but "
            f"the files list has {len(source_files)} non-empty lines."
        )

    timestamps = []
    for row, source_file in enumerate(source_files):
        timestamps.append(parse_timestamp_from_filename(source_file))

    return scores, source_files, timestamps


def audit_timestamps(timestamps, source_files, out_dir):
    """Write timestamp/coverage diagnostics and return row indices by month."""
    rows_by_month = defaultdict(list)
    duplicates = defaultdict(list)

    for row, timestamp in enumerate(timestamps):
        rows_by_month[timestamp.month].append(row)
        duplicates[timestamp].append(row)

    duplicate_timestamps = {
        timestamp: rows
        for timestamp, rows in duplicates.items()
        if len(rows) > 1
    }

    unique_times = sorted(set(timestamps))
    deltas = [
        int((later - earlier).total_seconds())
        for earlier, later in zip(unique_times[:-1], unique_times[1:])
    ]

    expected_step_seconds = None
    missing_internal_times = []

    if deltas:
        expected_step_seconds = Counter(deltas).most_common(1)[0][0]
        expected_step = timedelta(seconds=expected_step_seconds)

        current = unique_times[0]
        final = unique_times[-1]
        observed = set(unique_times)

        while current <= final:
            if current not in observed:
                missing_internal_times.append(current)
            current += expected_step

    audit_path = os.path.join(out_dir, "pc_score_input_audit.txt")
    with open(audit_path, "w") as f:
        f.write(f"PC-score source files: {len(source_files)}\n")
        f.write(f"First timestamp: {min(timestamps).isoformat()}\n")
        f.write(f"Last timestamp:  {max(timestamps).isoformat()}\n")
        f.write(f"Duplicate timestamps: {len(duplicate_timestamps)}\n")

        if expected_step_seconds is not None:
            f.write(
                f"Inferred nominal timestep: "
                f"{expected_step_seconds / 3600:.2f} hours\n"
            )

        f.write(
            f"Missing timestamps within observed date range: "
            f"{len(missing_internal_times)}\n\n"
        )

        f.write("Rows per calendar month:\n")
        for month in range(1, 13):
            f.write(f"  {YEAR}-{month:02d}: {len(rows_by_month[month])}\n")

        if duplicate_timestamps:
            f.write("\nDuplicate timestamps:\n")
            for timestamp, rows in sorted(duplicate_timestamps.items()):
                f.write(f"  {timestamp.isoformat()}: rows {rows}\n")

        if missing_internal_times:
            f.write("\nMissing internal timestamps:\n")
            for timestamp in missing_internal_times:
                f.write(f"  {timestamp.isoformat()}\n")

    print(f"Wrote PC-score audit: {audit_path}")
    return rows_by_month


# =============================================================================
# GraphCast mesh
# =============================================================================

def vertices_to_latlon(vertices):
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    return lat, lon


def get_mesh_latlon(splits=6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=splits
    )
    return vertices_to_latlon(meshes[splits].vertices)


# =============================================================================
# GLORYS loading and interpolation
# =============================================================================

def coordinate_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    raise KeyError(f"Could not find any of {candidates} in dataset coordinates.")


def load_glorys_surface_thetao(path):
    """
    Return monthly mean GLORYS surface potential temperature [lat, lon].
    The query already requested the shallowest depth, but this safely selects it.
    """
    with xr.open_dataset(path) as ds:
        if "thetao" not in ds:
            raise KeyError(f"{path} does not contain thetao")

        field = ds["thetao"]

        for depth_name in ("depth", "deptho", "lev"):
            if depth_name in field.dims:
                field = field.isel({depth_name: 0})
                break

        if "time" in field.dims:
            field = field.mean("time", skipna=True)

        lat_name = coordinate_name(field, ("latitude", "lat"))
        lon_name = coordinate_name(field, ("longitude", "lon"))

        return field.transpose(lat_name, lon_name).load(), lat_name, lon_name


def interpolate_to_mesh(field, lat_name, lon_name, mesh_lat, mesh_lon):
    """
    Bilinear, periodic-longitude interpolation from regular GLORYS grid to mesh.

    Ocean-land values are NaN in GLORYS and remain NaN after interpolation.
    """
    field = field.transpose(lat_name, lon_name)

    lat = np.asarray(field[lat_name].values)
    lon = np.mod(np.asarray(field[lon_name].values), 360.0)
    values = np.asarray(field.values, dtype=np.float64)

    # Ensure latitude and longitude coordinates increase.
    lat_order = np.argsort(lat)
    lat = lat[lat_order]
    values = values[lat_order, :]

    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    values = values[:, lon_order]

    # Remove a possible duplicated 0/360 longitude.
    lon, unique_indices = np.unique(lon, return_index=True)
    values = values[:, unique_indices]

    # Make the interpolation explicitly periodic at the date line.
    lon_extended = np.concatenate(([lon[-1] - 360.0], lon, [lon[0] + 360.0]))
    values_extended = np.concatenate(
        [values[:, -1:], values, values[:, :1]],
        axis=1,
    )

    interpolator = RegularGridInterpolator(
        (lat, lon_extended),
        values_extended,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    points = np.column_stack([mesh_lat, np.mod(mesh_lon, 360.0)])
    return interpolator(points)


# =============================================================================
# Metrics
# =============================================================================

def mean_pc_map(scores, rows, pc_index):
    """NaN-safe mean PC-score map over selected PC-score rows."""
    n_nodes = scores.shape[1]
    score_sum = np.zeros(n_nodes, dtype=np.float64)
    valid_count = np.zeros(n_nodes, dtype=np.int64)

    for row in rows:
        values = np.asarray(scores[row, :, pc_index], dtype=np.float64)
        valid = np.isfinite(values)
        score_sum[valid] += values[valid]
        valid_count[valid] += 1

    output = np.full(n_nodes, np.nan)
    valid = valid_count > 0
    output[valid] = score_sum[valid] / valid_count[valid]

    return output


def area_weighted_correlation(pc_scores, ocean_field, lat):
    """
    Pearson spatial correlation over valid ocean mesh nodes.
    Weights are cos(latitude), proportional to surface area.
    """
    valid = (
        np.isfinite(pc_scores)
        & np.isfinite(ocean_field)
        & np.isfinite(lat)
    )

    x = pc_scores[valid]
    y = ocean_field[valid]
    weights = np.cos(np.deg2rad(lat[valid]))

    if len(x) < 3:
        return np.nan, np.nan, int(len(x))

    x_mean = np.average(x, weights=weights)
    y_mean = np.average(y, weights=weights)

    covariance = np.average((x - x_mean) * (y - y_mean), weights=weights)
    x_std = np.sqrt(np.average((x - x_mean) ** 2, weights=weights))
    y_std = np.sqrt(np.average((y - y_mean) ** 2, weights=weights))

    if x_std == 0 or y_std == 0:
        return np.nan, np.nan, int(len(x))

    r = covariance / (x_std * y_std)
    return float(r), float(r ** 2), int(len(x))


# =============================================================================
# Plots
# =============================================================================

def plot_mesh_map(
    values,
    mesh_lat,
    mesh_lon,
    title,
    colorbar_label,
    out_path,
    add_world_map=False,
):
    vmax = np.nanpercentile(np.abs(values), 99)
    vmax = max(vmax, 1e-8)

    if add_world_map:
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_global()

        ax.add_feature(
            cfeature.LAND.with_scale("110m"),
            facecolor="lightgray",
            edgecolor="none",
            alpha=0.8,
            zorder=0,
        )
        ax.add_feature(
            cfeature.COASTLINE.with_scale("110m"),
            linewidth=0.6,
            edgecolor="black",
            zorder=3,
        )

        scatter = ax.scatter(
            mesh_lon,
            mesh_lat,
            c=values,
            s=2,
            linewidths=0,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

        gridlines = ax.gridlines(
            draw_labels=True,
            linewidth=0.35,
            color="gray",
            alpha=0.45,
            linestyle="--",
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.xlabel_style = {"size": 15}
        gridlines.ylabel_style = {"size": 15}

    else:
        fig, ax = plt.subplots(figsize=(15, 7))

        scatter = ax.scatter(
            mesh_lon,
            mesh_lat,
            c=values,
            s=2,
            linewidths=0,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )

        ax.set(
            xlim=(-180, 180),
            ylim=(-90, 90),
            xlabel="Longitude",
            ylabel="Latitude",
        )
        ax.set_xlabel("Longitude", fontsize=18)
        ax.set_ylabel("Latitude", fontsize=18)
        ax.tick_params(axis="both", labelsize=15)

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        pad=0.02,
        shrink=0.9,
    )
    colorbar.set_label(colorbar_label, fontsize=18, labelpad=14)
    colorbar.ax.tick_params(labelsize=15)

    ax.set_title(title, fontsize=24, pad=16)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_native_glorys_map(field, lat_name, lon_name, title, out_path):
    lon = ((field[lon_name].values + 180) % 360) - 180
    lat = field[lat_name].values
    order = np.argsort(lon)
    lon = lon[order]
    values = field.values[:, order]

    vmax = np.nanpercentile(np.abs(values), 99)
    vmax = max(vmax, 1e-8)

    fig, ax = plt.subplots(figsize=(13, 6))
    image = ax.pcolormesh(
        lon,
        lat,
        values,
        shading="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(
        image,
        ax=ax,
        label="Surface potential-temperature anomaly (°C)",
    )
    ax.set(
        xlim=(-180, 180),
        ylim=(-80, 90),
        xlabel="Longitude",
        ylabel="Latitude",
        title=title,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cache_path = os.path.join(OUT_DIR, CACHE_FILENAME)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reload cached annual PC2/GLORYS fields and remake plots only.",
        )
    
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"No cached plotting data found at {cache_path}. "
                "Run the full analysis once first."
            )

        with np.load(cache_path, allow_pickle=False) as cached:
            mesh_lat = cached["mesh_lat"]
            mesh_lon = cached["mesh_lon"]
            annual_pc2_ocean_only = cached["annual_pc2_ocean_only"]
            annual_anomaly_on_mesh = cached["annual_anomaly_on_mesh"]

            annual_anomaly = xr.DataArray(
                cached["annual_anomaly_grid"],
                dims=("lat", "lon"),
                coords={
                    "lat": cached["glorys_lat"],
                    "lon": cached["glorys_lon"],
                },
            )

        plot_mesh_map(
            annual_pc2_ocean_only,
            mesh_lat,
            mesh_lon,
            "2021 annual-mean PC2 activation over ocean only",
            "Sign-flipped PC2 score",
            os.path.join(
                OUT_DIR,
                "pc2_annual_mean_ocean_only_2021.png",
            ),
            add_world_map=True,
        )

        plot_native_glorys_map(
            annual_anomaly,
            "lat",
            "lon",
            "2021 GLORYS annual surface temperature anomaly",
            os.path.join(OUT_DIR, "glorys_thetao_zonal_anomaly_2021.png"),
        )

        plot_mesh_map(
            annual_anomaly_on_mesh,
            mesh_lat,
            mesh_lon,
            "2021 GLORYS annual surface temperature anomaly on GraphCast mesh",
            "Surface potential-temperature anomaly (°C)",
            os.path.join(OUT_DIR, "glorys_thetao_anomaly_on_mesh_2021.png"),
            add_world_map=True,
        )

        print(f"Replotted cached results from: {cache_path}")
        return

    scores, source_files, timestamps = load_pc_score_metadata(
        PC_SCORES_PATH,
        PC_SCORES_FILES_LIST,
    )

    print(f"PC-score array shape: {scores.shape}")
    print(f"Using PC{PC_INDEX + 1}")

    if scores.shape[2] <= PC_INDEX:
        raise ValueError(
            f"Only {scores.shape[2]} PCs available; PC index {PC_INDEX} is invalid."
        )

    rows_by_month = audit_timestamps(timestamps, source_files, OUT_DIR)

    mesh_lat, mesh_lon = get_mesh_latlon(MESH_SPLITS)
    if scores.shape[1] != len(mesh_lat):
        raise ValueError(
            f"PC scores contain {scores.shape[1]} nodes, but mesh level "
            f"{MESH_SPLITS} has {len(mesh_lat)} nodes."
        )

    glorys_files = sorted(glob(GLORYS_GLOB))
    if not glorys_files:
        raise FileNotFoundError(f"No GLORYS files found: {GLORYS_GLOB}")

    monthly_ocean = {}
    lat_name = lon_name = None

    for path in glorys_files:
        month_match = re.search(r"_2021_(\d{2})\.nc$", os.path.basename(path))
        if month_match is None:
            print(f"Skipping file with unrecognised month: {path}")
            continue

        month = int(month_match.group(1))
        field, lat_name, lon_name = load_glorys_surface_thetao(path)

        # GLORYS uses potential temperature in degrees Celsius.
        monthly_ocean[month] = field
        print(f"Loaded GLORYS month {month:02d}: {path}")

    missing_glorys_months = sorted(set(range(1, 13)) - set(monthly_ocean))
    if missing_glorys_months:
        message = f"Missing GLORYS months: {missing_glorys_months}"
        if REQUIRE_ALL_GLORYS_MONTHS:
            raise FileNotFoundError(message)
        print(f"WARNING: {message}")

    # Weighted annual mean from monthly means, accounting for unequal month lengths.
    annual_sum = None
    annual_weight = None

    metric_rows = []

    for month, temperature in sorted(monthly_ocean.items()):
        days = datetime(YEAR, month % 12 + 1, 1).replace(
            year=YEAR + (month == 12)
        ) - datetime(YEAR, month, 1)
        days = days.days

        values = temperature.values.astype(np.float64)
        valid = np.isfinite(values)

        if annual_sum is None:
            annual_sum = np.zeros_like(values, dtype=np.float64)
            annual_weight = np.zeros_like(values, dtype=np.float64)

        annual_sum[valid] += values[valid] * days
        annual_weight[valid] += days

        # Monthly anomaly removes zonal-mean temperature at each latitude.
        temperature_anomaly = temperature - temperature.mean(
            dim=lon_name,
            skipna=True,
        )

        pc_map = PC_SIGN * mean_pc_map(scores, rows_by_month[month], PC_INDEX)
        raw_on_mesh = interpolate_to_mesh(
            temperature, lat_name, lon_name, mesh_lat, mesh_lon
        )
        anomaly_on_mesh = interpolate_to_mesh(
            temperature_anomaly, lat_name, lon_name, mesh_lat, mesh_lon
        )

        r_raw, r2_raw, n_raw = area_weighted_correlation(
            pc_map, raw_on_mesh, mesh_lat
        )
        r_anom, r2_anom, n_anom = area_weighted_correlation(
            pc_map, anomaly_on_mesh, mesh_lat
        )

        metric_rows.extend([
            {
                "period": f"{YEAR}-{month:02d}",
                "comparison": "PC2_vs_raw_surface_thetao",
                "r_area_weighted": r_raw,
                "r_squared": r2_raw,
                "n_ocean_mesh_nodes": n_raw,
                "n_pc_timesteps": len(rows_by_month[month]),
            },
            {
                "period": f"{YEAR}-{month:02d}",
                "comparison": "PC2_vs_zonal_thetao_anomaly",
                "r_area_weighted": r_anom,
                "r_squared": r2_anom,
                "n_ocean_mesh_nodes": n_anom,
                "n_pc_timesteps": len(rows_by_month[month]),
            },
        ])

    annual_temperature = monthly_ocean[next(iter(monthly_ocean))].copy(
        data=annual_sum / annual_weight
    )
    annual_anomaly = annual_temperature - annual_temperature.mean(
        dim=lon_name,
        skipna=True,
    )

    all_rows = list(range(scores.shape[0]))
    annual_pc2 = PC_SIGN * mean_pc_map(scores, all_rows, PC_INDEX)

    annual_raw_on_mesh = interpolate_to_mesh(
        annual_temperature, lat_name, lon_name, mesh_lat, mesh_lon
    )
    annual_anomaly_on_mesh = interpolate_to_mesh(
        annual_anomaly, lat_name, lon_name, mesh_lat, mesh_lon
    )

    # Ocean-only PC2 for plotting. The correlation already applies this
    # equivalent finite-value mask internally.
    ocean_mesh_mask = np.isfinite(annual_raw_on_mesh)

    annual_pc2_ocean_only = np.where(
        ocean_mesh_mask,
        annual_pc2,
        np.nan,
    )

    np.savez_compressed(
        cache_path,
        mesh_lat=mesh_lat,
        mesh_lon=mesh_lon,
        annual_pc2_ocean_only=annual_pc2_ocean_only,
        annual_anomaly_on_mesh=annual_anomaly_on_mesh,
        annual_anomaly_grid=annual_anomaly.values.astype(np.float32),
        glorys_lat=annual_anomaly[lat_name].values,
        glorys_lon=annual_anomaly[lon_name].values,
    )

    print(f"Saved reusable plotting cache: {cache_path}")

    r_raw, r2_raw, n_raw = area_weighted_correlation(
        annual_pc2_ocean_only, annual_raw_on_mesh, mesh_lat
    )
    r_anom, r2_anom, n_anom = area_weighted_correlation(
        annual_pc2_ocean_only, annual_anomaly_on_mesh, mesh_lat
    )

    metric_rows.extend([
        {
            "period": f"{YEAR}_annual",
            "comparison": "PC2_vs_raw_surface_thetao",
            "r_area_weighted": r_raw,
            "r_squared": r2_raw,
            "n_ocean_mesh_nodes": n_raw,
            "n_pc_timesteps": len(all_rows),
        },
        {
            "period": f"{YEAR}_annual",
            "comparison": "PC2_vs_zonal_thetao_anomaly",
            "r_area_weighted": r_anom,
            "r_squared": r2_anom,
            "n_ocean_mesh_nodes": n_anom,
            "n_pc_timesteps": len(all_rows),
        },
    ])

    metrics_path = os.path.join(OUT_DIR, "pc2_glorys_thetao_metrics_2021.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_rows[0].keys())
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"\nAnnual PC2 vs raw surface thetao: r = {r_raw:.4f}")
    print(f"Annual PC2 vs zonal thetao anomaly: r = {r_anom:.4f}")
    print(f"Saved metrics: {metrics_path}")

    plot_mesh_map(
        annual_pc2_ocean_only,
        mesh_lat,
        mesh_lon,
        "2021 annual-mean PC2 activation over ocean only",
        "PC2 score (flipped sign)",
        os.path.join(OUT_DIR, "pc2_annual_mean_2021_ocean_only.png"),
        add_world_map=True,
    )

    # plot_native_glorys_map(
    #     annual_anomaly,
    #     lat_name,
    #     lon_name,
    #     "2021 GLORYS annual surface temperature anomaly",
    #     os.path.join(OUT_DIR, "glorys_thetao_zonal_anomaly_2021.png"),
    # )

    # plot_mesh_map(
    #     annual_anomaly_on_mesh,
    #     mesh_lat,
    #     mesh_lon,
    #     "2021 GLORYS annual surface temperature anomaly on GraphCast mesh",
    #     "Surface potential-temperature anomaly (°C)",
    #     os.path.join(OUT_DIR, "glorys_thetao_anomaly_on_mesh_2021.png"),
    # )


if __name__ == "__main__":
    main()