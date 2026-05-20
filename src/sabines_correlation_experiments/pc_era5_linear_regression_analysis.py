#!/usr/bin/env python3
import os
import json
import gc
import numpy as np
import xarray as xr
from collections import defaultdict, Counter
from graphcast import icosahedral_mesh

# --- CONFIGURATION ---
era5_dir = "/share/prj-4d/graphcast_shared/data/era5_daily_nc"
activations_dir = "/share/prj-4d/graphcast_shared/data/graphcast_activation_2021"

era5_vars = [
    "geopotential", "specific_humidity", "temperature", "u_component_of_wind",
    "v_component_of_wind", "vertical_velocity", "2m_temperature",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "total_precipitation_6hr",
    "toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask"
]

# How many PCs to analyze per timestep (adjustable)
n_pcs_to_check = 20

# Coarse grid for aggregation (adjust if you want coarser/finer)
LAT_RES = 2.5
LON_RES = 2.5
lat_bins = np.arange(-90, 90.1, LAT_RES)
lon_bins = np.arange(0, 360.1, LON_RES)

# Output
results_dir = "plots/sabines_experiments"
os.makedirs(results_dir, exist_ok=True)
monthly_summary_path = os.path.join(results_dir, "pc_era5_monthly_summary.json")


# -------------------------
# Utilities
# -------------------------
def get_mesh_latlon(splits=6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    vertices = meshes[6].vertices
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    return lat, lon  # length n_nodes

def get_era5_latlon(ds):
    era5_lat = ds['lat'].values
    era5_lon = ds['lon'].values
    lon2d, lat2d = np.meshgrid(era5_lon, era5_lat)
    return lat2d.ravel(), lon2d.ravel()

def aggregate_to_grid(values, lats, lons, lat_bins, lon_bins):
    # values, lats, lons are 1D arrays
    grid = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan)
    sums = np.zeros_like(grid)
    counts = np.zeros_like(grid, dtype=np.int32)
    lat_idx = np.digitize(lats, lat_bins) - 1
    lon_idx = np.digitize(lons % 360, lon_bins) - 1
    for v, i, j in zip(values, lat_idx, lon_idx):
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            if not np.isnan(v):
                sums[i, j] += v
                counts[i, j] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        grid = sums / counts
    return grid

def safe_corr(a, b, min_points=10):
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < min_points:
        return None
    a2 = a[mask]; b2 = b[mask]
    if np.nanstd(a2) == 0 or np.nanstd(b2) == 0:
        return None
    return float(np.corrcoef(a2, b2)[0, 1])

def _open_and_trim(path: str) -> xr.Dataset:
    ds = xr.open_dataset(path)
    if "time" in ds.dims and ds.sizes["time"] > 4:
        ds = ds.isel(time=slice(0, 4))
    return ds

def three_step_window(data_dir: str, center_time: str) -> xr.Dataset | None:
    t0 = np.datetime64(center_time)
    t_minus = t0 - np.timedelta64(6, "h")
    t_plus  = t0 + np.timedelta64(6, "h")
    needed_days = sorted({
        np.datetime64(t_minus, "D"),
        np.datetime64(t0, "D"),
        np.datetime64(t_plus, "D"),
    })
    file_paths = [os.path.join(data_dir, f"era5_{str(d)[:10]}.nc") for d in needed_days]
    if any(not os.path.exists(p) for p in file_paths):
        return None
    daily = [_open_and_trim(p) for p in file_paths]
    var_time = [v for v, da in daily[0].data_vars.items() if "time" in da.dims]
    var_static = [v for v, da in daily[0].data_vars.items() if "time" not in da.dims]
    ds_time = xr.concat([d[var_time] for d in daily], dim="time").sortby("time")
    ds_static = daily[0][var_static]
    ds = xr.merge([ds_time, ds_static])
    target_times = np.array([t_minus, t0, t_plus], dtype=ds.time.dtype)
    if not all(t in ds.time.values for t in target_times):
        print(f"Missing needed times for center {center_time}: expected {target_times}, got {ds.time.values}")
        return None
    ds = ds.sel(time=target_times)
    ds_new = ds.copy()
    for v in ds_new.data_vars:
        if "time" in ds_new[v].dims:
            ds_new[v] = ds_new[v].expand_dims("batch")
    for c in ds.coords:
        if "time" in ds[c].dims:
            ds_new = ds_new.assign_coords({c: ds[c].expand_dims("batch")})
    time_orig = ds["time"]
    t_ref = time_orig.values[0]
    time_delta = time_orig - t_ref
    ds_new = ds_new.assign_coords(time=time_delta)
    ds_new = ds_new.assign_coords(datetime=("time", time_orig.values))
    ds_new = ds_new.assign_coords({"datetime": ds_new["datetime"].expand_dims("batch")})
    return ds_new


# -------------------------
# Main processing loop
# -------------------------
def main():
    # Prepare mesh lat/lon once
    mesh_lat, mesh_lon = get_mesh_latlon(splits=6)
    n_mesh_nodes = mesh_lat.size

    # Activation files / centers
    activation_files = sorted([os.path.join(activations_dir, f) for f in os.listdir(activations_dir) if f.endswith(".npy")])
    centers = [os.path.basename(f).split("_t")[-1].replace(".npy", "") for f in activation_files]

    # Running monthly statistics:
    # monthly_stats[month][pc_idx][var] = {'sum_abs_r': float, 'count': int, 'wins': int}
    monthly_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'sum_abs_r':0.0, 'count':0, 'wins':0})))
    # Also keep number of timesteps processed per month
    month_timestep_count = Counter()

    for act_file, center_str in zip(activation_files, centers):
        month = center_str[:7]  # "YYYY-MM" assuming center_str like "2021-02-01T06"
        print(f"\nProcessing {center_str} (month {month})")

        # Load activations
        activations = np.load(act_file)
        if activations.dtype == np.dtype("|V2"):
            activations = activations.view(np.float16)
        activations = np.asarray(activations, dtype=np.float32)
        activations = np.squeeze(activations)
        print(" activations.shape:", activations.shape)
        if activations.ndim != 2:
            print(" Unexpected activation shape; skipping timestep.")
            del activations
            gc.collect()
            continue
        n_nodes_act, n_pcs_found = activations.shape
        if n_nodes_act != n_mesh_nodes:
            print(f" Warning: activations node count {n_nodes_act} != mesh nodes {n_mesh_nodes}")
            # proceed but mapping will be inconsistent unless user has matching mesh
        # Limit PCs to check
        n_pcs = min(n_pcs_to_check, n_pcs_found)
        if n_pcs_found > n_pcs_to_check:
            print(f" Found {n_pcs_found} PCs; analyzing first {n_pcs}")

        # Aggregate activation maps (per PC) to coarse grid
        activation_grids = []
        for pc_idx in range(n_pcs):
            pc_map = activations[:, pc_idx]
            activation_grid = aggregate_to_grid(pc_map, mesh_lat, mesh_lon, lat_bins, lon_bins)
            activation_grids.append(activation_grid)
        activation_grids = np.stack(activation_grids, axis=0)  # (n_pcs, nlat, nlon)

        # Load ERA5 window
        ds = three_step_window(era5_dir, center_str)
        if ds is None:
            print(" Missing ERA5 window; skipping timestep.")
            del activations, activation_grids
            gc.collect()
            continue
        era5_lat, era5_lon = get_era5_latlon(ds)

        # Build ERA5 coarse grids for all variables (treat levels individually)
        era5_features = []  # list of (name, grid)
        for var in era5_vars:
            if var not in ds:
                continue
            arr = ds[var].values
            arr = np.squeeze(arr)
            if arr.ndim == 4:  # (window, level, lat, lon)
                arr_mean = np.nanmean(arr, axis=0)  # (level, lat, lon)
                for lev_idx in range(arr_mean.shape[0]):
                    feat = arr_mean[lev_idx].ravel()
                    era5_grid = aggregate_to_grid(feat, era5_lat, era5_lon, lat_bins, lon_bins)
                    era5_features.append((f"{var}_lev{lev_idx}", era5_grid))
            elif arr.ndim == 3:  # (window, lat, lon)
                arr_mean = np.nanmean(arr, axis=0)
                feat = arr_mean.ravel()
                era5_grid = aggregate_to_grid(feat, era5_lat, era5_lon, lat_bins, lon_bins)
                era5_features.append((var, era5_grid))
            elif arr.ndim == 2:  # (lat, lon)
                feat = arr.ravel()
                era5_grid = aggregate_to_grid(feat, era5_lat, era5_lon, lat_bins, lon_bins)
                era5_features.append((var, era5_grid))
            else:
                print(f" Unexpected shape for {var}: {arr.shape}")

        # For each PC compute correlations across ERA5 features
        for pc_idx in range(n_pcs):
            pc_grid = activation_grids[pc_idx]
            # compute r for all features
            var_rs = {}
            for name, era5_grid in era5_features:
                r = safe_corr(pc_grid.ravel(), era5_grid.ravel())
                if r is None:
                    continue
                var_rs[name] = r
                # Running mean abs r and count
                s = monthly_stats[month][pc_idx][name]
                s['sum_abs_r'] += abs(r)
                s['count'] += 1
            # Determine winning variable for this PC at this timestep (highest abs r)
            if var_rs:
                winner = max(var_rs.items(), key=lambda kv: abs(kv[1]))  # (name, r)
                win_name, win_r = winner
                monthly_stats[month][pc_idx][win_name]['wins'] += 1

        # Bookkeeping: one more timestep processed in this month
        month_timestep_count[month] += 1

        # Free memory for this timestep
        del activations
        del activation_grids
        del era5_features
        del ds
        gc.collect()

    # -------------------------
    # Post-process monthly stats -> summary structure
    # -------------------------
    summary = {}
    for month, pcs_dict in monthly_stats.items():
        summary[month] = {}
        for pc_idx, var_dict in pcs_dict.items():
            # For each variable compute mean_abs_r and wins
            rows = []
            for varname, vals in var_dict.items():
                if vals['count'] > 0:
                    mean_abs = vals['sum_abs_r'] / vals['count']
                else:
                    mean_abs = None
                rows.append({'var': varname, 'mean_abs_r': mean_abs, 'wins': vals['wins'], 'count': vals['count']})
            # sort rows by mean_abs_r (desc), fall back to wins
            rows_sorted = sorted([r for r in rows if r['mean_abs_r'] is not None], key=lambda x: (-x['mean_abs_r'], -x['wins']))
            top3 = rows_sorted[:3]
            summary[month][f"pc_{pc_idx+1}"] = {'top3': top3, 'timesteps': month_timestep_count.get(month, 0)}
    # Save summary JSON
    with open(monthly_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print a concise summary
    print("\n=== Monthly PC -> ERA5 summary (top3 by mean |r|) ===")
    for month, pcs in sorted(summary.items()):
        print(f"\nMonth: {month}  (timesteps: {pcs[next(iter(pcs))]['timesteps'] if pcs else 0})")
        for pc_key, info in sorted(pcs.items(), key=lambda kv: int(kv[0].split('_')[1])):
            top3 = info['top3']
            if not top3:
                print(f" {pc_key}: no valid correlations")
                continue
            print(f" {pc_key}:")
            for r in top3:
                print(f"   - {r['var']}: mean_abs_r={r['mean_abs_r']:.3f}, wins={r['wins']}, count={r['count']}")

    print(f"\nSaved monthly summary to {monthly_summary_path}")

if __name__ == "__main__":
    main()