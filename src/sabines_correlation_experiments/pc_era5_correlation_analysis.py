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
pca_components_path = "/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021.npy"
pca_mean_path = "/share/prj-4d/graphcast_shared/data/pca_components/pca_mean_2021.npy"

era5_vars = [
    "geopotential", "specific_humidity", "temperature", "u_component_of_wind",
    "v_component_of_wind", "vertical_velocity", "2m_temperature",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "mean_sea_level_pressure", "total_precipitation_6hr",
    "toa_incident_solar_radiation", "geopotential_at_surface", "land_sea_mask"
]

# How many top PCs to analyze
n_pcs_to_check = 20
# How many top ERA5 variables to keep per PC
top_k = 5

# Coarse grid for aggregation
LAT_RES = 2.5
LON_RES = 2.5
lat_bins = np.arange(-90, 90.1, LAT_RES)
lon_bins = np.arange(0, 360.1, LON_RES)

results_dir = "plots/sabines_experiments"
os.makedirs(results_dir, exist_ok=True)
yearly_summary_path = os.path.join(results_dir, "pc_era5_yearly_top_variables.json")


# -------------------------
# Utilities
# -------------------------
def get_mesh_latlon(splits=6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    vertices = meshes[6].vertices
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    return lat, lon

def get_era5_latlon(ds):
    era5_lat = ds['lat'].values
    era5_lon = ds['lon'].values
    lon2d, lat2d = np.meshgrid(era5_lon, era5_lat)
    return lat2d.ravel(), lon2d.ravel()

def aggregate_to_grid(values, lats, lons, lat_bins, lon_bins):
    values = np.asarray(values, dtype=np.float32).ravel()
    lats = np.asarray(lats, dtype=np.float32).ravel()
    lons = np.asarray(lons, dtype=np.float32).ravel()
    
    grid = np.full((len(lat_bins)-1, len(lon_bins)-1), np.nan, dtype=np.float32)
    sums = np.zeros_like(grid, dtype=np.float64)
    counts = np.zeros_like(grid, dtype=np.int32)
    
    lat_idx = np.digitize(lats, lat_bins) - 1
    lon_idx = np.digitize(lons % 360, lon_bins) - 1
    
    for v, i, j in zip(values, lat_idx, lon_idx):
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            if np.isfinite(v):
                sums[i, j] += float(v)
                counts[i, j] += 1
    
    valid = counts > 0
    grid[valid] = (sums[valid] / counts[valid]).astype(np.float32)
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
# Main
# -------------------------
def main():
    print("Loading PCA components and mean...")
    pca_components = np.load(pca_components_path)  # (n_pcs, n_features)
    pca_mean = np.load(pca_mean_path)  # (n_features,)
    print(f" pca_components.shape: {pca_components.shape}")
    print(f" pca_mean.shape: {pca_mean.shape}")

    mesh_lat, mesh_lon = get_mesh_latlon(splits=6)
    n_mesh_nodes = mesh_lat.size

    activation_files = sorted([os.path.join(activations_dir, f) for f in os.listdir(activations_dir) if f.endswith(".npy")])
    centers = [os.path.basename(f).split("_t")[-1].replace(".npy", "") for f in activation_files]

    # Per-PC aggregation: pc_idx -> var_name -> {'sum_abs_r': float, 'count': int, 'wins': int}
    pc_stats = defaultdict(lambda: defaultdict(lambda: {'sum_abs_r': 0.0, 'count': 0, 'wins': 0}))
    pc_timestep_count = Counter()

    n_pcs = min(n_pcs_to_check, pca_components.shape[0])
    print(f"\nAnalyzing {n_pcs} PCs (out of {pca_components.shape[0]} available)\n")

    print(f"Loading activations and a matching era5 window per timestep...")
    for act_file, center_str in zip(activation_files, centers):
        print(f"  Processing {center_str}")

        # Load GC activations
        activations = np.load(act_file)
        if activations.dtype == np.dtype("|V2"):
            activations = activations.view(np.float16)
        activations = np.asarray(activations, dtype=np.float32)
        activations = np.squeeze(activations)
        
        if activations.ndim != 2:
            print(f"  Skipping: unexpected shape {activations.shape}")
            del activations
            gc.collect()
            continue

        n_nodes_act, n_feat_act = activations.shape
        if n_nodes_act != n_mesh_nodes:
            print(f"  Warning: activation nodes {n_nodes_act} != mesh nodes {n_mesh_nodes}")

        # Load ERA5 for this timestep
        ds = three_step_window(era5_dir, center_str)
        if ds is None:
            print(f"  Skipping: missing ERA5 data")
            del activations
            gc.collect()
            continue
        era5_lat, era5_lon = get_era5_latlon(ds)
        print(f"  Done loading ERA5 data with variables: {list(ds.data_vars.keys())}")

        # Build ERA5 coarse grids
        era5_features = []
        for var in era5_vars:
            if var not in ds:
                continue
            arr = ds[var].values
            if arr.dtype == np.dtype("|V2"):
                arr = arr.view(np.float16)
            arr = np.asarray(arr, dtype=np.float32)
            arr = np.squeeze(arr)
            
            if arr.ndim == 4:  # (window, level, lat, lon)
                arr_mean = np.nanmean(arr, axis=0)
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
        print(f"  Done processing ERA5 features: {[v[0] for v in era5_features]} and aggregating to grid")

        # For each PC: project activations onto PC, then correlate with ERA5
        for pc_idx in range(n_pcs):
            pc_component = pca_components[pc_idx]  # (n_features,)
            # Project activations onto this PC: (n_nodes, n_features) @ (n_features,) -> (n_nodes,)
            pc_scores = activations @ pc_component  # spatial map of this PC at this timestep
            pc_grid = aggregate_to_grid(pc_scores, mesh_lat, mesh_lon, lat_bins, lon_bins)
            print(f"  Done projecting activations onto PC_{pc_idx+1} and aggregating to grid")

            # Correlate with each ERA5 variable
            var_rs = {}
            for var_name, era5_grid in era5_features:
                r = safe_corr(pc_grid.ravel(), era5_grid.ravel())
                if r is None:
                    continue
                var_rs[var_name] = r
                # Update running stats
                s = pc_stats[pc_idx][var_name]
                s['sum_abs_r'] += abs(r)
                s['count'] += 1

            # Record winner for this PC at this timestep
            if var_rs:
                winner_name, winner_r = max(var_rs.items(), key=lambda kv: abs(kv[1]))
                pc_stats[pc_idx][winner_name]['wins'] += 1
            print(f"For timestep {center_str} and PC_{pc_idx+1} - top variable = {winner_name} with r={winner_r:.3f}" if var_rs else f"    for timestep {center_str}: PC_{pc_idx+1} - no valid correlations")

        pc_timestep_count[pc_idx] += 1

        # Free memory
        del activations
        del era5_features
        del ds
        gc.collect()

    # -------------------------
    # Post-process: for each PC, rank ERA5 variables by mean |r|
    # -------------------------
    summary = {}
    for pc_idx in range(n_pcs):
        var_list = []
        for var_name, stats in pc_stats[pc_idx].items():
            if stats['count'] > 0:
                mean_abs_r = stats['sum_abs_r'] / stats['count']
            else:
                mean_abs_r = 0.0
            var_list.append({
                'variable': var_name,
                'mean_abs_r': mean_abs_r,
                'wins': stats['wins'],
                'count': stats['count']
            })
        
        # Sort by mean |r| descending
        var_list_sorted = sorted(var_list, key=lambda x: (-x['mean_abs_r'], -x['wins']))
        top_vars = var_list_sorted[:top_k]
        
        summary[f"PC_{pc_idx+1}"] = {
            'top_variables': top_vars,
            'total_timesteps': pc_timestep_count[pc_idx]
        }

    # Save JSON summary
    with open(yearly_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n=== Yearly PC -> ERA5 Summary (Top Variables per PC) ===\n")
    for pc_key, info in sorted(summary.items(), key=lambda kv: int(kv[0].split('_')[1])):
        print(f"{pc_key} (timesteps: {info['total_timesteps']})")
        for i, var_info in enumerate(info['top_variables'], 1):
            print(f"  {i}. {var_info['variable']}: mean_|r|={var_info['mean_abs_r']:.3f}, wins={var_info['wins']}, count={var_info['count']}")
        print()

    print(f"\nSaved summary to {yearly_summary_path}")

if __name__ == "__main__":
    main()