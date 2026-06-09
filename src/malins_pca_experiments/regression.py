import os
import re
from glob import glob
from functools import lru_cache

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from graphcast import icosahedral_mesh


# =====================
# CONFIG
# =====================

PC_SCORES_PATH = "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/pc_scores_2021_per_timestep.npy"
TIMESTEP_FILES_TXT = "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/pc_scores_2021_per_timestep_files.txt"

ACTS_DIR = "/share/prj-4d/graphcast_shared/data/graphcast_activation_2021"

ERA5_DAILY_DIR = "/share/prj-4d/graphcast_shared/data/era5_daily_nc"
FEATURE_COUNTS_RAW = [512]


#NODES_PER_TIMESTEP = 500
#RANDOM_SEED = 42

node_hierarchy_level = 4

PC_COUNTS = [5, 10, 25, 50, 100, 200, 400]
#PC_COUNTS = [5, 10]

TARGETS_OLD = [
    {"name": "z500", "var": "geopotential", "level": 500},
    {"name": "t850", "var": "temperature", "level": 850},
    {"name": "q850", "var": "specific_humidity", "level": 850},
    {"name": "u850", "var": "u_component_of_wind", "level": 850},
    {"name": "v850", "var": "v_component_of_wind", "level": 850},
    {"name": "msl", "var": "mean_sea_level_pressure", "level": None},
    {"name": "2t", "var": "2m_temperature", "level": None},
]

TARGETS = [
    # Surface variables
    {"name": "2t", "var": "2m_temperature", "level": None},
    {"name": "10u", "var": "10m_u_component_of_wind", "level": None},
    {"name": "10v", "var": "10m_v_component_of_wind", "level": None},
    {"name": "msl", "var": "mean_sea_level_pressure", "level": None},
    {"name": "tp", "var": "total_precipitation", "level": None},

    # Atmospheric variables at bold pressure levels
    {"name": "t50", "var": "temperature", "level": 50},
    {"name": "t250", "var": "temperature", "level": 250},
    {"name": "t500", "var": "temperature", "level": 500},
    {"name": "t600", "var": "temperature", "level": 600},
    {"name": "t700", "var": "temperature", "level": 700},
    {"name": "t850", "var": "temperature", "level": 850},
    {"name": "t1000", "var": "temperature", "level": 1000},

    {"name": "u50", "var": "u_component_of_wind", "level": 50},
    {"name": "u250", "var": "u_component_of_wind", "level": 250},
    {"name": "u500", "var": "u_component_of_wind", "level": 500},
    {"name": "u600", "var": "u_component_of_wind", "level": 600},
    {"name": "u700", "var": "u_component_of_wind", "level": 700},
    {"name": "u850", "var": "u_component_of_wind", "level": 850},
    {"name": "u1000", "var": "u_component_of_wind", "level": 1000},

    {"name": "v50", "var": "v_component_of_wind", "level": 50},
    {"name": "v250", "var": "v_component_of_wind", "level": 250},
    {"name": "v500", "var": "v_component_of_wind", "level": 500},
    {"name": "v600", "var": "v_component_of_wind", "level": 600},
    {"name": "v700", "var": "v_component_of_wind", "level": 700},
    {"name": "v850", "var": "v_component_of_wind", "level": 850},
    {"name": "v1000", "var": "v_component_of_wind", "level": 1000},

    {"name": "z50", "var": "geopotential", "level": 50},
    {"name": "z250", "var": "geopotential", "level": 250},
    {"name": "z500", "var": "geopotential", "level": 500},
    {"name": "z600", "var": "geopotential", "level": 600},
    {"name": "z700", "var": "geopotential", "level": 700},
    {"name": "z850", "var": "geopotential", "level": 850},
    {"name": "z1000", "var": "geopotential", "level": 1000},

    {"name": "q50", "var": "specific_humidity", "level": 50},
    {"name": "q250", "var": "specific_humidity", "level": 250},
    {"name": "q500", "var": "specific_humidity", "level": 500},
    {"name": "q600", "var": "specific_humidity", "level": 600},
    {"name": "q700", "var": "specific_humidity", "level": 700},
    {"name": "q850", "var": "specific_humidity", "level": 850},
    {"name": "q1000", "var": "specific_humidity", "level": 1000},

    {"name": "w50", "var": "vertical_velocity", "level": 50},
    {"name": "w250", "var": "vertical_velocity", "level": 250},
    {"name": "w500", "var": "vertical_velocity", "level": 500},
    {"name": "w600", "var": "vertical_velocity", "level": 600},
    {"name": "w700", "var": "vertical_velocity", "level": 700},
    {"name": "w850", "var": "vertical_velocity", "level": 850},
    {"name": "w1000", "var": "vertical_velocity", "level": 1000},
]

DERIVED_TARGETS = [
    {"name": "windspeed850"},
    {"name": "rh850"},
    {"name": "thickness500_850"},
]


run_analysis_on = "direct"  #"drevived" or "direct"

score_values = "raw_activations" #"raw_activations" or "PCA"

OUT_DIR = f"plots/malins_experiments/2021_pc_physical_variable_regression/{run_analysis_on}_{score_values}targets"
os.makedirs(OUT_DIR, exist_ok=True)


# =====================
# HELPERS
# =====================

def parse_timestamp_from_path(path):
    fname = os.path.basename(path)
    m = re.search(r"t(\d{4})-(\d{2})-(\d{2})T(\d{2})", fname)
    if not m:
        raise ValueError(f"Could not parse timestamp from {fname}")
    y, mo, d, h = map(int, m.groups())
    return pd.Timestamp(y, mo, d, h)


def load_timestamps(files_txt):
    with open(files_txt, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    timestamps = pd.to_datetime([parse_timestamp_from_path(p) for p in files])
    return files, timestamps


def vertices_to_latlon(vertices):
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0])) % 360
    return lat, lon


def get_mesh_latlon(splits=6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    vertices = meshes[splits].vertices
    return vertices_to_latlon(vertices)


@lru_cache(maxsize=400)
def open_era5_day(date_str):
    path = os.path.join(ERA5_DAILY_DIR, f"era5_{date_str}.nc")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return xr.open_dataset(path)


def select_level(da, level):
    if level is None:
        return da

    for coord in ["level", "pressure_level", "isobaricInhPa"]:
        if coord in da.coords:
            return da.sel({coord: level}, method="nearest")

    raise ValueError(f"No pressure-level coord found for {da.name}")


def sample_all_targets_at_nodes(timestamp, targets, lat, lon, node_indices):
    date_str = pd.Timestamp(timestamp).strftime("%Y-%m-%d")
    ds = open_era5_day(date_str)

    node_lat = xr.DataArray(lat[node_indices], dims="sample")
    node_lon = xr.DataArray(lon[node_indices], dims="sample")

    out = {}

    for target in targets:
        da = ds[target["var"]]
        da = select_level(da, target["level"])

        if "time" in da.coords:
            da = da.sel(time=np.datetime64(timestamp), method="nearest")

        y = da.interp(
            lat=node_lat,
            lon=node_lon,
            method="nearest"
        ).values.astype(np.float32)

        out[target["name"]] = y

    return out


def sample_all_derived_targets_at_nodes(timestamp, targets, lat, lon, node_indices):
    """
    Sample ERA5 variables at mesh nodes and compute derived diagnostics.

    Supported derived target names:
        - windspeed850
        - windspeed500
        - thickness500_850
        - thickness500_1000
        - rh850

    Returns:
        dict: {target_name: np.ndarray [n_nodes]}
    """
    date_str = pd.Timestamp(timestamp).strftime("%Y-%m-%d")
    ds = open_era5_day(date_str)

    node_lat = xr.DataArray(lat[node_indices], dims="sample")
    node_lon = xr.DataArray(lon[node_indices], dims="sample")

    def get_var(var_name, level=None):
        da = ds[var_name]
        da = select_level(da, level)

        if "time" in da.coords:
            da = da.sel(time=np.datetime64(timestamp), method="nearest")

        values = da.interp(
            lat=node_lat,
            lon=node_lon,
            method="nearest",
        ).values.astype(np.float32)

        return values

    def relative_humidity_from_t_q_p(t_kelvin, q, pressure_pa):
        """
        Approximate RH from temperature, specific humidity, and pressure.

        t_kelvin: temperature in K
        q: specific humidity in kg/kg
        pressure_pa: pressure in Pa

        Returns RH in percent.
        """
        # vapor pressure from specific humidity
        epsilon = 0.622
        e = (q * pressure_pa) / (epsilon + (1.0 - epsilon) * q)

        # saturation vapor pressure over liquid water, Bolton-style approximation
        t_celsius = t_kelvin - 273.15
        e_s = 611.2 * np.exp((17.67 * t_celsius) / (t_celsius + 243.5))

        rh = 100.0 * e / e_s
        return np.clip(rh, 0.0, 150.0).astype(np.float32)

    out = {}

    for target in targets:
        name = target["name"]

        if name == "windspeed850":
            u = get_var("u_component_of_wind", 850)
            v = get_var("v_component_of_wind", 850)
            out[name] = np.sqrt(u**2 + v**2).astype(np.float32)

        elif name == "windspeed500":
            u = get_var("u_component_of_wind", 500)
            v = get_var("v_component_of_wind", 500)
            out[name] = np.sqrt(u**2 + v**2).astype(np.float32)

        elif name == "thickness500_850":
            z500 = get_var("geopotential", 500)
            z850 = get_var("geopotential", 850)
            out[name] = (z500 - z850).astype(np.float32)

        elif name == "thickness500_1000":
            z500 = get_var("geopotential", 500)
            z1000 = get_var("geopotential", 1000)
            out[name] = (z500 - z1000).astype(np.float32)

        elif name == "rh850":
            t = get_var("temperature", 850)
            q = get_var("specific_humidity", 850)
            out[name] = relative_humidity_from_t_q_p(
                t_kelvin=t,
                q=q,
                pressure_pa=85000.0,
            )

        else:
            raise ValueError(f"Unknown derived target: {name}")

    return out

def get_coarse_mesh_node_indices(fine_splits=6, coarse_splits=4, decimals=8):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=fine_splits
    )

    fine_vertices = meshes[fine_splits].vertices
    coarse_vertices = meshes[coarse_splits].vertices

    fine_keys = {
        tuple(np.round(v, decimals)): i
        for i, v in enumerate(fine_vertices)
    }

    coarse_indices = []
    missing = 0

    for v in coarse_vertices:
        key = tuple(np.round(v, decimals))
        if key in fine_keys:
            coarse_indices.append(fine_keys[key])
        else:
            missing += 1

    if missing > 0:
        raise ValueError(f"Could not match {missing} coarse vertices to fine mesh")

    return np.array(coarse_indices, dtype=int)



def load_activations(path):
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    return x.astype(np.float32)


def corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return np.corrcoef(a[mask], b[mask])[0, 1]





# =====================
# MAIN
# =====================

def main():

    lat, lon = get_mesh_latlon(splits=6)



    coarse_nodes = get_coarse_mesh_node_indices(
    fine_splits=6,
    coarse_splits=node_hierarchy_level,   # M4 ≈ 2562 nodes
    )

    print("Nodes per timestep:", len(coarse_nodes))
    print(f"Using M{node_hierarchy_level} coarse mesh nodes: {len(coarse_nodes)}")




    # after sampled_nodes is defined
    all_nodes = coarse_nodes
    samples_per_t = len(all_nodes)

    if score_values == "PCA":
        _, timestamps = load_timestamps(TIMESTEP_FILES_TXT)
        pc_scores = np.load(PC_SCORES_PATH, mmap_mode="r")
        T, N, K = pc_scores.shape
        max_features = min(max(PC_COUNTS), K)
        feature_counts = PC_COUNTS

        X = np.asarray(pc_scores[:, all_nodes, :max_features], dtype=np.float32)

        X = X.reshape(T * samples_per_t, max_features)

    elif score_values == "raw_activations":
        act_files = sorted(glob(os.path.join(ACTS_DIR, "*.npy")))

        valid_files = []

        for f in act_files:
            X_t = load_activations(f)

            if np.isnan(X_t).any():
                continue

            valid_files.append(f)


        act_files = valid_files
        timestamps = pd.to_datetime([parse_timestamp_from_path(p) for p in act_files])
        T = len(act_files)

        X_parts = []

        for i, f in enumerate(act_files):
            X_t = load_activations(f)  # [nodes, 512]
            X_parts.append(X_t[all_nodes, :])

            if (i + 1) % 100 == 0:
                print(f"Loaded raw activations for {i + 1}/{T}")

        X = np.stack(X_parts, axis=0).astype(np.float32)
        max_features = X.shape[2]
        feature_counts = [max_features]

        X = X.reshape(T * samples_per_t, max_features)

    else:
        raise ValueError(f"wrong input for score_values. Your input was {score_values}")

    
    
    

    time_index = np.repeat(timestamps.values, samples_per_t)
    time_index = pd.to_datetime(time_index)

    train_mask = time_index < pd.Timestamp("2021-10-01")
    test_mask = ~train_mask


    print("X shape:", X.shape)
    print("Train samples:", train_mask.sum())
    print("Test samples:", test_mask.sum())

    results = []

    if run_analysis_on == "derived":
        targets = DERIVED_TARGETS
    elif run_analysis_on == "direct":
        targets = TARGETS
    else:
        raise ValueError(f"wrong input for run_analysis_on: {run_analysis_on}")

    # Precompute all y values once
    y_parts_by_target = {target["name"]: [] for target in targets}

    for t_idx, timestamp in enumerate(timestamps):
        if run_analysis_on == "direct":
            ys = sample_all_targets_at_nodes(timestamp, targets, lat, lon, all_nodes)

        elif run_analysis_on == "derived":
            ys = sample_all_derived_targets_at_nodes(timestamp, targets, lat, lon, all_nodes)

        for name, y in ys.items():
            y_parts_by_target[name].append(y)

        if (t_idx + 1) % 100 == 0:
            print(f"Loaded ERA5 targets for {t_idx + 1}/{T} timesteps")

    y_by_target = {
        name: np.concatenate(parts).astype(np.float32)
        for name, parts in y_parts_by_target.items()
    }

    valid_X = np.all(np.isfinite(X), axis=1)

    for target in targets:
        print(f"\nTarget: {target['name']}")

        y = y_by_target[target["name"]]

        valid = np.isfinite(y) & valid_X
        valid_train = valid & train_mask
        valid_test = valid & test_mask

        for n_features in feature_counts:


            X_train = X[valid_train, :n_features]
            X_test = X[valid_test, :n_features]
            y_train = y[valid_train]
            y_test = y[valid_test]

            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=1.0)
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r = corr(y_test, y_pred)

            results.append({
                "target": target["name"],
                "n_features": n_features,
                "r2_test": r2,
                "rmse_test": rmse,
                "corr_test": r,
                "n_train": len(y_train),
                "n_test": len(y_test),
            })

            print(
                f"{target['name']:>6s} | PCs/features={n_features:>3d} | "
                f"R2={r2:.3f} | r={r:.3f} | RMSE={rmse:.3f}"
            )

    df = pd.DataFrame(results)

    out_csv = os.path.join(OUT_DIR, "pc_regression_physical_variables.csv")
    df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()