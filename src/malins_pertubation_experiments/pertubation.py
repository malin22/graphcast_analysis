#for now copied from the notebook, probs not all necessary
import sys
import os

LOCAL_ROOT = "/home/student/m/mbraatz/share/graphcast_analysis/graphcast"

sys.path.insert(0, LOCAL_ROOT)

import dataclasses
import datetime
import functools
import math
import os
import glob
import time
from typing import Optional
from google.cloud import storage
import haiku as hk
import jax
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import xarray as xr

import pandas as pd

from graphcast import icosahedral_mesh



from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
    rollout,
    xarray_jax,
    xarray_tree,
)


from graphcast.deep_typed_graph_net import get_activation_manager, DirectionInjector
from google.cloud import storage

import inspect

print("graphcast module:", graphcast.__file__)
print("GraphCast object:", graphcast.GraphCast)
print("GraphCast signature:", inspect.signature(graphcast.GraphCast))




GAMMA = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]

WEATHER_FEATURE = "AR"

MASK_TIME = "2021-01-03T05"

N_DAYS = 5

THRESHOLD = 0.8


def round_to_nearest_6h(t):
    t = pd.Timestamp(t)
    hour = round(t.hour / 6) * 6

    if hour == 24:
        return pd.Timestamp(t.date()) + pd.Timedelta(days=1)

    return pd.Timestamp(t.date()) + pd.Timedelta(hours=hour)


PROBE_DIRECTION_PATH = (
    "/home/student/m/mbraatz/share/graphcast_analysis/plots/malins_experiments/logistic_regression/AR/raw_activations/probe_direction_AR_raw_activations_intersection_M5_512_features_2020_train_only.npz"
)

direction = np.load(PROBE_DIRECTION_PATH)["direction_raw_delta"].astype(np.float32)

center = round_to_nearest_6h(MASK_TIME)
centers = [np.datetime64(center)]

probe = np.load(PROBE_DIRECTION_PATH)

direction = probe["direction_raw_delta"].astype(np.float32)
scaler_mean = probe["scaler_mean"].astype(np.float32)
scaler_scale = probe["scaler_scale"].astype(np.float32)
coef_z = probe["coef_z"].astype(np.float32)
intercept = probe["intercept"].astype(np.float32)

#Authenticate with google Cloud Storage (to Access Graphcast storage)
gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "graphcast/"

data_dir = '/share/prj-4d/graphcast_shared/data/era5_daily_nc'        # contains era5_YYYY-MM-DD.nc
acts_dir = '/share/prj-4d/graphcast_shared/data/graphcast_activation_2021'
os.makedirs(acts_dir, exist_ok=True)







MASK_DIR = "/share/prj-4d/graphcast_shared/data/ClimateNetLarge/AR_labels_cleaned"


def vertices_to_latlon(vertices):
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0])) % 360
    return lat, lon


def get_mesh_latlon(splits=6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    vertices = meshes[splits].vertices
    return vertices_to_latlon(vertices)


def nearest_mask_file(center_time, max_hours=3):
    mask_files = sorted(glob.glob(os.path.join(MASK_DIR, "*.nc")))

    center_time = pd.Timestamp(str(center_time))

    best_file = None
    best_diff = None

    for f in mask_files:
        t = pd.Timestamp(os.path.basename(f).replace(".nc", ""))
        diff = abs(t - center_time)

        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_file = f

    if best_diff > pd.Timedelta(hours=max_hours):
        return None

    return best_file



# ============================================================
# ERA5 WINDOWING 
# ============================================================

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

    file_paths = [
        os.path.join(data_dir, f"era5_{str(d)[:10]}.nc")
        for d in needed_days
    ]

    if any(not os.path.exists(p) for p in file_paths):
        return None

    daily = [_open_and_trim(p) for p in file_paths]

    var_time   = [v for v, da in daily[0].data_vars.items() if "time" in da.dims]
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
            ds_new = ds_new.assign_coords(
                {c: ds[c].expand_dims("batch")}
            )

    time_orig = ds["time"]
    t_ref = time_orig.values[0]
    time_delta = time_orig - t_ref

    ds_new = ds_new.assign_coords(time=time_delta)
    ds_new = ds_new.assign_coords(datetime=("time", time_orig.values))
    ds_new = ds_new.assign_coords(
        {"datetime": ds_new["datetime"].expand_dims("batch")}
    )

    return ds_new


def forecast_window(data_dir, center_time, n_days):
    t0 = np.datetime64(center_time)
    start = t0 - np.timedelta64(6, "h")
    end = t0 + np.timedelta64(n_days * 24, "h")

    needed_days = pd.date_range(
        str(np.datetime64(start, "D")),
        str(np.datetime64(end, "D")),
        freq="D",
    )

    file_paths = [
        os.path.join(data_dir, f"era5_{d.strftime('%Y-%m-%d')}.nc")
        for d in needed_days
    ]

    daily = [_open_and_trim(p) for p in file_paths]

    var_time = [v for v, da in daily[0].data_vars.items() if "time" in da.dims]
    var_static = [v for v, da in daily[0].data_vars.items() if "time" not in da.dims]

    ds_time = xr.concat([d[var_time] for d in daily], dim="time").sortby("time")
    ds_static = daily[0][var_static]

    ds = xr.merge([ds_time, ds_static])

    target_times = pd.date_range(
        pd.Timestamp(str(start)),
        pd.Timestamp(str(end)),
        freq="6h",
    ).values.astype(ds.time.dtype)

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
    ds_new = ds_new.assign_coords(
        {"datetime": ds_new["datetime"].expand_dims("batch")}
    )

    return ds_new


# ============================================================
# LOAD GRAPHCAST + STATS — *EXACTLY YOUR CODE*
# ============================================================
gcs = storage.Client.create_anonymous_client()
print("gcs set")
bucket = gcs.get_bucket("dm_graphcast")
print("bucket set")
prefix = "graphcast/"

model_source = (
    "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 "
    "- mesh 2to6 - precipitation input and output.npz"
)

with bucket.blob(f"{prefix}params/{model_source}").open("rb") as f:
    ckpt = checkpoint.load(f, graphcast.CheckPoint)

model_config = ckpt.model_config
task_config = ckpt.task_config
params = ckpt.params
state = {}

with bucket.blob(prefix + "stats/diffs_stddev_by_level.nc").open("rb") as f:
    diffs_stddev_by_level = xr.load_dataset(f).compute()

with bucket.blob(prefix + "stats/mean_by_level.nc").open("rb") as f:
    mean_by_level = xr.load_dataset(f).compute()

with bucket.blob(prefix + "stats/stddev_by_level.nc").open("rb") as f:
    stddev_by_level = xr.load_dataset(f).compute()


    
# ============================================================
# GRAPHCAST CONSTRUCTION — UNCHANGED
# ============================================================

def construct_wrapped_graphcast(model_config, task_config, gamma):

    direction_injector = DirectionInjector(
        direction=direction,
        gamma=gamma,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        coef_z=coef_z,
        intercept=intercept,
        threshold=THRESHOLD,   # or None for soft p_AR weighting everywhere
    )

    predictor = graphcast.GraphCast(
        model_config,
        task_config,
        mesh_direction_injector=direction_injector,
        mesh_direction_steps=[8],
        mesh_direction_node_sets=["mesh_nodes"],
    )

    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )
    predictor = autoregressive.Predictor(
        predictor, gradient_checkpointing=True
    )
    return predictor


@hk.transform_with_state
def run_forward(model_config, task_config, gamma, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config, gamma)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


def with_configs(fn):
    return functools.partial(fn, model_config=model_config, task_config=task_config)


def with_params(fn):
    return functools.partial(fn, params=params, state=state)


def drop_state(fn):
    return lambda **kw: fn(**kw)[0]






# ============================================================
# ACTIVATION MANAGER — DISK, SUPPORTED
# ============================================================

am = get_activation_manager()
am.__init__(
    enabled=False,
    save_dir=acts_dir,
    save_steps=None,
    save_node_sets=None,
    mode="post_res",
)


# ============================================================
# MAIN LOOP — SAME SEMANTICS AS YOUR SCRIPT
# ============================================================

t_start = time.time()



for gamma in GAMMA:
    print(f"using gamma = {gamma}")
    for center in centers:
        center_str = np.datetime_as_string(center, unit="h")
        print(f"[TIME] {center_str}")


        am.set_time(center_str)

        ds = forecast_window(data_dir, center_str, N_DAYS)

        inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
            ds,
            target_lead_times=slice("6h", f"{N_DAYS * 24}h"),
            **dataclasses.asdict(task_config),
        )

        mask_path = nearest_mask_file(center_str, max_hours=3)

        if mask_path is None:
            print(f"[NO AR MASK] {center_str}")
            continue



        run_forward_jitted = drop_state(
            with_params(
                jax.jit(with_configs(run_forward.apply,  gamma=gamma,))
            )
        )

        print("Using AR mask:", mask_path)

        
        print("inputs time:", inputs.time.values)
        print("targets time:", targets.time.values)
        print("forcings time:", forcings.time.values)

        print("n input steps:", inputs.sizes["time"])
        print("n target steps:", targets.sizes["time"])
        print("n forcing steps:", forcings.sizes["time"])


        pred = rollout.chunked_prediction(
            run_forward_jitted,
            rng=jax.random.PRNGKey(0),
            inputs=inputs,
            targets_template=targets * np.nan,
            forcings=forcings,
        )

        out_dir = os.path.join(
            "plots",
            "malins_experiments",
            "pertubation_experiments",
            WEATHER_FEATURE,
            f"pertubation_threshold_{THRESHOLD}",
            center_str,         
            "data",
            
        )

        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(
            out_dir,
            f"gamma_{gamma}.nc",
        )

        pred.to_netcdf(out_path)

        print("Saved prediction:", out_path)

        print("Saved prediction:", out_path)

        print(f"[DONE] {center_str}, {gamma}")

print(f"[ALL DONE] {time.time() - t_start:.1f}s")