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
import joblib

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

REPRESENTATION = "PCA"   # "raw_activations" or "PCA"
N_PCS = 200                          # only used when REPRESENTATION == "PCA"


START_TIME = "2021-02-12T18"

N_DAYS = 5

THRESHOLD = 0.9
NODE_HIERARCHY_LEVEL = 6



def round_to_nearest_6h(t):
    t = pd.Timestamp(t)
    hour = round(t.hour / 6) * 6

    if hour == 24:
        return pd.Timestamp(t.date()) + pd.Timedelta(days=1)

    return pd.Timestamp(t.date()) + pd.Timedelta(hours=hour)

center = round_to_nearest_6h(START_TIME)
centers = [np.datetime64(center)]

PROBE_BASE_DIR = (
    f"/home/student/m/mbraatz/share/graphcast_analysis/"
    f"plots/malins_experiments/logistic_regression/"
    f"{WEATHER_FEATURE}/Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}"
)
pca_components = None
pca_mean = None

if REPRESENTATION == "raw_activations":

    N_FEATURES = 512

    PROBE_DIRECTION_PATH = (
        f"{PROBE_BASE_DIR}/raw_activations/"
        f"probe_direction_{WEATHER_FEATURE}_raw_activations_"
        f"intersection_M{NODE_HIERARCHY_LEVEL}_{N_FEATURES}_features_"
        f"2019_2020_train_only.npz"
    )

elif REPRESENTATION == "PCA":

    N_FEATURES = N_PCS

    PROBE_DIRECTION_PATH = (
        f"{PROBE_BASE_DIR}/PCA/"
        f"probe_direction_{WEATHER_FEATURE}_PCA_"
        f"intersection_M{NODE_HIERARCHY_LEVEL}_{N_PCS}_features_"
        f"2019_2020_train_only.npz"
    )

else:
    raise ValueError(f"Unknown REPRESENTATION: {REPRESENTATION}")


print("Using representation:", REPRESENTATION)
print("Using probe:", PROBE_DIRECTION_PATH)

probe = np.load(PROBE_DIRECTION_PATH)

scaler_mean = probe["scaler_mean"].astype(np.float32)
scaler_scale = probe["scaler_scale"].astype(np.float32)
coef_z = probe["coef_z"].astype(np.float32)
intercept = probe["intercept"].astype(np.float32)


if REPRESENTATION == "raw_activations":

    direction = probe["coef_z_unit"].astype(np.float32)


elif REPRESENTATION == "PCA":

    PCA_COMPONENTS_PATH = (
        "/share/prj-4d/graphcast_shared/data/"
        "pca_components/512_PCs/layer8_only/"
        "pca_components_2019_2020_layer8.npy"
    )

    PCA_MEAN_PATH = (
        "/share/prj-4d/graphcast_shared/data/"
        "pca_components/512_PCs/layer8_only/"
        "pca_mean_2019_2020_layer8.npy"
    )

    all_pca_components = np.load(PCA_COMPONENTS_PATH).astype(np.float32)
    pca_mean = np.load(PCA_MEAN_PATH).astype(np.float32)

    # Only use the first N_PCS
    pca_components = all_pca_components[:N_PCS]
    # logistic direction in ordinary PC-score coordinates
    direction_pc = coef_z / scaler_scale

    # Map PC direction back into the 512-D GraphCast activation space
    direction = pca_components.T @ direction_pc
    direction = direction.astype(np.float32)

    # Normalize injected raw-space direction
    direction /= np.linalg.norm(direction) + 1e-8


print("Final perturbation direction shape:", direction.shape)

#Authenticate with google Cloud Storage (to Access Graphcast storage)
gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "graphcast/"

data_dir = '/share/prj-4d/graphcast_shared/data/era5_daily_nc'        # contains era5_YYYY-MM-DD.nc
acts_dir = '/share/prj-4d/graphcast_shared/data/graphcast_activation_2021'
os.makedirs(acts_dir, exist_ok=True)




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
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        coef_z=coef_z,
        intercept=intercept,
        threshold=THRESHOLD,
        gamma=gamma,

        representation=REPRESENTATION,
        pca_components=pca_components,
        pca_mean=pca_mean,
    )

    print("REPRESENTATION:", REPRESENTATION)
    print("direction:", direction.shape)
    print("coef_z:", coef_z.shape)
    print("scaler_mean:", scaler_mean.shape)
    print("scaler_scale:", scaler_scale.shape)

    if REPRESENTATION == "PCA":
        print("pca_components:", pca_components.shape)
        print("pca_mean:", pca_mean.shape)

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


def make_run_forward_jitted(gamma):
    @hk.transform_with_state
    def run_forward_gamma(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config, gamma)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    return drop_state(
        with_params(
            jax.jit(with_configs(run_forward_gamma.apply))
        )
    )




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

        run_forward_jitted = make_run_forward_jitted(gamma)
        
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


        
        if REPRESENTATION == "PCA":
            representation_dir = f"PCA_{N_PCS}"
        else:
            representation_dir = "raw_activations"

        out_dir = os.path.join(
            "plots",
            "malins_experiments",
            "pertubation_experiments",
            WEATHER_FEATURE,
            f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}",
            representation_dir,
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