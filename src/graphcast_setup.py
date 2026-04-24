#for now copied from the notebook, probs not all necessary

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

from graphcast.deep_typed_graph_net import get_activation_manager
from google.cloud import storage

#Authenticate with google Cloud Storage (to Access Graphcast storage)
gcs_client = storage.Client.create_anonymous_client()
gcs_bucket = gcs_client.get_bucket("dm_graphcast")
dir_prefix = "graphcast/"

data_dir = '/share/prj-4d/graphcast_shared/data/era5_daily_nc'        # contains era5_YYYY-MM-DD.nc
acts_dir = '/share/prj-4d/graphcast_shared/data/graphcast_activation'
os.makedirs(acts_dir, exist_ok=True)

### extracting time 00, 06, 12, 18
centers = np.arange(
    np.datetime64("2021-01-29T06"), # produces nodes for time 06, 12, 18
    np.datetime64("2021-01-29T12"),
    np.timedelta64(6, "h"),
)


# ============================================================
# ERA5 WINDOWING — *EXACTLY YOUR CODE*
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

def construct_wrapped_graphcast(model_config, task_config):
    predictor = graphcast.GraphCast(model_config, task_config)
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
def run_forward(model_config, task_config, inputs, targets_template, forcings):
    predictor = construct_wrapped_graphcast(model_config, task_config)
    return predictor(inputs, targets_template=targets_template, forcings=forcings)


def with_configs(fn):
    return functools.partial(fn, model_config=model_config, task_config=task_config)


def with_params(fn):
    return functools.partial(fn, params=params, state=state)


def drop_state(fn):
    return lambda **kw: fn(**kw)[0]


run_forward_jitted = drop_state(
    with_params(
        jax.jit(with_configs(run_forward.apply))
    )
)





# ============================================================
# ACTIVATION MANAGER — DISK, SUPPORTED
# ============================================================

am = get_activation_manager()
am.__init__(
    enabled=True,
    save_dir=acts_dir,
    save_steps=[8],
    save_node_sets=["mesh_nodes"],
    mode="post_res",
)


# ============================================================
# MAIN LOOP — SAME SEMANTICS AS YOUR SCRIPT
# ============================================================

t_start = time.time()

for center in centers:
    center_str = np.datetime_as_string(center, unit="h")
    print(f"[TIME] {center_str}")

    am.set_time(center_str)

    ds = three_step_window(data_dir, center_str)
    if ds is None:
        print(f"[MISS] {center_str}")
        continue

    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        ds,
        target_lead_times=slice("6h", "6h"),
        **dataclasses.asdict(task_config),
    )

    _ = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=inputs,
        targets_template=targets * np.nan,
        forcings=forcings,
    )

    print(f"[DONE] {center_str}")

print(f"[ALL DONE] {time.time() - t_start:.1f}s")