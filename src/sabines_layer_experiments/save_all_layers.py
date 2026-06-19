# run_graphcast_save_all_layers_jan2021.py

#!/usr/bin/env python3
import dataclasses
import functools
import os
import time

from google.cloud import storage
import haiku as hk
import jax
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
)
from graphcast.deep_typed_graph_net import get_activation_manager


DATA_DIR = "/share/prj-4d/graphcast_shared/data/era5_daily_nc"
ACTS_DIR = "/share/prj-4d/graphcast_shared/data/graphcast_activations_all_layers_January_2021"

# Zero-based processor layers. These save as layer0000 ... layer0015.
SAVE_STEPS = list(range(16))

CENTERS = np.arange(
    np.datetime64("2021-01-01T00"),
    np.datetime64("2021-02-01T00"),
    np.timedelta64(6, "h"),
)


def _open_and_trim(path: str) -> xr.Dataset:
    ds = xr.open_dataset(path)
    if "time" in ds.dims and ds.sizes["time"] > 4:
        ds = ds.isel(time=slice(0, 4))
    return ds


def three_step_window(data_dir: str, center_time: str) -> xr.Dataset | None:
    t0 = np.datetime64(center_time)
    t_minus = t0 - np.timedelta64(6, "h")
    t_plus = t0 + np.timedelta64(6, "h")

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

    var_time = [v for v, da in daily[0].data_vars.items() if "time" in da.dims]
    var_static = [v for v, da in daily[0].data_vars.items() if "time" not in da.dims]

    ds_time = xr.concat([d[var_time] for d in daily], dim="time").sortby("time")
    ds_static = daily[0][var_static]
    ds = xr.merge([ds_time, ds_static])

    target_times = np.array([t_minus, t0, t_plus], dtype=ds.time.dtype)
    if not all(t in ds.time.values for t in target_times):
        print(f"Missing needed times for center {center_time}")
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


def main():
    os.makedirs(ACTS_DIR, exist_ok=True)

    gcs = storage.Client.create_anonymous_client()
    bucket = gcs.get_bucket("dm_graphcast")
    prefix = "graphcast/"

    model_source = (
        "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 "
        "- mesh 2to6 - precipitation input and output.npz"
    )

    print("Loading checkpoint...")
    with bucket.blob(f"{prefix}params/{model_source}").open("rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)

    model_config = ckpt.model_config
    task_config = ckpt.task_config
    params = ckpt.params
    state = {}

    print("Loading normalization stats...")
    with bucket.blob(prefix + "stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev_by_level = xr.load_dataset(f).compute()
    with bucket.blob(prefix + "stats/mean_by_level.nc").open("rb") as f:
        mean_by_level = xr.load_dataset(f).compute()
    with bucket.blob(prefix + "stats/stddev_by_level.nc").open("rb") as f:
        stddev_by_level = xr.load_dataset(f).compute()

    def construct_wrapped_graphcast(model_config, task_config):
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev_by_level,
            mean_by_level=mean_by_level,
            stddev_by_level=stddev_by_level,
        )
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
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

    run_forward_jitted = drop_state(with_params(jax.jit(with_configs(run_forward.apply))))

    am = get_activation_manager()
    am.__init__(
        enabled=True,
        save_dir=ACTS_DIR,
        save_steps=SAVE_STEPS,
        save_node_sets=["mesh_nodes"],
        mode="post_res",
    )

    t_start = time.time()

    for center in CENTERS:
        center_str = np.datetime_as_string(center, unit="h")
        print(f"[TIME] {center_str}", flush=True)

        expected_paths = [
            os.path.join(
                ACTS_DIR,
                f"layer{step:04d}_mesh_gnn_post_res_nodes_mesh_nodes_t{center_str}.npy",
            )
            for step in SAVE_STEPS
        ]

        if all(os.path.exists(p) for p in expected_paths):
            print(f"[SKIP existing all layers] {center_str}", flush=True)
            continue

        am.set_time(center_str)

        ds = three_step_window(DATA_DIR, center_str)
        if ds is None:
            print(f"[MISS] {center_str}", flush=True)
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

        print(f"[DONE] {center_str}", flush=True)

    print(f"[ALL DONE] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()