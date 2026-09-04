import dataclasses
import functools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import haiku as hk
import jax
import numpy as np
import pandas as pd
import xarray as xr
from google.cloud import storage

from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
    rollout,
)
from graphcast.deep_typed_graph_net import (
    DirectionInjector,
    get_activation_manager,
)


DEFAULT_MODEL_SOURCE = (
    "GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 "
    "- mesh 2to6 - precipitation input and output.npz"
)


@dataclass
class PerturbationDirection:
    direction: np.ndarray
    probe_weight: np.ndarray
    probe_bias: float
    threshold: float

    def validate(self) -> None:
        direction = np.asarray(self.direction)
        probe_weight = np.asarray(self.probe_weight)

        if direction.ndim != 1:
            raise ValueError(
                f"direction must be 1-D, got shape {direction.shape}"
            )

        if direction.shape[0] != 512:
            raise ValueError(
                "direction must be 512-D, "
                f"got shape {direction.shape}"
            )

        if probe_weight.ndim != 1:
            raise ValueError(
                f"probe_weight must be 1-D, got shape {probe_weight.shape}"
            )

        if probe_weight.shape[0] != 512:
            raise ValueError(
                "probe_weight must be 512-D, "
                f"got shape {probe_weight.shape}"
            )


@dataclass
class GraphCastResources:
    model_config: object
    task_config: object
    params: object
    state: dict
    diffs_stddev_by_level: xr.Dataset
    mean_by_level: xr.Dataset
    stddev_by_level: xr.Dataset


def round_to_nearest_6h(value) -> pd.Timestamp:
    value = pd.Timestamp(value)
    hour = round(value.hour / 6) * 6

    if hour == 24:
        return pd.Timestamp(value.date()) + pd.Timedelta(days=1)

    return pd.Timestamp(value.date()) + pd.Timedelta(hours=hour)


def _open_and_trim(path: str | Path) -> xr.Dataset:
    ds = xr.open_dataset(path)

    # ERA5 daily files used by this project are expected to contain
    # four 6-hourly timesteps. Preserve the previous behavior if a file
    # contains additional timesteps.
    if "time" in ds.dims and ds.sizes["time"] > 4:
        ds = ds.isel(time=slice(0, 4))

    return ds


def forecast_window(
    data_dir: str | Path,
    center_time,
    n_days: int,
) -> xr.Dataset:
    """
    Build the ERA5 window expected by GraphCast.

    The returned dataset contains:
      - the input state at center_time - 6h and center_time
      - target/forcing times through center_time + n_days
      - a batch dimension
      - relative `time` plus absolute `datetime`
    """
    data_dir = Path(data_dir)

    t0 = np.datetime64(center_time)
    start = t0 - np.timedelta64(6, "h")
    end = t0 + np.timedelta64(n_days * 24, "h")

    needed_days = pd.date_range(
        str(np.datetime64(start, "D")),
        str(np.datetime64(end, "D")),
        freq="D",
    )

    file_paths = [
        data_dir / f"era5_{day.strftime('%Y-%m-%d')}.nc"
        for day in needed_days
    ]

    missing = [path for path in file_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing ERA5 files required for forecast window:\n{missing_text}"
        )

    daily = [_open_and_trim(path) for path in file_paths]

    var_time = [
        name
        for name, array in daily[0].data_vars.items()
        if "time" in array.dims
    ]
    var_static = [
        name
        for name, array in daily[0].data_vars.items()
        if "time" not in array.dims
    ]

    ds_time = xr.concat(
        [dataset[var_time] for dataset in daily],
        dim="time",
    ).sortby("time")

    ds_static = daily[0][var_static]
    ds = xr.merge([ds_time, ds_static])

    target_times = pd.date_range(
        pd.Timestamp(str(start)),
        pd.Timestamp(str(end)),
        freq="6h",
    ).values.astype(ds.time.dtype)

    # Keep the same selection semantics as the original working script.
    # xarray will raise if one of the requested timesteps is unavailable.
    ds = ds.sel(time=target_times)
    ds_new = ds.copy()

    for variable in ds_new.data_vars:
        if "time" in ds_new[variable].dims:
            ds_new[variable] = ds_new[variable].expand_dims("batch")

    for coord in ds.coords:
        if "time" in ds[coord].dims:
            ds_new = ds_new.assign_coords(
                {coord: ds[coord].expand_dims("batch")}
            )

    time_orig = ds["time"]
    t_ref = time_orig.values[0]
    time_delta = time_orig - t_ref

    ds_new = ds_new.assign_coords(time=time_delta)
    ds_new = ds_new.assign_coords(
        datetime=("time", time_orig.values)
    )
    ds_new = ds_new.assign_coords(
        {"datetime": ds_new["datetime"].expand_dims("batch")}
    )

    return ds_new


def load_graphcast_resources(
    *,
    model_source: str = DEFAULT_MODEL_SOURCE,
    bucket_name: str = "dm_graphcast",
    prefix: str = "graphcast/",
) -> GraphCastResources:
    """
    Load GraphCast parameters, configs and normalization statistics
    from the public DeepMind GraphCast bucket.
    """
    client = storage.Client.create_anonymous_client()
    bucket = client.get_bucket(bucket_name)

    with bucket.blob(f"{prefix}params/{model_source}").open("rb") as file:
        ckpt = checkpoint.load(file, graphcast.CheckPoint)

    with bucket.blob(
        prefix + "stats/diffs_stddev_by_level.nc"
    ).open("rb") as file:
        diffs_stddev_by_level = xr.load_dataset(file).compute()

    with bucket.blob(
        prefix + "stats/mean_by_level.nc"
    ).open("rb") as file:
        mean_by_level = xr.load_dataset(file).compute()

    with bucket.blob(
        prefix + "stats/stddev_by_level.nc"
    ).open("rb") as file:
        stddev_by_level = xr.load_dataset(file).compute()

    return GraphCastResources(
        model_config=ckpt.model_config,
        task_config=ckpt.task_config,
        params=ckpt.params,
        state={},
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )


def construct_wrapped_graphcast(
    *,
    resources: GraphCastResources,
    intervention: PerturbationDirection,
    gamma: float,
    injection_steps: Sequence[int],
    injection_node_sets: Sequence[str],
):
    """
    Construct the modified GraphCast predictor for one gamma value.
    """
    intervention.validate()

    direction_injector = DirectionInjector(
        direction=np.asarray(
            intervention.direction,
            dtype=np.float32,
        ),
        probe_weight=np.asarray(
            intervention.probe_weight,
            dtype=np.float32,
        ),
        probe_bias=float(
            intervention.probe_bias
        ),
        threshold=float(
            intervention.threshold
        ),
        gamma=float(gamma),
    )

    predictor = graphcast.GraphCast(
        resources.model_config,
        resources.task_config,
        mesh_direction_injector=direction_injector,
        mesh_direction_steps=list(injection_steps),
        mesh_direction_node_sets=list(injection_node_sets),
    )

    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=resources.diffs_stddev_by_level,
        mean_by_level=resources.mean_by_level,
        stddev_by_level=resources.stddev_by_level,
    )
    predictor = autoregressive.Predictor(
        predictor,
        gradient_checkpointing=True,
    )

    return predictor


def make_run_forward_jitted(
    *,
    resources: GraphCastResources,
    intervention: PerturbationDirection,
    gamma: float,
    injection_steps: Sequence[int],
    injection_node_sets: Sequence[str],
):
    """
    Create the JIT-compiled forward function for one gamma value.
    """

    @hk.transform_with_state
    def run_forward(
        model_config,
        task_config,
        inputs,
        targets_template,
        forcings,
    ):
        # model_config/task_config are passed explicitly so Haiku/JAX sees
        # the same calling convention as in the original working script.
        del model_config, task_config

        predictor = construct_wrapped_graphcast(
            resources=resources,
            intervention=intervention,
            gamma=gamma,
            injection_steps=injection_steps,
            injection_node_sets=injection_node_sets,
        )

        return predictor(
            inputs,
            targets_template=targets_template,
            forcings=forcings,
        )

    apply_fn = functools.partial(
        run_forward.apply,
        params=resources.params,
        state=resources.state,
        model_config=resources.model_config,
        task_config=resources.task_config,
    )

    jitted = jax.jit(apply_fn)

    def drop_state(**kwargs):
        output, _state = jitted(**kwargs)
        return output

    return drop_state


def run_single_forecast(
    *,
    resources: GraphCastResources,
    intervention: PerturbationDirection,
    era5_window: xr.Dataset,
    gamma: float,
    n_days: int,
    injection_steps: Sequence[int] = (8,),
    injection_node_sets: Sequence[str] = ("mesh_nodes",),
    random_seed: int = 0,
) -> xr.Dataset:
    """
    Run one perturbed GraphCast forecast.
    """
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        era5_window,
        target_lead_times=slice("6h", f"{n_days * 24}h"),
        **dataclasses.asdict(resources.task_config),
    )

    run_forward_jitted = make_run_forward_jitted(
        resources=resources,
        intervention=intervention,
        gamma=gamma,
        injection_steps=injection_steps,
        injection_node_sets=injection_node_sets,
    )

    prediction = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(random_seed),
        inputs=inputs,
        targets_template=targets * np.nan,
        forcings=forcings,
    )

    return prediction


def run_perturbation_experiment(
    *,
    intervention: PerturbationDirection,
    gammas: Iterable[float],
    start_times: str | pd.Timestamp | Sequence[str | pd.Timestamp],
    n_days: int,
    era5_data_dir: str | Path,
    out_dir: str | Path,
    resources: GraphCastResources | None = None,
    model_source: str = DEFAULT_MODEL_SOURCE,
    injection_steps: Sequence[int] = (8,),
    injection_node_sets: Sequence[str] = ("mesh_nodes",),
    random_seed: int = 0,
    activation_manager_save_dir: str | Path | None = None,
) -> None:
    """
    Run a family of GraphCast perturbation forecasts.

    Experiment-specific code must construct the intervention first.

    Both the perturbation direction and the probe classifier must already
    be expressed in GraphCast's 512-D activation space.

    Output layout:
        out_dir/
            <center-time>/
                gamma_<gamma>.nc
    """
    intervention.validate()

    if isinstance(start_times, (str, pd.Timestamp)):
        start_times = [start_times]

    centers = [
        np.datetime64(round_to_nearest_6h(value))
        for value in start_times
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if resources is None:
        resources = load_graphcast_resources(
            model_source=model_source,
        )

    activation_manager = get_activation_manager()
    activation_manager.__init__(
        enabled=False,
        save_dir=(
            str(activation_manager_save_dir)
            if activation_manager_save_dir is not None
            else str(out_dir / "_unused_activations")
        ),
        save_steps=None,
        save_node_sets=None,
        mode="post_res",
    )

    started = time.time()

    for gamma in gammas:
        print(f"[gamma] {gamma}")

        for center in centers:
            center_str = np.datetime_as_string(center, unit="h")
            print(f"[time] {center_str}")

            activation_manager.set_time(center_str)

            era5_window = forecast_window(
                data_dir=era5_data_dir,
                center_time=center_str,
                n_days=n_days,
            )

            prediction = run_single_forecast(
                resources=resources,
                intervention=intervention,
                era5_window=era5_window,
                gamma=float(gamma),
                n_days=n_days,
                injection_steps=injection_steps,
                injection_node_sets=injection_node_sets,
                random_seed=random_seed,
            )

            center_out_dir = out_dir / center_str / "data"
            center_out_dir.mkdir(parents=True, exist_ok=True)

            out_path = center_out_dir / f"gamma_{gamma}.nc"
            prediction.to_netcdf(out_path)

            print(f"[saved] {out_path}")

    print(f"[all done] {time.time() - started:.1f}s")
