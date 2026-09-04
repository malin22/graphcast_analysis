from pathlib import Path

from malins_perturbation_experiments.perturbation import (
    PerturbationDirection,
    run_perturbation_experiment
)


# ============================================================
# SHARED PERTURBATION CONFIG
# ============================================================

PROJECT_ROOT = Path(
    "/home/student/m/mbraatz/share/graphcast_analysis"
)

ERA5_DATA_DIR = Path(
    "/share/prj-4d/graphcast_shared/data/era5_daily_nc"
)

DEFAULT_GAMMAS = [
    -1.0,
    -0.5,
    -0.2,
    0.0,
    0.2,
    0.5,
    1.0,
]


DEFAULT_N_DAYS = 5

DEFAULT_INJECTION_STEPS = (8,)
DEFAULT_INJECTION_NODE_SETS = ("mesh_nodes",)
DEFAULT_RANDOM_SEED = 0


def build_output_dir(
    *,
    weather_feature: str,
    node_hierarchy_level: int,
    experiment_name: str,
) -> Path:
    """
    Build the shared output directory for one perturbation experiment.
    """

    return (
        PROJECT_ROOT
        / "results"
        / "perturbation"
        / weather_feature
        / f"Node_Hierarchy_Level_M{node_hierarchy_level}"
        / experiment_name/ )


def run_perturbation(
    *,
    intervention: PerturbationDirection,
    experiment_name: str,
    weather_feature: str,
    node_hierarchy_level: int,
    gammas=None,
    start_times=None,
    n_days: int = DEFAULT_N_DAYS,
    era5_data_dir: str | Path = ERA5_DATA_DIR,
    out_dir: str | Path | None = None,
    injection_steps=DEFAULT_INJECTION_STEPS,
    injection_node_sets=DEFAULT_INJECTION_NODE_SETS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """
    Run a perturbation experiment using a pre-built 512-D intervention.

    The experiment script is responsible for constructing:

        PerturbationDirection(
            direction=...,       # 512-D perturbation direction
            probe_weight=...,    # 512-D probe classifier
            probe_bias=...,      # scalar
            threshold=...,
        )

    This function handles the shared experiment settings and delegates
    the actual GraphCast run to `run_perturbation_experiment`.
    """

    if gammas is None:
        gammas = DEFAULT_GAMMAS

    if start_times is None:
        raise ValueError(
            "Please specify start_times for the perturbation experiment.")

    if out_dir is None:
        out_dir = build_output_dir(
            weather_feature=weather_feature,
            node_hierarchy_level=node_hierarchy_level,
            experiment_name=experiment_name,
        )

    out_dir = Path(out_dir)

    print("============================================")
    print("Perturbation experiment")
    print("============================================")
    print("Experiment:", experiment_name)
    print("Weather feature:", weather_feature)
    print("Node hierarchy level:", node_hierarchy_level)
    print("Start time(s):", start_times)
    print("Forecast days:", n_days)
    print("Gammas:", gammas)
    print("Output:", out_dir)
    print("============================================")

    run_perturbation_experiment(
        intervention=intervention,
        gammas=gammas,
        start_times=start_times,
        n_days=n_days,
        era5_data_dir=era5_data_dir,
        out_dir=out_dir,
        injection_steps=injection_steps,
        injection_node_sets=injection_node_sets,
        random_seed=random_seed,
    )
