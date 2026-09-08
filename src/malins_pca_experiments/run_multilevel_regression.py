import os

import pandas as pd

from malins_helper_scripts.mesh_context import (
    get_coarse_mesh_node_indices,
)

import argparse

from malins_pca_experiments.config import (
    NODE_HIERARCHY_LEVEL,
    REGRESSION_TYPE,
    SCORE_VALUES,
    OUT_DIR,
    PRESSURE_LEVELS,
)

from malins_pca_experiments.streaming_regression import (
    accumulate_training_statistics_multi,
    fit_regressions_from_statistics_multi,
    evaluate_streaming_regressions_multi,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variable",
        required=True,
        choices=[
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "geopotential",
            "specific_humidity",
            "vertical_velocity",
        ],
    )

    args = parser.parse_args()

    VARIABLE_SHORT_NAMES = {
        "temperature": "t",
        "u_component_of_wind": "u",
        "v_component_of_wind": "v",
        "geopotential": "z",
        "specific_humidity": "q",
        "vertical_velocity": "w",
    }

    short_name = VARIABLE_SHORT_NAMES[args.variable]

    targets = [
        {
            "name": f"{short_name}{level}",
            "var": args.variable,
            "level": level,
        }
        for level in PRESSURE_LEVELS
    ]

    pc_counts = [
        5,
        10,
        25,
        50,
        100,
        200,
        400,
        512,
    ]

    print(
    f"Variable: {args.variable}"
    )

    print(
        f"Pressure levels: {PRESSURE_LEVELS}"
    )

    print(
        f"Number of targets: {len(targets)}"
    )

    print(
        f"PC counts: {pc_counts}"
    )

    # --------------------------------------------------

    all_nodes = get_coarse_mesh_node_indices(
        fine_splits=6,
        coarse_splits=NODE_HIERARCHY_LEVEL,
    )

    print(
        "Nodes per timestep:",
        len(all_nodes),
    )

    if SCORE_VALUES != "PCA":
        raise NotImplementedError(
            "PCA only"
        )

    if REGRESSION_TYPE == "linear":
        alpha = 0.0

    elif REGRESSION_TYPE == "ridge":
        alpha = 1.0

    else:
        raise ValueError(
            REGRESSION_TYPE
        )

    max_features = max(
        pc_counts
    )

    print(
        f"Targets: "
        f"{[t['name'] for t in targets]}"
    )

    print(
        f"PC counts: {pc_counts}"
    )

    print(
        "\nAccumulating shared training statistics..."
    )

    stats = accumulate_training_statistics_multi(
        targets=targets,
        all_nodes=all_nodes,
        max_features=max_features,
    )

    print(
        "Training samples:",
        stats["n"],
    )

    models = {}

    for n_features in pc_counts:

        print(
            f"Fitting all targets "
            f"with {n_features} PCs"
        )

        coef, intercept = (
            fit_regressions_from_statistics_multi(
                stats=stats,
                n_features=n_features,
                regression_type=REGRESSION_TYPE,
                alpha=alpha,
            )
        )

        print(
            "  coef shape:",
            coef.shape,
        )

        models[n_features] = {
            "coef": coef,
            "intercept": intercept,
        }

    print(
        "\nEvaluating all models "
        "in one 2021 pass..."
    )

    test_metrics = (
        evaluate_streaming_regressions_multi(
            targets=targets,
            all_nodes=all_nodes,
            models=models,
        )
    )

    rows = []

    for n_features in pc_counts:

        metrics = test_metrics[
            n_features
        ]

        for j, target in enumerate(
            targets
        ):
            row = {
                "target": target["name"],
                "variable": target["var"],
                "level": target["level"],
                "n_features": n_features,
                "alpha": alpha,
                "r2_test": metrics[
                    "r2_test"
                ][j],
                "rmse_test": metrics[
                    "rmse_test"
                ][j],
                "corr_test": metrics[
                    "corr_test"
                ][j],
                "n_train": stats["n"],
                "n_test": metrics[
                    "n_test"
                ],
            }

            rows.append(row)

            print(
                f"{target['name']:>8s} | "
                f"features={n_features:>3d} | "
                f"R2={row['r2_test']:.4f} | "
                f"r={row['corr_test']:.4f} | "
                f"RMSE={row['rmse_test']:.4f}"
            )

    df = pd.DataFrame(
        rows
    )

    safe_name = args.variable

    out_path = os.path.join(
        OUT_DIR,
        f"{safe_name}_"
        "all_pressure_levels_"
        "2019_2020train_2021test.csv",
    )

    df.to_csv(
        out_path,
        index=False,
    )

    print(
        "\nSaved:",
        out_path,
    )


if __name__ == "__main__":
    main()