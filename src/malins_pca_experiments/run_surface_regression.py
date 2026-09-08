import os

import pandas as pd

from malins_helper_scripts.mesh_context import (
    get_coarse_mesh_node_indices,
)

from malins_pca_experiments.config import (
    NODE_HIERARCHY_LEVEL,
    REGRESSION_TYPE,
    SCORE_VALUES,
    OUT_DIR,
)

from malins_pca_experiments.streaming_regression import (
    accumulate_training_statistics_multi,
    fit_regressions_from_statistics_multi,
    evaluate_streaming_regressions_multi,
)


SURFACE_TARGETS = [
    {
        "name": "2t",
        "var": "2m_temperature",
        "level": None,
    },
    {
        "name": "10u",
        "var": "10m_u_component_of_wind",
        "level": None,
    },
    {
        "name": "10v",
        "var": "10m_v_component_of_wind",
        "level": None,
    },
    {
        "name": "msl",
        "var": "mean_sea_level_pressure",
        "level": None,
    },
    {
        "name": "tp",
        "var": "total_precipitation_6hr",
        "level": None,
    },
]


PC_COUNTS = [
    5,
    10,
    25,
    50,
    100,
    200,
    400,
    512,
]


def main():

    targets = SURFACE_TARGETS
    pc_counts = PC_COUNTS

    all_nodes = get_coarse_mesh_node_indices(
        fine_splits=6,
        coarse_splits=NODE_HIERARCHY_LEVEL,
    )

    print(
        "Nodes per timestep:",
        len(all_nodes),
    )

    print(
        "Surface targets:",
        [target["name"] for target in targets],
    )

    print(
        "PC counts:",
        pc_counts,
    )

    if SCORE_VALUES != "PCA":
        raise NotImplementedError(
            "This runner currently supports PCA only"
        )

    if REGRESSION_TYPE == "linear":
        alpha = 0.0

    elif REGRESSION_TYPE == "ridge":
        alpha = 1.0

    else:
        raise ValueError(
            f"Unsupported regression type: "
            f"{REGRESSION_TYPE}"
        )

    max_features = max(
        pc_counts
    )

    # --------------------------------------------------
    # TRAINING
    # --------------------------------------------------

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

    # --------------------------------------------------
    # FIT ALL PC COUNTS
    # --------------------------------------------------

    models = {}

    for n_features in pc_counts:

        print(
            f"Fitting all surface targets "
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

    # --------------------------------------------------
    # TEST
    # --------------------------------------------------

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

    # --------------------------------------------------
    # BUILD OUTPUT TABLE
    # --------------------------------------------------

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

            rows.append(
                row
            )

            print(
                f"{target['name']:>5s} | "
                f"features={n_features:>3d} | "
                f"R2={row['r2_test']:.4f} | "
                f"r={row['corr_test']:.4f} | "
                f"RMSE={row['rmse_test']:.4f}"
            )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    df = pd.DataFrame(
        rows
    )

    out_path = os.path.join(
        OUT_DIR,
        "surface_variables_"
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