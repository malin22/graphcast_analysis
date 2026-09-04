import os

import pandas as pd

from malins_helper_scripts.mesh_context import (
    get_coarse_mesh_node_indices,
)

from malins_pca_experiments.config import (
    NODE_HIERARCHY_LEVEL,
    PC_COUNTS,
    REGRESSION_TYPE,
    SCORE_VALUES,
    OUT_DIR,
    TARGETS,
)

from malins_pca_experiments.streaming_regression import (
    accumulate_training_statistics,
    fit_regression_from_statistics,
    evaluate_streaming_regressions,
)


def main():

    all_nodes = get_coarse_mesh_node_indices(
        fine_splits=6,
        coarse_splits=NODE_HIERARCHY_LEVEL,
    )

    samples_per_t = len(all_nodes)

    print("Nodes per timestep:", samples_per_t)
    print(
        f"Using M{NODE_HIERARCHY_LEVEL} coarse mesh nodes: "
        f"{samples_per_t}"
    )

    if SCORE_VALUES != "PCA":
        raise NotImplementedError(
            "Streaming implementation currently supports PCA only."
        )

    print(
        f"Running streaming {REGRESSION_TYPE} regression with PCA "
        f"to predict {len(TARGETS)} targets"
    )

    if REGRESSION_TYPE == "linear":
        alpha = 0.0

    elif REGRESSION_TYPE == "ridge":
        alpha = 1.0

    else:
        raise ValueError(
            f"Streaming version currently supports "
            f"'linear' and 'ridge', got {REGRESSION_TYPE}"
        )

    results = []

    max_features = max(PC_COUNTS)

    for target in TARGETS:
        print(f"\nTarget: {target['name']}")

        print(
            f"Accumulating training statistics for "
            f"{target['name']} using {max_features} PCs"
        )

        stats = accumulate_training_statistics(
            target=target,
            all_nodes=all_nodes,
            max_features=max_features,
        )

        n_train = stats["n"]

        models = {}

        for n_features in PC_COUNTS:

            print(
                f"Fitting {target['name']} "
                f"with {n_features} PCs"
            )

            coef, intercept = fit_regression_from_statistics(
                stats=stats,
                n_features=n_features,
                regression_type=REGRESSION_TYPE,
                alpha=alpha,
            )

            models[n_features] = {
                "coef": coef,
                "intercept": intercept,
            }

        print(
            f"Evaluating all {len(models)} models "
            f"for {target['name']} in one test pass"
        )

        test_metrics = evaluate_streaming_regressions(
            target=target,
            all_nodes=all_nodes,
            models=models,
        )

        for n_features in PC_COUNTS:

            metrics = test_metrics[n_features]

            results.append({
                "target": target["name"],
                "n_features": n_features,
                "alpha": alpha,
                "r2_test": metrics["r2_test"],
                "rmse_test": metrics["rmse_test"],
                "corr_test": metrics["corr_test"],
                "n_train": n_train,
                "n_test": metrics["n_test"],
                "n_selected": n_features,
            })

            print(
                f"{target['name']:>6s} | "
                f"features={n_features:>3d} | "
                f"alpha={alpha:.4g} | "
                f"test R2={metrics['r2_test']:.3f} | "
                f"test r={metrics['corr_test']:.3f} | "
                f"test RMSE={metrics['rmse_test']:.3f}"
            )

        del stats
        del models

    df = pd.DataFrame(results)

    out_csv = os.path.join(
        OUT_DIR,
        "pc_regression_physical_variables_"
        "2019_2020train_2021test.csv",
    )

    df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)


if __name__ == "__main__":
    main()