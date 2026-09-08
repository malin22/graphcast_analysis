
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

from malins_regression.run_logistic_probe import (
    run_logistic_experiment,
)


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

WEATHER_FEATURE = "AR"  # "AR" or "TC"

FINE_MESH_LEVEL = 6
NODE_HIERARCHY_LEVEL = 6

LABEL_MODE = "intersection"
MAX_TIME_DIFFERENCE_HOURS = 3


TOTAL_N_PCS = 512


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

TRAIN_START = pd.Timestamp("2019-01-01")
TRAIN_END = pd.Timestamp("2020-11-01")

VAL_START = pd.Timestamp("2020-11-01")
VAL_END = pd.Timestamp("2021-01-01")

TEST_START = pd.Timestamp("2021-01-01")
TEST_END = pd.Timestamp("2022-01-01")


# ============================================================
# PCA DATA PATHS
# ============================================================

PC_SCORES_PATHS = {
    2019: (
        "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
        "pc_scores_2019_from_2019_2020_pca_per_timestep.npy"
    ),
    2020: (
        "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
        "pc_scores_2020_from_2019_2020_pca_per_timestep.npy"
    ),
    2021: (
        "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
        "pc_scores_2021_from_2019_2020_pca_per_timestep.npy"
    ),
}


TIMESTEP_FILES_TXTS = {
    2019: (
        "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
        "pc_scores_2019_from_2019_2020_pca_per_timestep_files.txt"
    ),
    2020: (
        "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
        "pc_scores_2020_from_2019_2020_pca_per_timestep_files.txt"
    ),
    2021: (
        "/share/prj-4d/graphcast_shared/data/pc_scores_per_timestep/"
        "pc_scores_2021_from_2019_2020_pca_per_timestep_files.txt"
    ),
}


# ============================================================
# MASK PATH
# ============================================================

MASK_DIR = (
    f"/share/prj-4d/graphcast_shared/data/"
    f"ClimateNetLarge/{WEATHER_FEATURE}_labels_cleaned"
)


# ============================================================
# RESULTS BASE
# ============================================================

RESULTS_BASE = Path(
    f"results/logistic_regression/"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}"
)


# ============================================================
# SAVED L1 C-SWEEP RESULTS
# ============================================================

C_SWEEP_RESULTS_DIR = (
    RESULTS_BASE
    / "l1_pc_selection"
    / "c_sweep_results"
)


# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = (
    RESULTS_BASE
    / "l1_selected_pcs_l2_sweep"
)

RUNS_DIR = (
    OUT_DIR
    / "runs"
)

SUMMARY_PATH = (
    OUT_DIR
    / "summary.csv"
)


# ============================================================
# LOAD L1 C-SWEEP RESULTS
# ============================================================

def load_l1_sweep_results(results_dir):
    """
    Load all saved L1-selection results.

    Each c_XX.npz should contain:
        c_index
        C
        selected_pcs
        n_selected

    plus optionally validation metrics from the initial
    selector evaluation.
    """

    result_paths = sorted(
        glob.glob(
            str(
                Path(results_dir)
                / "c_*.npz"
            )
        )
    )

    if not result_paths:
        raise FileNotFoundError(
            "Could not find any L1 C-sweep results in:\n"
            f"{results_dir}\n\n"
            "Run the L1 C sweep first."
        )

    results = []

    for path in result_paths:

        with np.load(
            path,
            allow_pickle=True,
        ) as data:

            required_keys = {
                "c_index",
                "C",
                "selected_pcs",
                "n_selected",
            }

            missing = (
                required_keys
                .difference(data.files)
            )

            if missing:
                raise KeyError(
                    f"{path} is missing required keys: "
                    f"{sorted(missing)}"
                )

            c_index = int(
                data["c_index"]
            )

            C = float(
                data["C"]
            )

            selected_pcs = np.asarray(
                data["selected_pcs"],
                dtype=int,
            )

            n_selected = int(
                data["n_selected"]
            )

            val_average_precision = (
                float(
                    data[
                        "val_average_precision"
                    ]
                )
                if "val_average_precision"
                in data.files
                else np.nan
            )

            val_roc_auc = (
                float(
                    data["val_roc_auc"]
                )
                if "val_roc_auc"
                in data.files
                else np.nan
            )

            val_f1 = (
                float(
                    data["val_f1"]
                )
                if "val_f1"
                in data.files
                else np.nan
            )

            l1_n_iter = (
                int(
                    data["l1_n_iter"]
                )
                if "l1_n_iter"
                in data.files
                else -1
            )

        # Sanity check
        if len(selected_pcs) != n_selected:
            raise ValueError(
                f"Inconsistent selection in {path}:\n"
                f"n_selected = {n_selected}\n"
                f"len(selected_pcs) = "
                f"{len(selected_pcs)}"
            )

        results.append(
            {
                "path": str(path),
                "c_index": c_index,
                "C": C,
                "selected_pcs": (
                    selected_pcs
                ),
                "n_selected": (
                    n_selected
                ),
                "val_average_precision": (
                    val_average_precision
                ),
                "val_roc_auc": (
                    val_roc_auc
                ),
                "val_f1": (
                    val_f1
                ),
                "l1_n_iter": (
                    l1_n_iter
                ),
            }
        )

    # Sort by C-index rather than relying on filenames
    results.sort(
        key=lambda x: x["c_index"]
    )

    return results


# ============================================================
# PRINT L1 SWEEP
# ============================================================

def print_l1_sweep_summary(results):

    print()
    print("=" * 100)
    print("L1 C-SWEEP RESULTS")
    print("=" * 100)

    for result in results:

        print(
            f"c_index={result['c_index']:02d} | "
            f"C={result['C']:.8g} | "
            f"selected={result['n_selected']:4d} | "
            f"selector val_AP="
            f"{result['val_average_precision']:.6f}"
        )

    print("=" * 100)
    print()


# ============================================================
# EXPECTED RUN OUTPUT CSV
# ============================================================

def get_expected_result_csv(
    run_out_dir,
    experiment_name,
):
    """
    Match the naming convention used by
    run_logistic_experiment.

    Example:

    logistic_probe_TC_l1_selected_pcs_c03_n7_
    intersection_M6_max_3hour.csv
    """

    filename = (
        f"logistic_probe_"
        f"{WEATHER_FEATURE}_"
        f"{experiment_name}_"
        f"{LABEL_MODE}_"
        f"M{NODE_HIERARCHY_LEVEL}_"
        f"max_{MAX_TIME_DIFFERENCE_HOURS}hour.csv"
    )

    return (
        Path(run_out_dir)
        / filename
    )


# ============================================================
# LOCATE GENERATED RESULT CSV
# ============================================================

def locate_result_csv(
    run_out_dir,
    experiment_name,
):
    """
    First try the expected filename.

    If the runner uses a slightly different filename,
    fall back to finding CSV files in the run directory.
    """

    expected = get_expected_result_csv(
        run_out_dir,
        experiment_name,
    )

    if expected.exists():
        return expected

    csv_files = sorted(
        Path(run_out_dir).glob("*.csv")
    )

    if len(csv_files) == 1:
        return csv_files[0]

    if len(csv_files) == 0:
        raise FileNotFoundError(
            "run_logistic_experiment finished, "
            "but no CSV result was found in:\n"
            f"{run_out_dir}"
        )

    raise RuntimeError(
        "Multiple CSV files were found and the "
        "expected result filename was not present:\n"
        f"{run_out_dir}\n\n"
        f"Found:\n"
        + "\n".join(
            str(path)
            for path in csv_files
        )
    )


# ============================================================
# RUN ONE L2 EXPERIMENT
# ============================================================

def run_one_l2_experiment(result):

    c_index = result["c_index"]
    C = result["C"]

    selected_pcs = np.asarray(
        result["selected_pcs"],
        dtype=int,
    )

    n_selected = int(
        result["n_selected"]
    )

    # --------------------------------------------------------
    # Experiment name
    # --------------------------------------------------------

    experiment_name = (
        f"l1_selected_pcs_"
        f"c{c_index:02d}_"
        f"n{n_selected}"
    )

    # --------------------------------------------------------
    # Separate directory for this L1 selection
    # --------------------------------------------------------

    run_out_dir = (
        RUNS_DIR
        / f"c_{c_index:02d}_"
        f"npcs_{n_selected}"
    )

    run_out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Print what is being run
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("RUNNING L2 PROBE")
    print("=" * 100)

    print(
        f"C index: {c_index}"
    )

    print(
        f"L1 C: {C:.8g}"
    )

    print(
        f"Selected PCs: {n_selected}"
    )

    print(
        "PC indices:"
    )
    print(
        selected_pcs
    )

    print(
        "PC numbers:"
    )
    print(
        selected_pcs + 1
    )

    print(
        "Output directory:"
    )
    print(
        run_out_dir
    )

    print("=" * 100)
    print()

    # --------------------------------------------------------
    # Tell run_logistic_experiment exactly which PCs to use
    # --------------------------------------------------------

    def use_selected_pcs(_):
        return selected_pcs

    # --------------------------------------------------------
    # Run standard L2 logistic regression experiment
    # --------------------------------------------------------

    run_logistic_experiment(
        experiment_name=(
            experiment_name
        ),

        weather_feature=(
            WEATHER_FEATURE
        ),

        feature_source="pca",

        # Exactly one feature set for this run
        feature_counts=[
            n_selected
        ],

        selected_features_fn=(
            use_selected_pcs
        ),

        fine_mesh_level=(
            FINE_MESH_LEVEL
        ),

        node_hierarchy_level=(
            NODE_HIERARCHY_LEVEL
        ),

        label_mode=(
            LABEL_MODE
        ),

        max_time_difference_hours=(
            MAX_TIME_DIFFERENCE_HOURS
        ),


        train_start=(
            TRAIN_START
        ),

        train_end=(
            TRAIN_END
        ),

        val_start=(
            VAL_START
        ),

        val_end=(
            VAL_END
        ),

        test_start=(
            TEST_START
        ),

        test_end=(
            TEST_END
        ),

        mask_dir=(
            MASK_DIR
        ),

        out_dir=str(
            run_out_dir
        ),

        pc_scores_paths=(
            PC_SCORES_PATHS
        ),

        timestep_files_txts=(
            TIMESTEP_FILES_TXTS
        ),

        extra_metadata={
            "feature_selection": (
                "l1_logistic_nonzero_coefficients"
            ),

            "selection_source": (
                "l1_c_sweep"
            ),

            "selection_path": (
                result["path"]
            ),

            "l1_c_index": (
                c_index
            ),

            "l1_C": (
                C
            ),

            "n_l1_selected_pcs": (
                n_selected
            ),

            "selected_pc_indices": (
                selected_pcs.tolist()
            ),

            "selected_pc_numbers": (
                (selected_pcs + 1).tolist()
            ),

            "l1_selector_val_average_precision": (
                result[
                    "val_average_precision"
                ]
            ),

            "l1_selector_val_roc_auc": (
                result[
                    "val_roc_auc"
                ]
            ),

            "l1_selector_val_f1": (
                result[
                    "val_f1"
                ]
            ),

            "l1_n_iter": (
                result["l1_n_iter"]
            ),
        },
    )

    # --------------------------------------------------------
    # Find the CSV produced by run_logistic_experiment
    # --------------------------------------------------------

    result_csv = locate_result_csv(
        run_out_dir=run_out_dir,
        experiment_name=experiment_name,
    )

    print(
        "Generated result CSV:"
    )
    print(
        result_csv
    )

    # --------------------------------------------------------
    # Load result
    # --------------------------------------------------------

    df = pd.read_csv(
        result_csv
    )

    if len(df) == 0:
        raise ValueError(
            f"Result CSV is empty:\n"
            f"{result_csv}"
        )

    # --------------------------------------------------------
    # Force a common feature-count column for plotting
    # --------------------------------------------------------

    df["n_features"] = (
        n_selected
    )

    # --------------------------------------------------------
    # Add explicit L1-selection metadata
    #
    # These columns are useful later when comparing points
    # that happen to have the same number of selected PCs.
    # --------------------------------------------------------

    df["l1_c_index"] = (
        c_index
    )

    df["l1_C"] = (
        C
    )

    df["n_l1_selected_pcs"] = (
        n_selected
    )

    df[
        "l1_selector_val_average_precision"
    ] = result[
        "val_average_precision"
    ]

    df[
        "l1_selector_val_roc_auc"
    ] = result[
        "val_roc_auc"
    ]

    df[
        "l1_selector_val_f1"
    ] = result[
        "val_f1"
    ]

    df["l1_n_iter"] = (
        result["l1_n_iter"]
    )

    df["l1_selection_path"] = (
        result["path"]
    )

    # Store selected PCs as strings in the CSV.
    df["selected_pc_indices"] = (
        ",".join(
            str(x)
            for x in selected_pcs
        )
    )

    df["selected_pc_numbers"] = (
        ",".join(
            str(x)
            for x in (
                selected_pcs + 1
            )
        )
    )

    return df


# ============================================================
# SAVE COMBINED SUMMARY
# ============================================================

def save_combined_summary(rows):

    if not rows:
        raise ValueError(
            "No L2 result rows were produced."
        )

    summary = pd.concat(
        rows,
        ignore_index=True,
        sort=False,
    )

    # --------------------------------------------------------
    # Sort primarily by number of features so plotting gives
    # a natural sparsity/performance curve.
    #
    # C index is secondary because multiple C values may
    # select the same number of PCs.
    # --------------------------------------------------------

    sort_columns = []

    if "n_features" in summary.columns:
        sort_columns.append(
            "n_features"
        )

    if "l1_c_index" in summary.columns:
        sort_columns.append(
            "l1_c_index"
        )

    if sort_columns:
        summary = (
            summary
            .sort_values(sort_columns)
            .reset_index(drop=True)
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print()
    print("=" * 100)
    print("COMBINED SUMMARY")
    print("=" * 100)

    print(
        summary
    )

    print()
    print(
        "Saved combined summary:"
    )
    print(
        SUMMARY_PATH
    )

    print("=" * 100)

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUNS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load L1 selections
    # --------------------------------------------------------

    results = load_l1_sweep_results(
        C_SWEEP_RESULTS_DIR
    )

    print_l1_sweep_summary(
        results
    )

    # --------------------------------------------------------
    # Keep only sparse representations
    #
    #   n_selected == 0
    #       cannot train a probe
    #
    #   n_selected == 512
    #       equivalent to the complete PCA representation
    #
    # Therefore:
    #
    #       0 < n_selected < 512
    # --------------------------------------------------------

    sparse_results = [
        result
        for result in results
        if (
            0
            < result["n_selected"]
            < TOTAL_N_PCS
        )
    ]

    if not sparse_results:
        raise ValueError(
            "No L1 selections satisfy:\n"
            f"0 < n_selected < {TOTAL_N_PCS}"
        )

    # --------------------------------------------------------
    # Print what will be evaluated
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("L1-SELECTED REPRESENTATIONS TO EVALUATE")
    print("=" * 100)

    for result in sparse_results:

        print(
            f"c_index={result['c_index']:02d} | "
            f"C={result['C']:.8g} | "
            f"n_features={result['n_selected']}"
        )

    print()
    print(
        f"Total L2 regressions: "
        f"{len(sparse_results)}"
    )

    print("=" * 100)
    print()

    # --------------------------------------------------------
    # Run all L2 probes
    # --------------------------------------------------------

    result_tables = []

    for i, result in enumerate(
        sparse_results,
        start=1,
    ):

        print()
        print("#" * 100)

        print(
            f"L2 REGRESSION "
            f"{i}/{len(sparse_results)}"
        )

        print("#" * 100)

        df = run_one_l2_experiment(
            result
        )

        result_tables.append(
            df
        )

        # ----------------------------------------------------
        # Update summary after every successful run.
        #
        # This means that if a later run crashes, all completed
        # experiments are already represented in summary.csv.
        # ----------------------------------------------------

        save_combined_summary(
            result_tables
        )

    # --------------------------------------------------------
    # Final combined result
    # --------------------------------------------------------

    summary = save_combined_summary(
        result_tables
    )

    print()
    print("=" * 100)
    print("ALL L1 -> L2 EXPERIMENTS COMPLETE")
    print("=" * 100)

    print(
        f"Completed L2 regressions: "
        f"{len(sparse_results)}"
    )

    print(
        f"Summary rows: "
        f"{len(summary)}"
    )

    print(
        "Final summary:"
    )

    print(
        SUMMARY_PATH
    )

    print("=" * 100)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()

