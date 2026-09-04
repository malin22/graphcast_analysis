import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

WEATHER_FEATURE = "AR"          # "TC" or "AR"
HIERARCHY_LEVEL = 6

RESULTS_BASE = Path(
    f"results/logistic_regression/"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{HIERARCHY_LEVEL}"
)

FOLDER = "all"


PLOTS_DIR = Path(
    f"malins_plots/logistic_regression/"
    f"{WEATHER_FEATURE}/"
    f"Node_Hierarchy_Level_M{HIERARCHY_LEVEL}/"
    f"probe_comparison/{FOLDER}/"
)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# Choose which experiment families appear in the plots.
# Set include=False to hide one.
#
# "kind":
#   "curve"    -> results contain several feature counts
#   "baseline" -> one horizontal reference line
#   "point"    -> one result at its actual selected feature count
#
# Update "relative_path" if one of your experiment runners writes
# its summary CSV under a slightly different filename.
EXPERIMENTS = {
    "raw": {
        "include": True,
        "label": "Raw activations",
        "kind": "baseline",
        "relative_path": "raw_activations/logistic_probe_AR_raw_activations_intersection_M6_max_3hour.csv",
    },
    "first_pcs": {
        "include": True,
        "label": "First-k PCs",
        "kind": "curve",
        "relative_path": "first_k_pcs/logistic_probe_AR_first_k_pcs_intersection_M6_max_3hour.csv",
    },
    "selected_pcs": {
        "include": True,
        "label": "Selected PCs (L2 coefficients)",
        "kind": "curve",
        "relative_path": "selected_pcs_after_coefs/logistic_probe_AR_selected_pcs_after_coefs_intersection_M6_max_3hour.csv",
    },
    "selected_raw": {
        "include": True,
        "label": "Selected raw features (L2 coefficients)",
        "kind": "curve",
        "relative_path": "selected_raw_acts_after_coefs/logistic_probe_AR_selected_raw_acts_after_coefs_intersection_M6_max_3hour.csv",
    },
    "l1_selected_pcs": {
        "include": False,
        "label": "L1-selected PCs + L2 probe",
        "kind": "point",
        "relative_path": "l1_selected_pcs_l2/summary.csv",
    },
}


METRICS = {
    "test_average_precision": {
        "ylabel": "Average Precision (AP)",
        "title": "Average Precision",
        "filename": "average_precision_vs_features.png",
    },
    "test_roc_auc": {
        "ylabel": "ROC-AUC",
        "title": "ROC-AUC",
        "filename": "roc_auc_vs_features.png",
    },
    "test_f1": {
        "ylabel": "F1 score",
        "title": "F1 score",
        "filename": "f1_vs_features.png",
    },
}


# ============================================================
# HELPERS
# ============================================================


def normalize_result_table(df, experiment_name):
    """
    Keep only the columns needed for plotting and make sure
    n_features is available.
    """
    df = df.copy()

    if "n_features" not in df.columns:
        # Some result tables may use a more specific name.
        alternatives = [
            "n_pcs",
            "n_selected",
            "feature_count",
            "n_selected_features",
        ]
        found = next((c for c in alternatives if c in df.columns), None)

        if found is None:
            raise ValueError(
                f"{experiment_name}: could not find feature-count column. "
                f"Available columns: {list(df.columns)}"
            )

        df["n_features"] = df[found]

    df["n_features"] = pd.to_numeric(df["n_features"], errors="coerce")
    df = df.dropna(subset=["n_features"])
    df["n_features"] = df["n_features"].astype(int)

    return df.sort_values("n_features")


def load_experiments():
    loaded = {}

    print(f"Results base: {RESULTS_BASE}\n")

    for name, config in EXPERIMENTS.items():
        if not config["include"]:
            continue

        path = RESULTS_BASE / config['relative_path']

        if path is None:
            print(
                f"[skip] {config['label']}: could not locate result CSV "
                f"under {RESULTS_BASE / Path(config['relative_path']).parent}"
            )
            continue

        try:
            df = pd.read_csv(path)
            df = normalize_result_table(df, name)
        except Exception as exc:
            print(f"[skip] {config['label']}: {exc}")
            continue

        loaded[name] = {
            **config,
            "path": path,
            "data": df,
        }

        print(
            f"[load] {config['label']}: {path} "
            f"({len(df)} row{'s' if len(df) != 1 else ''})"
        )

    if not loaded:
        raise RuntimeError("No experiment result tables could be loaded.")

    return loaded


def plot_metric(loaded, metric, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))

    plotted_anything = False
    all_feature_counts = []

    for experiment in loaded.values():
        df = experiment["data"]

        if metric not in df.columns:
            print(
                f"[skip metric] {experiment['label']}: "
                f"column '{metric}' not found"
            )
            continue

        valid = df[["n_features", metric]].dropna().copy()
        if valid.empty:
            continue

        x = valid["n_features"].to_numpy()
        y = valid[metric].to_numpy()

        kind = experiment["kind"]
        label = experiment["label"]

        if kind == "baseline":
            # A raw-activation experiment normally has one row.
            # Treat it as a reference value across the whole plot.
            raw_value = float(y[0])
            n_features = int(x[0])

            ax.axhline(
                raw_value,
                linestyle="--",
                linewidth=2,
                label=f"{label} ({n_features} dims)",
            )
            all_feature_counts.append(n_features)

        elif kind == "point":
            ax.scatter(
                x,
                y,
                s=90,
                marker="D",
                label=label,
                zorder=4,
            )
            all_feature_counts.extend(x.tolist())

        elif kind == "curve":
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                label=label,
            )
            all_feature_counts.extend(x.tolist())

        else:
            raise ValueError(f"Unknown plot kind: {kind}")

        plotted_anything = True

    if not plotted_anything:
        plt.close(fig)
        print(f"[skip plot] No usable data for {metric}")
        return

    # Log x-axis is useful for feature-count sweeps such as
    # 5, 10, 25, ..., 512.
    positive_counts = sorted({x for x in all_feature_counts if x > 0})

    if len(positive_counts) > 1:
        ax.set_xscale("log")
        ax.set_xticks(positive_counts)
        ax.set_xticklabels([str(x) for x in positive_counts])

    ax.set_xlabel("Number of latent features")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{WEATHER_FEATURE} — {title}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    loaded = load_experiments()

    print("\nIncluded experiments:")
    for experiment in loaded.values():
        print(f"  - {experiment['label']}")

    print()

    for metric, config in METRICS.items():
        plot_metric(
            loaded=loaded,
            metric=metric,
            ylabel=config["ylabel"],
            title=config["title"],
            filename=config["filename"],
        )


if __name__ == "__main__":
    main()
