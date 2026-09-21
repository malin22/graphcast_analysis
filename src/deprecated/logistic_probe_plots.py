import os
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt


# =====================
# CONFIG
# =====================

WEATHER_FEATURE = "AR"   # "AR" or "TC"
REPRESENTATION = "PCA"   # "PCA" or "raw_activations"
NODE_HIERARCHY_LEVEL = 5
LABEL_MODE = "intersection"
MAX_TIME_DIFFERENCE_HOURS = 3

BASE_DIR = f"plots/malins_experiments/pertubation_experiments/{WEATHER_FEATURE}/{REPRESENTATION}"

NODE_CSV = os.path.join(
    BASE_DIR,
    f"logistic_probe_{LABEL_MODE}_M{NODE_HIERARCHY_LEVEL}_max_{MAX_TIME_DIFFERENCE_HOURS}hour.csv",
)

EVENT_PATTERN = os.path.join(
    BASE_DIR,
    f"event_region_metrics_{LABEL_MODE}_M{NODE_HIERARCHY_LEVEL}_*_features_max_{MAX_TIME_DIFFERENCE_HOURS}hour.csv",
)

OUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# =====================
# HELPERS
# =====================

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def load_node_results():
    df = pd.read_csv(NODE_CSV)

    if "n_pcs" in df.columns and "n_features" not in df.columns:
        df = df.rename(columns={"n_pcs": "n_features"})

    df = df.sort_values("n_features")
    return df


def load_event_results():
    files = sorted(glob(EVENT_PATTERN))

    if not files:
        raise FileNotFoundError(f"No event files found with pattern:\n{EVENT_PATTERN}")

    parts = []
    for f in files:
        tmp = pd.read_csv(f)
        parts.append(tmp)

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["n_features", "threshold", "event_id"])
    return df


# =====================
# PLOT 1: NODE-LEVEL AP/AUC
# =====================

def plot_node_level(df):
    plt.figure(figsize=(7, 4.5))

    plt.plot(
        df["n_features"],
        df["average_precision"],
        marker="o",
        linewidth=2,
        label="Average precision",
    )

    plt.plot(
        df["n_features"],
        df["roc_auc"],
        marker="o",
        linewidth=2,
        label="ROC-AUC",
    )

    baseline = df["test_positive_rate"].iloc[0]
    plt.axhline(
        baseline,
        linestyle="--",
        linewidth=1.5,
        label=f"Random AP baseline ({baseline:.4f})",
    )

    plt.xscale("log")
    plt.xticks(
        df["n_features"],
        labels=[str(x) for x in df["n_features"]],
    )

    plt.xlabel("Number of PCs / features")
    plt.ylabel("Score")
    plt.title(f"{WEATHER_FEATURE} node-level linear decoding")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()

    savefig(f"{WEATHER_FEATURE}_node_level_AP_AUC_vs_features.png")


# =====================
# PLOT 2: EVENT COVERAGE VS FEATURES
# =====================

def plot_event_metric_vs_features(event_df, metric, ylabel, threshold=0.5):
    df = (
        event_df[event_df["threshold"] == threshold]
        .groupby("n_features")[metric]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("n_features")
    )

    plt.figure(figsize=(7, 4.5))

    plt.errorbar(
        df["n_features"],
        df["mean"],
        yerr=df["std"],
        marker="o",
        linewidth=2,
        capsize=3,
    )

    plt.xscale("log")
    plt.xticks(
        df["n_features"],
        labels=[str(x) for x in df["n_features"]],
    )

    plt.xlabel("Number of PCs / features")
    plt.ylabel(ylabel)
    plt.title(f"{WEATHER_FEATURE} event-level {ylabel} at threshold {threshold}")
    plt.grid(True, alpha=0.3)

    if metric in ["coverage_recall", "precision", "iou", "event_found"]:
        plt.ylim(0, 1.02)

    savefig(f"{WEATHER_FEATURE}_event_{metric}_vs_features_threshold_{threshold}.png")


# =====================
# PLOT 3: THRESHOLD TRADEOFF AT BEST FEATURE COUNT
# =====================

def plot_threshold_tradeoff(event_df):
    max_features = event_df["n_features"].max()

    df = (
        event_df[event_df["n_features"] == max_features]
        .groupby("threshold")[["coverage_recall", "precision", "iou"]]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(7, 4.5))

    for metric, label in [
        ("coverage_recall", "Coverage / recall"),
        ("precision", "Precision"),
        ("iou", "IoU"),
    ]:
        plt.plot(
            df["threshold"],
            df[metric],
            marker="o",
            linewidth=2,
            label=label,
        )

    plt.xlabel("Probability threshold")
    plt.ylabel("Mean event-level score")
    plt.title(f"{WEATHER_FEATURE} event-level threshold tradeoff ({max_features} features)")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()

    savefig(f"{WEATHER_FEATURE}_event_threshold_tradeoff_{max_features}_features.png")


# =====================
# PLOT 4: AREA RATIO VS FEATURES
# =====================

def plot_area_ratio(event_df, threshold=0.5):
    df = (
        event_df[event_df["threshold"] == threshold]
        .groupby("n_features")["area_ratio"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values("n_features")
    )

    plt.figure(figsize=(7, 4.5))

    plt.errorbar(
        df["n_features"],
        df["mean"],
        yerr=df["std"],
        marker="o",
        linewidth=2,
        capsize=3,
    )

    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        label="Perfect area match",
    )

    plt.xscale("log")
    plt.xticks(
        df["n_features"],
        labels=[str(x) for x in df["n_features"]],
    )

    plt.xlabel("Number of PCs / features")
    plt.ylabel("Predicted area / true area")
    plt.title(f"{WEATHER_FEATURE} predicted region size at threshold {threshold}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    savefig(f"{WEATHER_FEATURE}_event_area_ratio_vs_features_threshold_{threshold}.png")


# =====================
# PLOT 5: EVENT-LEVEL DISTRIBUTION
# =====================

def plot_event_distribution(event_df, metric="iou", threshold=0.5):
    max_features = event_df["n_features"].max()

    df = event_df[
        (event_df["n_features"] == max_features)
        & (event_df["threshold"] == threshold)
    ]

    plt.figure(figsize=(7, 4.5))

    plt.hist(df[metric].dropna(), bins=20)

    plt.xlabel(metric)
    plt.ylabel("Number of events")
    plt.title(
        f"{WEATHER_FEATURE} event-level {metric} distribution "
        f"({max_features} features, threshold {threshold})"
    )
    plt.grid(True, alpha=0.3)

    savefig(f"{WEATHER_FEATURE}_event_{metric}_hist_threshold_{threshold}.png")


# =====================
# MAIN
# =====================

def main():
    node_df = load_node_results()
    event_df = load_event_results()

    print("Node-level results:")
    print(node_df)

    print("\nEvent-level files loaded:")
    print(event_df[["n_features", "threshold", "event_id"]].head())

    plot_node_level(node_df)

    for metric, ylabel in [
        ("coverage_recall", "Coverage / recall"),
        ("precision", "Precision"),
        ("iou", "IoU"),
    ]:
        plot_event_metric_vs_features(
            event_df,
            metric=metric,
            ylabel=ylabel,
            threshold=0.5,
        )

    plot_area_ratio(event_df, threshold=0.5)
    plot_threshold_tradeoff(event_df)
    plot_event_distribution(event_df, metric="iou", threshold=0.5)
    plot_event_distribution(event_df, metric="coverage_recall", threshold=0.5)


if __name__ == "__main__":
    main()