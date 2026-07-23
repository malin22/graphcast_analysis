import os
import pandas as pd
import matplotlib.pyplot as plt


# =====================
# CONFIG
# =====================

WEATHER_FEATURE = "TC"  # "AR" or "TC"
HIERARCHY_LEVEL = 6  # Node hierarchy level (1-6) for PCA scores

BASE_PATH = f"plots/malins_experiments/logistic_regression/{WEATHER_FEATURE}/Node_Hierarchy_Level_M{HIERARCHY_LEVEL}"

PCA_CSV = os.path.join(BASE_PATH, "PCA", f"logistic_probe_2020train_2021test_intersection_M{HIERARCHY_LEVEL}_max_3hour.csv")
RAW_CSV = os.path.join(BASE_PATH, "raw_activations", f"logistic_probe_2020train_2021test_intersection_M{HIERARCHY_LEVEL}_max_3hour.csv")

OUT_DIR = os.path.join(BASE_PATH, "figures")
os.makedirs(OUT_DIR, exist_ok=True)


# =====================
# LOAD
# =====================

pca = pd.read_csv(PCA_CSV)
raw = pd.read_csv(RAW_CSV)

raw_ap = raw["test_average_precision"].iloc[0]
raw_auc = raw["test_roc_auc"].iloc[0]
raw_f1 = raw["test_f1"].iloc[0]
raw_n_features = raw["n_features"].iloc[0]


# =====================
# PLOT FUNCTION
# =====================

def plot_metric(metric, ylabel, title, raw_value, out_name):
    plt.figure(figsize=(7, 4.5))

    plt.plot(
        pca["n_features"],
        pca[metric],
        marker="o",
        linewidth=2,
        label="PCA scores",
    )

    plt.axhline(
        raw_value,
        linestyle="--",
        linewidth=2,
        label=f"Raw activations ({raw_n_features} dims)",
    )

    plt.xscale("log")
    plt.xticks(
        pca["n_features"],
        labels=[str(x) for x in pca["n_features"]],
    )

    plt.xlabel("Number of PCs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_path = os.path.join(OUT_DIR, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")


# =====================
# MAKE PLOTS
# =====================

positive_fraction = raw["test_positive_rate"].iloc[0]
positive_percent = 100 * positive_fraction
positives_per_10000 = positive_fraction * 10000

plot_metric(
    metric="test_average_precision",
    ylabel="Average Precision (AP)",
    title=f"{WEATHER_FEATURE} Average Precision PCs vs Raw Activations",
    raw_value=raw_ap,
    out_name=f"{WEATHER_FEATURE.lower()}_average_precision_vs_pcs.png",
)

plot_metric(
    metric="test_roc_auc",
    ylabel="ROC-AUC",
    title=f"{WEATHER_FEATURE} ROC-AUC PCs vs Raw Activations",
    raw_value=raw_auc,
    out_name=f"{WEATHER_FEATURE.lower()}_auc_vs_pcs.png",
)

plot_metric(
    metric="test_f1",
    ylabel="F1 Score",
    title=f"{WEATHER_FEATURE} F1 Score PCs vs Raw Activations \n"
    f"Test set positive fraction: {positive_percent:.3f}%",
    raw_value=raw_f1,
    out_name=f"{WEATHER_FEATURE.lower()}_f1_vs_pcs.png",
)