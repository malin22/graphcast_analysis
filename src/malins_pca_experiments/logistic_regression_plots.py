import os
import pandas as pd
import matplotlib.pyplot as plt


# =====================
# CONFIG
# =====================

WEATHER_FEATURE = "TC"  # "AR" or "TC"

PCA_CSV = f"plots/malins_experiments/2021_logistic_probe/{WEATHER_FEATURE}/PCA/logistic_probe_intersection_M5_max_3hour.csv"
RAW_CSV = f"plots/malins_experiments/2021_logistic_probe/{WEATHER_FEATURE}/raw_activations/logistic_probe_intersection_M5_max_3hour.csv"

OUT_DIR = f"plots/malins_experiments/2021_logistic_probe/{WEATHER_FEATURE}/figures"
os.makedirs(OUT_DIR, exist_ok=True)


# =====================
# LOAD
# =====================

pca = pd.read_csv(PCA_CSV)
raw = pd.read_csv(RAW_CSV)

raw_ap = raw["average_precision"].iloc[0]
raw_auc = raw["roc_auc"].iloc[0]
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

plot_metric(
    metric="average_precision",
    ylabel="Average Precision (AP)",
    title=f"{WEATHER_FEATURE} decodability from GraphCast latent PCs",
    raw_value=raw_ap,
    out_name=f"{WEATHER_FEATURE.lower()}_average_precision_vs_pcs.png",
)

plot_metric(
    metric="roc_auc",
    ylabel="ROC-AUC",
    title=f"{WEATHER_FEATURE} ROC-AUC from GraphCast latent PCs",
    raw_value=raw_auc,
    out_name=f"{WEATHER_FEATURE.lower()}_auc_vs_pcs.png",
)