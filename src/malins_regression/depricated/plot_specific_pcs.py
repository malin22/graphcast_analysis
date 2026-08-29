import os
import pandas as pd
import matplotlib.pyplot as plt


WEATHER_FEATURE = "TC"
NODE_HIERARCHY_LEVEL = 6

RESULTS_DIR = (
    f"plots/malins_experiments/logistic_regression/"
    f"{WEATHER_FEATURE}/Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    f"PCA_ranked_feature_selection/"
)

RESULTS_PATH = os.path.join(
    RESULTS_DIR,
    "ranked_vs_first_k_validation_results.csv",
)

RAW_RESULTS_PATH = (
    "plots/malins_experiments/"
    f"logistic_regression/{WEATHER_FEATURE}/Node_Hierarchy_Level_M{NODE_HIERARCHY_LEVEL}/"
    "raw_activations/"
    f"logistic_probe_2020_2019_train_2021test_"
    f"intersection_M{NODE_HIERARCHY_LEVEL}_max_3hour.csv"
)

OUT_PATH = os.path.join(
    RESULTS_DIR,
    "f1_vs_k_both_with_raw.png",
)


# ============================================================
# LOAD PCA RESULTS
# ============================================================

df = pd.read_csv(RESULTS_PATH)

ranked = (
    df[df["selection"] == "ranked_top_k"]
    .sort_values("k")
)

first = (
    df[df["selection"] == "first_k"]
    .sort_values("k")
)


# ============================================================
# LOAD RAW-ACTIVATION RESULT
# ============================================================

raw_df = pd.read_csv(RAW_RESULTS_PATH)

raw_f1 = raw_df["test_f1"].iloc[0]

print(f"Raw activation F1: {raw_f1:.4f}")


# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(11, 6))

# Equally spaced x positions
x = range(len(ranked))


# Ranked top-k PCs
ax.plot(
    x,
    ranked["val_f1"],
    marker="o",
    linewidth=2,
    markersize=6,
    label="Ranked top-k PCs",
)


# First-k PCs
ax.plot(
    x,
    first["val_f1"],
    marker="s",
    linewidth=2,
    markersize=6,
    label="First k PCs",
)


# Raw activations
ax.axhline(
    y=raw_f1,
    linestyle=":",
    linewidth=2.5,
    label=f"Raw activations (F1 = {raw_f1:.3f})",
)


# Actual k values
ax.set_xticks(x)

ax.set_xticklabels(
    ranked["k"],
    rotation=45,
    ha="right",
)


ax.set_xlabel("Number of PCs (k)")
ax.set_ylabel("F1 score")
ax.set_title(f"{WEATHER_FEATURE} F1 score vs. Number of Selected PCs")


# Horizontal + vertical grid
ax.grid(
    True,
    alpha=0.3,
    axis="both",
)


ax.legend()

fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

fig.savefig(
    OUT_PATH,
    dpi=300,
    bbox_inches="tight",
)

print(f"Saved plot to: {OUT_PATH}")

plt.show()