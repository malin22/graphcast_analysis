#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

OUT_FILE = Path(
    "/share/prj-4d/graphcast_shared/data/pca_components/512_PCs/"
    "layer8_only/rerun_for_cumulative_explained_variance_plot/"
    "2019_2020_onto_2021_explained_var_pca.out"
)

PER_LAYER_PCA_DIR = Path(
    "/share/prj-4d/graphcast_shared/data/pca_components/"
    "512_PCs/per_layer/layerwise_2019"
)

LAYERS = range(16)
THRESHOLDS = (0.50, 0.75, 0.90)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cumulative_variance_from_out(out_file: Path) -> np.ndarray:
    """Extract the final cumulative-variance NumPy list from a .out file."""
    text = out_file.read_text()

    matches = re.findall(
        r"Cumulative explained variance:\s*\n\s*(\[[\s\S]*?\])",
        text,
    )

    if not matches:
        raise ValueError(
            f"Could not find cumulative explained variance in {out_file}"
        )

    array_text = matches[-1]
    cumulative = np.fromstring(array_text.strip()[1:-1], sep=" ")

    if cumulative.size == 0:
        raise ValueError("Found a variance block but parsed no values.")

    return cumulative


def load_per_layer_variance_ratios(pca_dir: Path, layers):
    """
    Load per-component explained variance ratios for every independent layer PCA.
    """
    ratios_by_layer = {}

    for layer in layers:
        path = pca_dir / (
            f"pca_explained_variance_ratio_layer{layer:04d}.npy"
        )

        if not path.exists():
            raise FileNotFoundError(f"Missing variance-ratio file: {path}")

        ratios_by_layer[layer] = np.load(path)

    return ratios_by_layer


# ---------------------------------------------------------------------------
# Single-PCA plot: reads the cumulative values from the .out file
# ---------------------------------------------------------------------------

def plot_single_cumulative_variance(cumulative: np.ndarray, output_path: Path):
    components = np.arange(1, len(cumulative) + 1)

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(
        components,
        cumulative,
        color="#2166ac",
        linewidth=3.0,
    )

    for threshold in THRESHOLDS:
        ax.axhline(
            threshold,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.65,
        )

        first_index = np.searchsorted(cumulative, threshold)

        if first_index < len(cumulative):
            pc_number = first_index + 1

            ax.scatter(
                pc_number,
                cumulative[first_index],
                color="#b2182b",
                s=48,
                zorder=3,
            )

            ax.annotate(
                f"{threshold:.0%} at PC {pc_number}",
                xy=(pc_number, cumulative[first_index]),
                xytext=(9, 9),
                textcoords="offset points",
                fontsize=13,
                fontweight="bold",
                color="#b2182b",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                },
            )

    ax.set_xlabel("Number of principal components", fontsize=22, labelpad=12)
    ax.set_ylabel("Cumulative explained variance", fontsize=22, labelpad=12)
    ax.set_title(
        "Cumulative explained variance of layer-8 PCA",
        fontsize=27,
        pad=18,
    )

    ax.set_xlim(1, len(cumulative))
    ax.set_ylim(0, min(1.01, max(0.05, cumulative[-1] + 0.03)))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    ax.tick_params(axis="both", labelsize=17, length=6, width=1.2)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved single-PCA plot: {output_path}")


# ---------------------------------------------------------------------------
# Per-layer plot: reads explained-variance-ratio .npy files
# ---------------------------------------------------------------------------

def plot_per_layer_cumulative_variance(
    variance_ratios_by_layer: dict,
    output_path: Path,
):
    fig, ax = plt.subplots(figsize=(14, 9))

    layers = sorted(variance_ratios_by_layer)
    colours = plt.cm.tab20(np.linspace(0, 1, len(layers)))

    max_components = 0

    for colour, layer in zip(colours, layers):
        cumulative = np.cumsum(variance_ratios_by_layer[layer])
        max_components = max(max_components, len(cumulative))

        ax.plot(
            np.arange(1, len(cumulative) + 1),
            cumulative,
            label=f"Layer {layer}",
            linewidth=2.1,
            color=colour,
        )

    # Threshold lines are labelled once to keep a 16-layer figure readable.
    for threshold in THRESHOLDS:
        ax.axhline(
            threshold,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            zorder=0,
        )

        ax.text(
            1.005,
            threshold,
            f"{threshold:.0%}",
            transform=ax.get_yaxis_transform(),
            fontsize=15,
            fontweight="bold",
            va="center",
            ha="left",
            color="dimgray",
            clip_on=False,
        )

    ax.set_xlabel("Number of principal components", fontsize=22, labelpad=12)
    ax.set_ylabel("Cumulative explained variance", fontsize=22, labelpad=12)
    ax.set_title(
        "Cumulative explained variance across independently fitted layer PCAs",
        fontsize=26,
        pad=18,
    )

    ax.set_xlim(1, max_components)
    ax.set_ylim(0, 1.01)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    ax.tick_params(axis="both", labelsize=17, length=6, width=1.2)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        ncol=4,
        fontsize=12,
        frameon=False,
        loc="lower right",
        title="PCA basis",
        title_fontsize=13,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved per-layer PCA plot: {output_path}")


# ---------------------------------------------------------------------------
# Run both plots
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # # Existing layer-8 plot based on the logged cumulative list.
    # single_cumulative = load_cumulative_variance_from_out(OUT_FILE)

    # plot_single_cumulative_variance(
    #     cumulative=single_cumulative,
    #     output_path=OUT_FILE.with_name(
    #         "cumulative_explained_variance_2019_2020_onto_2021.png"
    #     ),
    # )

    # New plot based on per-layer ratio files.
    per_layer_ratios = load_per_layer_variance_ratios(
        pca_dir=PER_LAYER_PCA_DIR,
        layers=LAYERS,
    )

    plot_per_layer_cumulative_variance(
        variance_ratios_by_layer=per_layer_ratios,
        output_path=PER_LAYER_PCA_DIR / (
            "per_layer_cumulative_explained_variance_2019_on_2019.png"
        ),
    )