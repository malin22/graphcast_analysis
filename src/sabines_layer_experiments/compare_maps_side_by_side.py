#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


LAYER8_DIR = Path(
    "/home/student/s/sascholle/share/graphcast_analysis/plots/2021_pca_projected_on_2021"
)

ALL_LAYERS_DIR = Path(
    "/home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/layer_analysis/AllLayersJanuaryPCs"
)

OUT_DIR = Path(
    "/home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/layer_analysis/pc_basis_comparison"
)

N_LAYER8_PCS = 20
N_ALL_LAYER_PCS = 11


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def image_for_similarity(path, size=(256, 128)):
    """
    Load image, convert to grayscale, resize, flatten, and normalize.

    This is a rough PNG-level comparison. It includes map content plus some
    figure artifacts unless the input images were saved cleanly.
    """
    img = Image.open(path).convert("L")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32).ravel()

    arr = arr - arr.mean()
    norm = np.linalg.norm(arr)
    if norm < 1e-8:
        return arr
    return arr / norm


def cosine(a, b):
    return float(np.dot(a, b))


def layer8_path(pc):
    return LAYER8_DIR / f"pc{pc}_mean_activation_map_year.png"


def all_layers_path(pc):
    return ALL_LAYERS_DIR / f"PC{pc}_layers_grid.png"


def check_inputs():
    missing = []

    for pc in range(1, N_LAYER8_PCS + 1):
        p = layer8_path(pc)
        if not p.exists():
            missing.append(p)

    for pc in range(1, N_ALL_LAYER_PCS + 1):
        p = all_layers_path(pc)
        if not p.exists():
            missing.append(p)

    if missing:
        print("Missing files:")
        for p in missing:
            print(p)
        raise FileNotFoundError("Some expected PC plot files are missing.")


def make_same_index_side_by_side():
    """
    Compare PC1 old-vs-new, PC2 old-vs-new, ... PC11 old-vs-new.
    """
    n = N_ALL_LAYER_PCS
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, 4.0 * n))

    for row, pc in enumerate(range(1, n + 1)):
        left_img = load_rgb(layer8_path(pc))
        right_img = load_rgb(all_layers_path(pc))

        axes[row, 0].imshow(left_img)
        axes[row, 0].set_title(f"Layer8-only PCA: PC{pc}", fontsize=12)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(right_img)
        axes[row, 1].set_title(f"All-layer PCA: PC{pc}", fontsize=12)
        axes[row, 1].axis("off")

    fig.suptitle(
        "Same-index comparison: layer8-only PCA vs all-layer PCA",
        fontsize=16,
        y=0.995,
    )
    plt.tight_layout()
    out = OUT_DIR / "side_by_side_same_pc_index.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def compute_png_similarity_matrix():
    layer8_vecs = [
        image_for_similarity(layer8_path(pc))
        for pc in range(1, N_LAYER8_PCS + 1)
    ]

    all_layer_vecs = [
        image_for_similarity(all_layers_path(pc))
        for pc in range(1, N_ALL_LAYER_PCS + 1)
    ]

    sim = np.zeros((N_ALL_LAYER_PCS, N_LAYER8_PCS), dtype=np.float32)

    for i, a in enumerate(all_layer_vecs):
        for j, b in enumerate(layer8_vecs):
            sim[i, j] = cosine(a, b)

    return sim


def plot_similarity_heatmap(sim):
    plt.figure(figsize=(14, 6))
    im = plt.imshow(sim, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")

    plt.yticks(
        np.arange(N_ALL_LAYER_PCS),
        [f"All-layer PC{i}" for i in range(1, N_ALL_LAYER_PCS + 1)],
    )
    plt.xticks(
        np.arange(N_LAYER8_PCS),
        [f"L8 PC{i}" for i in range(1, N_LAYER8_PCS + 1)],
        rotation=90,
    )

    plt.colorbar(im, label="Rough PNG cosine similarity")
    plt.title("Image-level similarity: all-layer PCA maps vs layer8-only PCA maps")
    plt.xlabel("Layer8-only PCA maps")
    plt.ylabel("All-layer PCA maps")

    for i in range(N_ALL_LAYER_PCS):
        best_j = int(np.nanargmax(sim[i]))
        plt.text(
            best_j,
            i,
            "*",
            ha="center",
            va="center",
            color="black",
            fontsize=16,
            fontweight="bold",
        )

    plt.tight_layout()
    out = OUT_DIR / "png_similarity_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def make_best_match_plot(sim):
    best_matches = np.nanargmax(sim, axis=1)

    fig, axes = plt.subplots(
        nrows=N_ALL_LAYER_PCS,
        ncols=2,
        figsize=(12, 4.0 * N_ALL_LAYER_PCS),
    )

    for row in range(N_ALL_LAYER_PCS):
        all_pc = row + 1
        layer8_pc = int(best_matches[row]) + 1
        score = sim[row, layer8_pc - 1]

        left_img = load_rgb(all_layers_path(all_pc))
        right_img = load_rgb(layer8_path(layer8_pc))

        axes[row, 0].imshow(left_img)
        axes[row, 0].set_title(f"All-layer PCA: PC{all_pc}", fontsize=12)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(right_img)
        axes[row, 1].set_title(
            f"Best layer8-only match: PC{layer8_pc}  |  PNG sim={score:.2f}",
            fontsize=12,
        )
        axes[row, 1].axis("off")

    fig.suptitle(
        "Best visual matches between all-layer PCs and layer8-only PCs",
        fontsize=16,
        y=0.995,
    )
    plt.tight_layout()
    out = OUT_DIR / "best_png_matches.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    check_inputs()

    make_same_index_side_by_side()

    sim = compute_png_similarity_matrix()
    np.save(OUT_DIR / "png_similarity_matrix.npy", sim)

    plot_similarity_heatmap(sim)
    make_best_match_plot(sim)

    print("\nBest matches:")
    for row in range(N_ALL_LAYER_PCS):
        best_j = int(np.nanargmax(sim[row]))
        print(
            f"All-layer PC{row + 1:02d} best matches "
            f"layer8-only PC{best_j + 1:02d} "
            f"(PNG sim={sim[row, best_j]:.3f})"
        )


if __name__ == "__main__":
    main()