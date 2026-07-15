from __future__ import annotations

from pathlib import Path

import gc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from graphcast import icosahedral_mesh


JAN_ACTS_DIR = Path(
    "/share/prj-4d/graphcast_shared/data/graphcast_activations_all_layers_January_2021"
)

LAYER_PCA_COMPONENTS_JAN = {
    0: Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021_Jan_layer0.npy"),
    8: Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021_Jan_layer8.npy"),
    15: Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021_Jan_layer15.npy"),
}

LAYER_PCA_MEANS_JAN = {
    0: Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_mean_2021_Jan_layer0.npy"),
    8: Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_mean_2021_Jan_layer8.npy"),
    15: Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_mean_2021_Jan_layer15.npy"),
}

YEAR_LAYER8_PLOT_DIR = Path("/home/student/s/sascholle/share/graphcast_analysis/plots/2021_pca_projected_on_2021")

LAYER_COMPARE_OUT_DIR = Path(
    "/home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/layer_analysis/pc_basis_comparison/layer0_vs_8_vs_15"
)

LAYERS_TO_COMPARE = [0, 8, 15]
N_LAYER_COMPARE_PCS = 20


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def normalize_rows(X, eps=1e-12):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def signed_same_index_cosines(components_a, components_b):
    A = normalize_rows(components_a)
    B = normalize_rows(components_b)
    return np.sum(A * B, axis=1)


def load_activation_matrix(path: Path) -> np.ndarray:
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"Expected [nodes, features], got {x.shape} for {path.name}")

    return x


def get_mesh_latlon(splits: int = 6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    vertices = meshes[splits].vertices
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    return lat, lon


def plot_pc_map(scores, lat, lon, out_path, title):
    vmax = np.nanpercentile(np.abs(scores), 99)
    vmax = max(vmax, 1e-6)

    plt.figure(figsize=(12, 6))
    sc = plt.scatter(
        lon,
        lat,
        c=scores,
        s=2,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0,
    )
    plt.colorbar(sc, label="PC score")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_layer_mean_pc_maps(layer: int, n_pcs: int = N_LAYER_COMPARE_PCS):
    """
    Project January activations from one mesh_gnn layer onto that layer's own PCA basis.

    Returns:
      mean_scores: [nodes, n_pcs]
    """
    components = np.load(LAYER_PCA_COMPONENTS_JAN[layer])[:n_pcs].astype(np.float32)
    mean = np.load(LAYER_PCA_MEANS_JAN[layer]).astype(np.float32)

    files = sorted(
        JAN_ACTS_DIR.glob(
            f"layer{layer:04d}_mesh_gnn_post_res_nodes_mesh_nodes_t*.npy"
        )
    )

    if not files:
        raise FileNotFoundError(f"No activation files found for layer {layer} in {JAN_ACTS_DIR}")

    print(f"Layer {layer} from January 2021: found {len(files)} activation files")

    score_sum = None
    valid_count = 0

    for i, path in enumerate(files, start=1):
        X = load_activation_matrix(path)

        if np.isnan(X).any():
            print(f"WARNING: skipping {path.name}, contains NaNs")
            continue

        if X.shape[1] != mean.shape[0]:
            raise ValueError(
                f"Feature mismatch for {path.name}: activations {X.shape[1]}, mean {mean.shape[0]}"
            )

        scores = (X - mean) @ components.T  # [nodes, n_pcs]

        if score_sum is None:
            score_sum = np.zeros_like(scores, dtype=np.float64)

        score_sum += scores
        valid_count += 1

        if i % 25 == 0:
            print(f"  layer {layer}: processed {i}/{len(files)} files")

        del X, scores
        gc.collect()

    if valid_count == 0:
        raise ValueError(f"No valid files for layer {layer}")

    mean_scores = (score_sum / valid_count).astype(np.float32)
    print(f"Layer {layer}: averaged {valid_count} valid files")
    return mean_scores


def make_year_layer8_plot_paths(n_pcs: int = N_LAYER_COMPARE_PCS):
    paths = []
    for pc in range(1, n_pcs + 1):
        p = YEAR_LAYER8_PLOT_DIR / f"pc{pc}_mean_activation_map_year.png"
        if not p.exists():
            raise FileNotFoundError(f"Missing year layer-8 plot: {p}")
        paths.append(p)
    return paths


def plot_layer0_8_15_overview(layer_mean_scores_jan, year_layer8_plot_paths, lat, lon):
    """
    Left three columns:
      January 2021 mean PC maps for layers 0, 8, 15.

    Right column:
      Pre-rendered full-year layer 8 mean activation map images for PC1..PC20.
    """
    components_jan = {
        layer: np.load(LAYER_PCA_COMPONENTS_JAN[layer])[:N_LAYER_COMPARE_PCS].astype(np.float64)
        for layer in LAYERS_TO_COMPARE
    }

    cos_0_8 = signed_same_index_cosines(components_jan[0], components_jan[8])
    cos_8_15 = signed_same_index_cosines(components_jan[8], components_jan[15])
    cos_0_15 = signed_same_index_cosines(components_jan[0], components_jan[15])

    all_vals = np.concatenate(
        [layer_mean_scores_jan[layer][:, :N_LAYER_COMPARE_PCS].ravel() for layer in LAYERS_TO_COMPARE]
    )
    vmax = np.nanpercentile(np.abs(all_vals), 99)
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(
        nrows=N_LAYER_COMPARE_PCS,
        ncols=4,
        figsize=(20, 3.0 * N_LAYER_COMPARE_PCS),
        sharex=True,
        sharey=True,
    )

    for row in range(N_LAYER_COMPARE_PCS):
        pc = row + 1

        for col, layer in enumerate(LAYERS_TO_COMPARE):
            ax = axes[row, col]
            scores = layer_mean_scores_jan[layer][:, row]

            sc = ax.scatter(
                lon,
                lat,
                c=scores,
                s=1,
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
                linewidths=0,
            )

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(f"Jan 2021 layer {layer}", fontsize=13)

        ax = axes[row, 3]
        img = load_rgb(year_layer8_plot_paths[row])
        ax.imshow(img)
        ax.axis("off")
        if row == 0:
            ax.set_title("Full 2021 layer 8", fontsize=13)

        axes[row, 0].set_ylabel(
            (
                f"PC{pc}\n"
                f"L0-L8 {cos_0_8[row]:+.2f}\n"
                f"L8-L15 {cos_8_15[row]:+.2f}\n"
                f"L0-L15 {cos_0_15[row]:+.2f}"
            ),
            fontsize=9,
            rotation=0,
            labelpad=55,
            va="center",
        )

    fig.suptitle(
        "January 2021 mean PC maps for layers 0, 8, 15 vs full-2021 layer 8 plots",
        fontsize=16,
        y=0.995,
    )

    cbar = fig.colorbar(sc, ax=axes[:, :3].ravel().tolist(), shrink=0.6)
    cbar.set_label("PC score")

    out = LAYER_COMPARE_OUT_DIR / "layer0_8_15_jan_vs_full_year_layer8_mean_pc_maps.png"
    LAYER_COMPARE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=250, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def make_layer_mean_maps_and_overview():
    """
    Creates:
      - individual January PC maps for layers 0, 8, 15
      - one overview figure with January layers 0, 8, 15 vs pre-rendered full-year layer 8 plots
    """
    LAYER_COMPARE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    lat, lon = get_mesh_latlon(splits=6)

    year_layer8_plot_paths = make_year_layer8_plot_paths(N_LAYER_COMPARE_PCS)

    layer_mean_scores_jan = {}

    for layer in LAYERS_TO_COMPARE:
        layer_dir = LAYER_COMPARE_OUT_DIR / f"layer{layer}_jan"
        layer_dir.mkdir(parents=True, exist_ok=True)

        mean_scores = compute_layer_mean_pc_maps(layer, n_pcs=N_LAYER_COMPARE_PCS)

        if mean_scores.shape[0] != len(lat):
            raise ValueError(
                f"Layer {layer} node mismatch: scores have {mean_scores.shape[0]} nodes, "
                f"mesh has {len(lat)} nodes"
            )

        layer_mean_scores_jan[layer] = mean_scores
        np.save(layer_dir / f"layer{layer}_jan_mean_pc_scores.npy", mean_scores)

        for pc_idx in range(N_LAYER_COMPARE_PCS):
            pc = pc_idx + 1
            plot_pc_map(
                mean_scores[:, pc_idx],
                lat,
                lon,
                layer_dir / f"PC{pc}_mean_activation_map_Jan.png",
                title=f"January 2021 layer {layer} PCA: PC{pc} mean map",
            )

    plot_layer0_8_15_overview(layer_mean_scores_jan, year_layer8_plot_paths, lat, lon)


RUN = True


def main():
    if RUN:
        make_layer_mean_maps_and_overview()


if __name__ == "__main__":
    main()
