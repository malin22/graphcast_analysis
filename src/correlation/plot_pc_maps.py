#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from graphcast import icosahedral_mesh


def load_activations(path: Path) -> np.ndarray:
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"{path}: expected [nodes, features], got {x.shape}")

    return x.astype(np.float32, copy=False)


def get_mesh_latlon(splits: int = 6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=splits
    )
    vertices = meshes[6].vertices

    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))

    return lat, lon


def plot_pc_map(
    values,
    lat,
    lon,
    output_path,
    title,
    add_world_map=False,
    world_map_resolution="110m",
):
    vmax = max(np.nanpercentile(np.abs(values), 99), 1e-8)

    if add_world_map:
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_global()

        ax.add_feature(
            cfeature.LAND.with_scale(world_map_resolution),
            facecolor="lightgray",
            edgecolor="none",
            alpha=0.55,
            zorder=0,
        )
        ax.add_feature(
            cfeature.COASTLINE.with_scale(world_map_resolution),
            linewidth=0.5,
            edgecolor="black",
            zorder=3,
        )

        scatter = ax.scatter(
            lon,
            lat,
            c=values,
            s=2,
            linewidths=0,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )

        gridlines = ax.gridlines(
            draw_labels=True,
            linewidth=0.3,
            color="gray",
            alpha=0.4,
            linestyle="--",
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.xlabel_style = {"size": 14}
        gridlines.ylabel_style = {"size": 14}

    else:
        fig, ax = plt.subplots(figsize=(13, 6))

        scatter = ax.scatter(
            lon,
            lat,
            c=values,
            s=2,
            linewidths=0,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )

        ax.set(
            xlim=(-180, 180),
            ylim=(-90, 90),
            xlabel="Longitude",
            ylabel="Latitude",
        )
        ax.tick_params(labelsize=14)
        ax.set_xlabel("Longitude", fontsize=18)
        ax.set_ylabel("Latitude", fontsize=18)

    colourbar = plt.colorbar(scatter, ax=ax)
    colourbar.set_label("PC score", fontsize=16, labelpad=12)
    colourbar.ax.tick_params(labelsize=13)

    ax.set_title(title, fontsize=25, pad=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_explained_variance(variance_ratio, output_path):
    cumulative = np.cumsum(variance_ratio)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        np.arange(1, len(cumulative) + 1),
        cumulative,
        linewidth=2,
        color="steelblue",
    )
    ax.set(
        xlabel="Number of principal components",
        ylabel="Cumulative explained variance",
        ylim=(0, 1.01),
    )
    ax.set_title("Cumulative explained variance", fontsize=22, pad=12)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_pc_indices(n_available, n_pcs, use_last_pcs, pc_indices):
    if pc_indices is not None:
        indices = np.asarray(pc_indices, dtype=int) - 1

        if np.any(indices < 0) or np.any(indices >= n_available):
            raise ValueError(
                f"Requested PC numbers must be between 1 and {n_available}."
            )
        return indices

    n_pcs = min(n_pcs, n_available)

    if use_last_pcs:
        return np.arange(n_available - n_pcs, n_available)

    return np.arange(n_pcs)


def compute_yearly_mean_scores(
    activations_dir,
    pca_components,
    pca_mean,
    pc_indices,
):
    pattern = "layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t*.npy"
    files = sorted(Path(activations_dir).glob(pattern))

    if not files:
        raise FileNotFoundError(f"No matching activations in {activations_dir}")

    lat, _ = get_mesh_latlon(splits=6)
    selected_components = pca_components[pc_indices]

    score_sum = None
    valid_count = 0

    for file_path in files:
        activations = load_activations(file_path)

        if not np.isfinite(activations).all():
            print(f"Skipping non-finite file: {file_path.name}")
            continue

        if activations.shape[0] != len(lat):
            raise ValueError(
                f"Mesh-node mismatch in {file_path.name}: "
                f"{activations.shape[0]} versus {len(lat)}"
            )

        scores = (activations - pca_mean) @ selected_components.T

        if score_sum is None:
            score_sum = np.zeros_like(scores, dtype=np.float64)

        score_sum += scores
        valid_count += 1

    if valid_count == 0:
        raise RuntimeError("No valid activation files were available.")

    return (score_sum / valid_count).astype(np.float32), valid_count

def run_plotting(
    activations_dir,
    pca_components_path,
    pca_mean_path,
    variance_ratio_path,
    out_dir,
    label="2021",
    n_pcs=10,
    use_last_pcs=True,
    pc_indices=None,
    add_world_map=True,
    plot_only=False,
    cache_path=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cache_path is None:
        cache_path = out_dir / "mean_pc_scores_cache.npz"
    else:
        cache_path = Path(cache_path)

    if plot_only:
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cache found at {cache_path}. "
                "Run once with plot_only=False first."
            )

        print(f"Loading cached scores from: {cache_path}")
        cache = np.load(cache_path)

        mean_scores = cache["mean_scores"]
        lat = cache["lat"]
        lon = cache["lon"]
        pc_numbers = cache["pc_numbers"]

    else:
        pca_components = np.load(pca_components_path)
        pca_mean = np.load(pca_mean_path)

        pc_indices = select_pc_indices(
            n_available=pca_components.shape[0],
            n_pcs=n_pcs,
            use_last_pcs=use_last_pcs,
            pc_indices=pc_indices,
        )

        mean_scores, n_files = compute_yearly_mean_scores(
            activations_dir=activations_dir,
            pca_components=pca_components,
            pca_mean=pca_mean,
            pc_indices=pc_indices,
        )

        lat, lon = get_mesh_latlon(splits=6)
        pc_numbers = pc_indices + 1

        np.savez(
            cache_path,
            mean_scores=mean_scores,
            lat=lat,
            lon=lon,
            pc_numbers=pc_numbers,
            n_files=n_files,
        )

        print(f"Saved mean-score cache from {n_files} files: {cache_path}")

    for column, pc_number in enumerate(pc_numbers):
        plot_pc_map(
            values=mean_scores[:, column],
            lat=lat,
            lon=lon,
            output_path=out_dir / f"pc{pc_number}_mean_activation_{label}.png",
            title=f"{label} annual-mean PC{pc_number} activation map",
            add_world_map=add_world_map,
        )

    if variance_ratio_path is not None:
        variance_ratio = np.load(variance_ratio_path)

        plot_explained_variance(
            variance_ratio=variance_ratio,
            output_path=out_dir / "pca_cumulative_explained_variance.png",
        )

    print(f"Saved plots to: {out_dir}")

if __name__ == "__main__":
    run_plotting(
        activations_dir=(
            "/share/prj-4d/graphcast_shared/data/"
            "graphcast_activation_2021"
        ),
        pca_components_path=(
            "/share/prj-4d/graphcast_shared/data/"
            "pca_components/512_PCs/layer8_only/"
            "pca_components_2019_2020_layer8.npy"
        ),
        pca_mean_path=(
            "/share/prj-4d/graphcast_shared/data/"
            "pca_components/512_PCs/layer8_only/"
            "pca_mean_2019_2020_layer8.npy"
        ),
        variance_ratio_path=(
            "/share/prj-4d/graphcast_shared/data/"
            "pca_components/512_PCs/layer8_only/"
            "pca_explained_variance_ratio_2019_2020_layer8.npy"
        ),
        out_dir="plots/pc_activation_maps/last_pcs_2021",
        label="2021",
        n_pcs=10,
        use_last_pcs=True,
        add_world_map=True,
        plot_only=False,
    )