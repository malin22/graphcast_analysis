#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from graphcast import icosahedral_mesh


ACTS_DIR = Path(
    "/share/prj-4d/graphcast_shared/data/"
    "graphcast_activations_all_layers_2019"
)

PLOTS_ROOT = Path(
    "/home/student/s/sascholle/share/graphcast_analysis/plots/layers"
)

ALL_LAYERS_PCA_COMPONENTS = Path(
    "/share/prj-4d/graphcast_shared/data/pca_components/"
    "512_PCs/all_layers/pca_components_2019_all_layers.npy"
)

ALL_LAYERS_PCA_MEAN = Path(
    "/share/prj-4d/graphcast_shared/data/pca_components/"
    "512_PCs/all_layers/pca_mean_2019_all_layers.npy"
)

PER_LAYER_PCA_DIR = Path(
    "/share/prj-4d/graphcast_shared/data/pca_components/"
    "512_PCs/per_layer/layerwise_2019"
)

def load_activation(path: Path):
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"Expected [nodes, features], got {x.shape} for {path}")

    return x


def mesh_latlon(splits=6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=splits
    )
    vertices = meshes[splits].vertices

    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))

    return lat.astype(np.float32), lon.astype(np.float32)


def load_pca_basis(
    pca_mode: str,
    layer: int,
    all_layers_components,
    all_layers_mean,
    per_layer_pca_dir: Path,
):
    """
    Return PCA mean and components appropriate for one layer.

    In all_layers mode, every layer uses the same common PCA basis.
    In per_layer mode, each layer loads its separately fitted basis.
    """
    if pca_mode == "all_layers":
        return all_layers_mean, all_layers_components

    components_path = (
        per_layer_pca_dir / f"pca_components_layer{layer:04d}.npy"
    )
    mean_path = per_layer_pca_dir / f"pca_mean_layer{layer:04d}.npy"

    if not components_path.exists():
        raise FileNotFoundError(f"Missing PCA components: {components_path}")

    if not mean_path.exists():
        raise FileNotFoundError(f"Missing PCA mean: {mean_path}")

    return np.load(mean_path), np.load(components_path)


def compute_layer_pc_means(
    acts_dir: Path,
    pca_mode: str,
    pc_indices,
    layers,
    all_layers_components,
    all_layers_mean,
    per_layer_pca_dir: Path,
):
    """
    Compute one annual-mean PC map per requested PC and layer.

    Returns:
      layer_maps: [n_layers, n_requested_pcs, n_mesh_nodes]
      valid_counts: [n_layers]
    """
    layer_maps = []
    valid_counts = []

    for layer in layers:
        pca_mean, pca_components = load_pca_basis(
            pca_mode=pca_mode,
            layer=layer,
            all_layers_components=all_layers_components,
            all_layers_mean=all_layers_mean,
            per_layer_pca_dir=per_layer_pca_dir,
        )

        if pca_components.ndim != 2:
            raise ValueError(
                f"Layer {layer:04d}: expected PCA components [PCs, features], "
                f"got {pca_components.shape}"
            )

        if np.max(pc_indices) >= pca_components.shape[0]:
            raise ValueError(
                f"Layer {layer:04d}: requested PC index "
                f"{np.max(pc_indices)} but basis contains only "
                f"{pca_components.shape[0]} PCs"
            )

        files = sorted(
            acts_dir.glob(
                f"layer{layer:04d}_mesh_gnn_post_res_nodes_mesh_nodes_t2019-*.npy"
            )
        )

        if not files:
            raise FileNotFoundError(f"No 2019 files found for layer {layer:04d}")

        print(
            f"Layer {layer:04d}: {len(files)} activation files; "
            f"PCA mode = {pca_mode}",
            flush=True,
        )

        score_sum = None
        valid_count = 0

        selected_components = pca_components[pc_indices]

        for path in files:
            activations = load_activation(path)

            if activations.shape[1] != pca_mean.shape[0]:
                raise ValueError(
                    f"{path.name}: activation feature dimension "
                    f"{activations.shape[1]} does not match PCA mean dimension "
                    f"{pca_mean.shape[0]}"
                )

            if not np.isfinite(activations).all():
                print(f"  Skipping non-finite activation file: {path.name}")
                continue

            scores = (
                (activations - pca_mean) @ selected_components.T
            )  # [nodes, requested PCs]

            pc_maps = scores.T.astype(np.float32)  # [requested PCs, nodes]

            if not np.isfinite(pc_maps).all():
                print(f"  Skipping non-finite score map: {path.name}")
                continue

            if score_sum is None:
                score_sum = np.zeros_like(pc_maps, dtype=np.float64)

            score_sum += pc_maps
            valid_count += 1

        if valid_count == 0:
            raise ValueError(f"No valid activation files for layer {layer:04d}")

        print(f"  Used {valid_count}/{len(files)} files", flush=True)

        layer_maps.append((score_sum / valid_count).astype(np.float32))
        valid_counts.append(valid_count)

    return np.stack(layer_maps, axis=0), np.asarray(valid_counts)


def plot_static_layer_grid(
    layer_maps,
    layers,
    pc_idx_local,
    pc_label,
    lat,
    lon,
    output_path,
    pca_mode,
):
    """Save a 4 x 4 map grid, one annual-mean map per layer."""
    n_layers = len(layers)
    ncols = 4
    nrows = int(np.ceil(n_layers / ncols))

    values = layer_maps[:, pc_idx_local, :]
    vmax = np.nanpercentile(np.abs(values), 99)
    vmax = max(float(vmax), 1e-6)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(16, 3.2 * nrows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).ravel()

    scatter = None

    for i, layer in enumerate(layers):
        ax = axes[i]

        scatter = ax.scatter(
            lon,
            lat,
            c=values[i],
            s=1,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            linewidths=0,
        )

        ax.set_title(f"Layer {layer:04d}", fontsize=11)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n_layers:]:
        ax.axis("off")

    fig.suptitle(
        f"{pc_label}: 2019 annual-mean map across layers\n"
        f"PCA mode: {pca_mode}",
        fontsize=16,
    )

    fig.colorbar(
        scatter,
        ax=axes[:n_layers],
        shrink=0.8,
        label="PC score",
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def animate_pc_layers(
    layer_maps,
    layers,
    pc_idx_local,
    pc_label,
    lat,
    lon,
    output_path,
    pca_mode,
):
    """Optional layer-by-layer animation for one PC."""
    values = layer_maps[:, pc_idx_local, :]

    vmax = np.nanpercentile(np.abs(values), 99)
    vmax = max(float(vmax), 1e-6)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    scatter = ax.scatter(
        lon,
        lat,
        c=values[0],
        s=2,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0,
    )

    ax.set(
        xlim=(-180, 180),
        ylim=(-90, 90),
        xlabel="Longitude",
        ylabel="Latitude",
    )

    fig.colorbar(scatter, ax=ax, label="PC score")
    title = ax.set_title("")

    def update(frame):
        scatter.set_array(values[frame])
        title.set_text(
            f"{pc_label}; layer {layers[frame]:04d}; PCA mode: {pca_mode}"
        )
        return scatter, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(layers),
        interval=700,
        blit=False,
        repeat=True,
    )

    if output_path.suffix.lower() == ".gif":
        ani.save(output_path, writer="pillow", dpi=150)
    else:
        ani.save(output_path, writer="ffmpeg", dpi=150)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot annual-mean GraphCast PC maps across layers."
    )

    parser.add_argument("--acts-dir", type=Path, default=ACTS_DIR)
    parser.add_argument("--plots-root", type=Path, default=PLOTS_ROOT)

    parser.add_argument(
        "--pca-mode",
        choices=["all_layers", "per_layer"],
        required=True,
        help=(
            "all_layers: shared PCA fitted to all layers; "
            "per_layer: a separately fitted PCA basis for each layer."
        ),
    )

    parser.add_argument(
        "--pc-indices",
        type=int,
        nargs="+",
        default=[3],
        help="Zero-based PC indices. E.g. 1 means PC2.",
    )

    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(range(16)),
        help="GraphCast layers to plot; defaults to layers 0 through 15.",
    )

    parser.add_argument(
        "--make-animation",
        action="store_true",
        help="Also make GIFs or MP4s. Grids are always saved.",
    )

    parser.add_argument(
        "--animation-format",
        choices=["gif", "mp4"],
        default="gif",
    )

    parser.add_argument(
        "--all-layers-components",
        type=Path,
        default=ALL_LAYERS_PCA_COMPONENTS,
    )

    parser.add_argument(
        "--all-layers-mean",
        type=Path,
        default=ALL_LAYERS_PCA_MEAN,
    )

    parser.add_argument(
        "--per-layer-pca-dir",
        type=Path,
        default=PER_LAYER_PCA_DIR,
    )

    args = parser.parse_args()

    pc_indices = np.asarray(args.pc_indices, dtype=np.int64)

    if np.any(pc_indices < 0):
        raise ValueError("PC indices must be non-negative.")

    output_dir = args.plots_root / f"{args.pca_mode}_2019"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_layers_components = None
    all_layers_mean = None

    if args.pca_mode == "all_layers":
        all_layers_components = np.load(args.all_layers_components)
        all_layers_mean = np.load(args.all_layers_mean)

        print(
            f"Loaded shared PCA basis:\n"
            f"  components: {args.all_layers_components}\n"
            f"  mean:       {args.all_layers_mean}"
        )

    lat, lon = mesh_latlon(splits=6)

    layer_maps, valid_counts = compute_layer_pc_means(
        acts_dir=args.acts_dir,
        pca_mode=args.pca_mode,
        pc_indices=pc_indices,
        layers=args.layers,
        all_layers_components=all_layers_components,
        all_layers_mean=all_layers_mean,
        per_layer_pca_dir=args.per_layer_pca_dir,
    )

    np.save(output_dir / "2019_layer_mean_pc_maps.npy", layer_maps)
    np.save(output_dir / "2019_layer_indices.npy", np.asarray(args.layers))
    np.save(output_dir / "2019_pc_indices.npy", pc_indices)
    np.save(output_dir / "2019_valid_file_counts.npy", valid_counts)

    with open(output_dir / "run_metadata.txt", "w") as f:
        f.write(f"pca_mode={args.pca_mode}\n")
        f.write(f"layers={list(args.layers)}\n")
        f.write(f"pc_indices_zero_based={pc_indices.tolist()}\n")
        f.write(f"valid_file_counts={valid_counts.tolist()}\n")

    for local_pc_index, pc_index in enumerate(pc_indices):
        pc_label = f"PC{pc_index + 1}"

        plot_static_layer_grid(
            layer_maps=layer_maps,
            layers=args.layers,
            pc_idx_local=local_pc_index,
            pc_label=pc_label,
            lat=lat,
            lon=lon,
            output_path=output_dir / f"{pc_label}_layers_grid.png",
            pca_mode=args.pca_mode,
        )

        if args.make_animation:
            extension = "gif" if args.animation_format == "gif" else "mp4"

            animate_pc_layers(
                layer_maps=layer_maps,
                layers=args.layers,
                pc_idx_local=local_pc_index,
                pc_label=pc_label,
                lat=lat,
                lon=lon,
                output_path=(
                    output_dir / f"{pc_label}_layers_animation.{extension}"
                ),
                pca_mode=args.pca_mode,
            )

    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()