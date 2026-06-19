# animate_pc_layers.py

#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from graphcast import icosahedral_mesh


ACTS_DIR = Path("/share/prj-4d/graphcast_shared/data/graphcast_activations_all_layers_1week_January_2021")
OUT_DIR = Path("/home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/layer_analysis")

PCA_COMPONENTS_PATH = Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021.npy")
PCA_MEAN_PATH = Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_mean_2021.npy")


def to_float32(x):
    arr = np.asarray(x)
    if arr.dtype == np.dtype("|V2"):
        arr = arr.view(np.float16)
    return np.asarray(arr, dtype=np.float32)


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


def parse_layer(path: Path):
    m = re.search(r"layer(\d{4})_", path.name)
    if not m:
        raise ValueError(f"Could not parse layer from {path.name}")
    return int(m.group(1))


def mesh_latlon(splits=6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    vertices = meshes[-1].vertices
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    return lat.astype(np.float32), lon.astype(np.float32)


def compute_layer_pc_means(acts_dir, pca_mean, pca_components, pc_indices, layers):
    """
    Returns:
      layer_maps: [n_layers, n_pcs, n_nodes]
    """
    layer_maps = []

    for layer in layers:
        files = sorted(
            acts_dir.glob(
                f"layer{layer:04d}_mesh_gnn_post_res_nodes_mesh_nodes_t*.npy"
            )
        )

        if not files:
            raise FileNotFoundError(f"No files found for layer {layer:04d}")

        print(f"Layer {layer:04d}: {len(files)} files", flush=True)

        sum_maps = None
        count = 0

        for path in files:
            acts = load_activation(path)

            if acts.shape[1] != pca_mean.shape[0]:
                raise ValueError(
                    f"{path.name}: feature dim {acts.shape[1]} does not match "
                    f"PCA mean dim {pca_mean.shape[0]}"
                )

            centered = acts - pca_mean
            scores = centered @ pca_components[pc_indices].T  # [nodes, n_pcs]
            pc_maps = scores.T.astype(np.float32)             # [n_pcs, nodes]

            if not np.isfinite(pc_maps).all():
                print(f"  skipping NaN file: {path.name}", flush=True)
                continue

            if sum_maps is None:
                sum_maps = np.zeros_like(pc_maps, dtype=np.float64)

            sum_maps += pc_maps
            count += 1

        if count == 0:
            raise ValueError(f"No valid files for layer {layer:04d}")

        layer_maps.append((sum_maps / count).astype(np.float32))

    return np.stack(layer_maps, axis=0)


def plot_static_layer_grid(layer_maps, layers, pc_idx_local, pc_label, lat, lon, output_path):
    n_layers = len(layers)
    ncols = 4
    nrows = int(np.ceil(n_layers / ncols))

    values = layer_maps[:, pc_idx_local, :]
    vmax = np.nanpercentile(np.abs(values), 99)
    vmax = max(float(vmax), 1e-6)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows), constrained_layout=True)
    axes = np.asarray(axes).ravel()

    for i, layer in enumerate(layers):
        ax = axes[i]
        sc = ax.scatter(
            lon,
            lat,
            c=values[i],
            s=1,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_title(f"Layer {layer:04d}")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[n_layers:]:
        ax.axis("off")

    fig.suptitle(f"{pc_label}: January 2021 mean PC map across layers")
    fig.colorbar(sc, ax=axes[:n_layers], shrink=0.8, label="PC score")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def animate_pc_layers(layer_maps, layers, pc_idx_local, pc_label, lat, lon, output_path):
    values = layer_maps[:, pc_idx_local, :]
    vmax = np.nanpercentile(np.abs(values), 99)
    vmax = max(float(vmax), 1e-6)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sc = ax.scatter(
        lon,
        lat,
        c=values[0],
        s=2,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0,
    )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(sc, ax=ax, label="PC score")

    title = ax.set_title("")

    def update(frame):
        sc.set_array(values[frame])
        title.set_text(f"{pc_label}: layer {layers[frame]:04d}")
        return sc, title

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(layers),
        interval=700,
        blit=False,
        repeat=True,
    )

    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        ani.save(output_path, writer="pillow", dpi=150)
    else:
        ani.save(output_path, writer="ffmpeg", dpi=150)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Animate PC maps across GraphCast layers.")
    parser.add_argument("--acts-dir", type=Path, default=ACTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--pc-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(16)))
    parser.add_argument("--animation-format", choices=["gif", "mp4"], default="gif")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pca_components = np.load(PCA_COMPONENTS_PATH)
    pca_mean = np.load(PCA_MEAN_PATH)

    pc_indices = np.asarray(args.pc_indices, dtype=np.int64)
    lat, lon = mesh_latlon(splits=6)

    layer_maps = compute_layer_pc_means(
        acts_dir=args.acts_dir,
        pca_mean=pca_mean,
        pca_components=pca_components,
        pc_indices=pc_indices,
        layers=args.layers,
    )

    np.save(
        args.out_dir / "1weekjanuary2021_layer_mean_pc_maps.npy",
        layer_maps,
    )
    np.save(
        args.out_dir / "1weekjanuary2021_layer_indices.npy",
        np.asarray(args.layers, dtype=np.int32),
    )
    np.save(
        args.out_dir / "1weekjanuary2021_pc_indices.npy",
        pc_indices,
    )

    for local_i, pc_idx in enumerate(pc_indices):
        pc_label = f"PC{pc_idx + 1}"

        plot_static_layer_grid(
            layer_maps=layer_maps,
            layers=args.layers,
            pc_idx_local=local_i,
            pc_label=pc_label,
            lat=lat,
            lon=lon,
            output_path=args.out_dir / f"{pc_label}_layers_grid.png",
        )

        ext = "gif" if args.animation_format == "gif" else "mp4"
        animate_pc_layers(
            layer_maps=layer_maps,
            layers=args.layers,
            pc_idx_local=local_i,
            pc_label=pc_label,
            lat=lat,
            lon=lon,
            output_path=args.out_dir / f"{pc_label}_layers_animation.{ext}",
        )

    print(f"Saved maps and animations to {args.out_dir}")


if __name__ == "__main__":
    main()