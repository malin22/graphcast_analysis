#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from graphcast import icosahedral_mesh


def mesh_hierarchy_indices(splits: int = 6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    verts = [m.vertices for m in meshes]
    v6 = verts[6]

    def to_key(v):
        return tuple(np.round(v, 12))

    m6_map = {to_key(v): i for i, v in enumerate(v6)}

    def level_idx_in_m6(level):
        return np.array([m6_map[to_key(v)] for v in verts[level]], dtype=np.int64)

    idx_by_level_cumulative = {}
    idx_by_level_only = {}

    prev = np.array([], dtype=np.int64)
    for level in range(splits + 1):
        idx = level_idx_in_m6(level)
        idx_by_level_cumulative[level] = idx
        idx_by_level_only[level] = np.setdiff1d(idx, prev)
        prev = idx

    return verts[6], idx_by_level_cumulative, idx_by_level_only


def vertices_to_latlon(vertices: np.ndarray):
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    return lat, lon


def load_activations(path: Path) -> np.ndarray:
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    if x.ndim != 2:
        raise ValueError(f"Expected [nodes, features], got {x.shape} for {path}")

    return x.astype(np.float32)


def parse_timestamp(path: Path) -> str:
    m = re.search(r"_t(.+)\.npy$", path.name)
    if not m:
        return path.stem
    return m.group(1)


def collect_files(acts_dir: Path, max_files: int | None = None):
    pattern = "layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t*.npy"
    files = sorted(acts_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No activation files found in {acts_dir} with {pattern}")

    if max_files is not None:
        files = files[:max_files]

    return files


def load_dataset(files, mesh_level, use_only_new_nodes):
    vertices_m6, idx_by_level_cumulative, idx_by_level_only = mesh_hierarchy_indices(splits=6)

    if use_only_new_nodes:
        selected_idx = idx_by_level_only[mesh_level]
    else:
        selected_idx = idx_by_level_cumulative[mesh_level]

    selected_vertices = vertices_m6[selected_idx]
    lat, lon = vertices_to_latlon(selected_vertices)

    X_all = []
    time_labels = []
    node_indices = []
    lat_all = []
    lon_all = []
    timestamps = []

    for time_idx, path in enumerate(files):
        timestamp = parse_timestamp(path)
        print(f"Loading {path.name}")

        X = load_activations(path)

        if np.isnan(X).any():
            print(f"WARNING: skipping {path.name}, contains NaNs")
            continue

        if X.shape[0] != vertices_m6.shape[0]:
            raise ValueError(
                f"Expected m6 activations with {vertices_m6.shape[0]} nodes, got {X.shape[0]}"
            )

        X_sel = X[selected_idx]

        X_all.append(X_sel)
        time_labels.extend([time_idx] * X_sel.shape[0])
        node_indices.extend(selected_idx.tolist())
        lat_all.extend(lat.tolist())
        lon_all.extend(lon.tolist())
        timestamps.append(timestamp)

        print(f"  kept mesh level {mesh_level}: {X_sel.shape[0]} nodes, feature dim {X_sel.shape[1]}")

    if not X_all:
        raise ValueError("No valid activation files loaded")

    X_all = np.vstack(X_all).astype(np.float32)

    return (
        X_all,
        np.asarray(time_labels, dtype=np.int32),
        np.asarray(node_indices, dtype=np.int32),
        np.asarray(lat_all, dtype=np.float32),
        np.asarray(lon_all, dtype=np.float32),
        timestamps,
    )

def plot_embedding_node_trajectories(
    embedding,
    time_labels,
    node_indices,
    timestamps,
    out_path,
    max_nodes=80,
    lat=None, 
    seed=0,
):
    """
    Plot t-SNE embedding with faint line segments connecting the same mesh node
    across timesteps.

    Assumes:
      embedding: [n_points, 2]
      time_labels: [n_points], integer timestep label
      node_indices: [n_points], original m6 node index
      timestamps: list of timestamp strings
    """
    rng = np.random.default_rng(seed)

    # unique_nodes = np.unique(node_indices)

    # if len(unique_nodes) > max_nodes:
    #     chosen_nodes = unique_nodes[:max_nodes]
    #     #chosen_nodes = rng.choice(unique_nodes, size=max_nodes, replace=False)
    # else:
    #     chosen_nodes = unique_nodes

    unique_nodes = np.unique(node_indices)

    node_mean_lat = {}
    for node in unique_nodes:
        node_mean_lat[node] = float(np.nanmean(lat[node_indices == node]))

    # Northernmost nodes first.
    nodes_sorted_by_lat = sorted(
        unique_nodes,
        key=lambda node: node_mean_lat[node],
        reverse=True,
    )

    chosen_nodes = np.asarray(nodes_sorted_by_lat[:max_nodes])

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background: all points in light gray.
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=5,
        alpha=0.18,
        color="lightgray",
        rasterized=True,
    )

    cmap = plt.get_cmap("viridis")
    n_times = max(1, len(timestamps) - 1)

    for node in chosen_nodes:
        mask = node_indices == node

        if mask.sum() < 2:
            continue

        node_embedding = embedding[mask]
        node_times = time_labels[mask]

        order = np.argsort(node_times)
        node_embedding = node_embedding[order]
        node_times = node_times[order]

        # Faint line through this node's temporal path.
        ax.plot(
            node_embedding[:, 0],
            node_embedding[:, 1],
            color="black",
            alpha=0.16,
            linewidth=0.8,
            zorder=2,
        )

        # Colored points along the path.
        ax.scatter(
            node_embedding[:, 0],
            node_embedding[:, 1],
            c=node_times,
            cmap=cmap,
            vmin=0,
            vmax=n_times,
            s=14,
            alpha=0.85,
            edgecolor="none",
            zorder=3,
        )

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0, vmax=n_times),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)

    if len(timestamps) <= 8:
        cbar.set_ticks(np.arange(len(timestamps)))
        cbar.set_ticklabels(timestamps)

    cbar.set_label("Timestep")

    ax.set_title("Same-node trajectories through t-SNE latent space", fontsize=14, pad=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_embedding_by_timestep(embedding, time_labels, timestamps, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab10" if len(timestamps) <= 10 else "tab20")

    for time_idx, timestamp in enumerate(timestamps):
        mask = time_labels == time_idx
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=8,
            alpha=0.65,
            color=cmap(time_idx % cmap.N),
            label=timestamp,
            rasterized=True,
        )

    ax.set_title("Layer 8 node activation latent space by timestep", fontsize=14, pad=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=2, fontsize=8, frameon=True, loc="best")
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_embedding_by_latlon(embedding, values, label, out_path, cmap="coolwarm"):
    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=values,
        s=8,
        alpha=0.75,
        cmap=cmap,
        rasterized=True,
    )

    ax.set_title(f"Layer 8 node activation latent space colored by {label}", fontsize=14, pad=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.15)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(label)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Plot t-SNE of GraphCast node activation latent space."
    )
    parser.add_argument(
        "--acts-dir",
        type=Path,
        default=Path("/share/prj-4d/graphcast_shared/data/graphcast_activation_2019"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/mapping_experiments/latent_space_tsne"))
    parser.add_argument("--max-files", type=int, default=4)
    parser.add_argument("--mesh-level", type=int, default=3, choices=range(0, 7))
    parser.add_argument(
        "--use-only-new-nodes",
        action="store_true",
        help="Use only nodes introduced at this mesh level, not cumulative hierarchy nodes.",
    )
    parser.add_argument("--pca-dim", type=int, default=None)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--learning-rate", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument("--npz", type=Path, default="/home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/mapping_experiments/latent_space_tsne/latent_tsne_2019_m3.npz")
    parser.add_argument("--max-trajectory-nodes", type=int, default=80)
    
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.npz is not None: 
        print(f"Loading precomputed tsne from {args.npz}")

        data = np.load(args.npz, allow_pickle=True)
        embedding = data["embedding"]
        time_labels = data["time_labels"]
        node_indices = data["node_indices"]
        lat = data["lat"]
        lon = data["lon"]
        timestamps = data["timestamps"]

        mesh_level = int(data["mesh_level"]) if "mesh_level" in data.files else args.mesh_level

        # plot_embedding_by_timestep(
        #     embedding,
        #     time_labels,
        #     timestamps,
        #     args.out_dir / f"latent_tsne_2019_m{mesh_level}_by_timestep.png",
        # )

        # plot_embedding_by_latlon(
        #     embedding,
        #     lat,
        #     "latitude",
        #     args.out_dir / f"latent_tsne_2019_m{mesh_level}_by_latitude.png",
        #     cmap="coolwarm",
        # )

        # plot_embedding_by_latlon(
        #     embedding,
        #     lon,
        #     "longitude",
        #     args.out_dir / f"latent_tsne_2019_m{mesh_level}_by_longitude.png",
        #     cmap="twilight",
        # )

        plot_embedding_node_trajectories(
            embedding=embedding,
            time_labels=time_labels,
            node_indices=node_indices,
            timestamps=timestamps,
            lat=lat,
            out_path=args.out_dir / f"latent_tsne_2019_m{mesh_level}_same_node_trajectories_Northern.png",
            max_nodes=args.max_trajectory_nodes,
            seed=args.seed,
        )

        print(f"Saved plots from existing t-SNE .npz to {args.out_dir}")
        return

    files = collect_files(args.acts_dir, max_files=args.max_files)

    X, time_labels, node_indices, lat, lon, timestamps = load_dataset(
        files=files,
        mesh_level=args.mesh_level,
        use_only_new_nodes=args.use_only_new_nodes,
    )

    print(f"\nPooled matrix shape: {X.shape}")
    print("Rows are node-timestep samples; columns are activation channels.")

    if args.pca_dim is not None and args.pca_dim < X.shape[1]:
        print(f"Running PCA pre-reduction to {args.pca_dim} dims")
        pca = PCA(n_components=args.pca_dim, random_state=args.seed)
        X_reduced = pca.fit_transform(X)
        print(f"Cumulative variance in PCA pre-step: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        X_reduced = X

    print("Running t-SNE")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        learning_rate=args.learning_rate,
        init="pca",
        random_state=args.seed,
        max_iter=args.tsne_iterations,
        verbose=1,
    )

    embedding = tsne.fit_transform(X_reduced)

    np.savez(
        args.out_dir / f"latent_tsne_2019_m{args.mesh_level}.npz",
        embedding=embedding.astype(np.float32),
        time_labels=time_labels,
        node_indices=node_indices,
        lat=lat,
        lon=lon,
        timestamps=np.asarray(timestamps),
        files=np.asarray([str(f) for f in files]),
        mesh_level=args.mesh_level,
    )

    plot_embedding_by_timestep(
        embedding,
        time_labels,
        timestamps,
        args.out_dir / f"latent_tsne_2019_m{args.mesh_level}_by_timestep.png",
    )

    plot_embedding_by_latlon(
        embedding,
        lat,
        "latitude",
        args.out_dir / f"latent_tsne_2019_m{args.mesh_level}_by_latitude.png",
        cmap="coolwarm",
    )

    plot_embedding_by_latlon(
        embedding,
        lon,
        "longitude",
        args.out_dir / f"latent_tsne_2019_m{args.mesh_level}_by_longitude.png",
        cmap="twilight",
    )

    plot_embedding_node_trajectories(
        embedding=embedding,
        time_labels=time_labels,
        node_indices=node_indices,
        timestamps=timestamps,
        out_path=args.out_dir / f"latent_tsne_2019_m{args.mesh_level}_same_node_trajectories.png",
        max_nodes=80,
        seed=args.seed,
    )

    print(f"Saved outputs to {args.out_dir}")


if __name__ == "__main__":
    main()