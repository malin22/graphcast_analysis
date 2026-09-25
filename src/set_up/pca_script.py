import os
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from graphcast import icosahedral_mesh
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from collections import Counter
import re

def load_activations(path: str) -> np.ndarray:
    """Load activation file, handling dtype conversions."""
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    if x.ndim != 2:
        raise ValueError(f"Expected [nodes, features], got shape {x.shape}")

    return x.astype(np.float32)


def vertices_to_latlon(vertices: np.ndarray):
    """Convert 3D mesh vertices to lat/lon."""
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    return lat, lon


def get_mesh_latlon(splits: int = 6):
    """Get lat/lon coordinates for mesh nodes."""
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(splits=splits)
    vertices = meshes[6].vertices
    return vertices_to_latlon(vertices)

def make_pc_map_axes(add_world_map=False, world_map_resolution="110m"):
    """
    Create map axes. If add_world_map=True, tries to use cartopy.
    Falls back to plain matplotlib axes if cartopy is unavailable.
    """
    if not add_world_map:
        fig, ax = plt.subplots(figsize=(12, 6))
        return fig, ax, None

    try:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()

        ax.add_feature(
            cfeature.LAND.with_scale(world_map_resolution),
            facecolor="lightgray",
            edgecolor="none",
            alpha=0.35,
            zorder=0,
        )
        ax.add_feature(
            cfeature.COASTLINE.with_scale(world_map_resolution),
            linewidth=0.45,
            edgecolor="black",
            alpha=0.55,
            zorder=1,
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale(world_map_resolution),
            linewidth=0.25,
            edgecolor="black",
            alpha=0.25,
            zorder=1,
        )

        return fig, ax, ccrs.PlateCarree()

    except Exception as e:
        print(f"WARNING: could not add cartopy world map background: {e}")
        print("Falling back to plain lon/lat scatter plot.")
        fig, ax = plt.subplots(figsize=(12, 6))
        return fig, ax, None


def plot_pc_map(
    scores,
    lat,
    lon,
    out_path,
    title,
    add_world_map=False,
    world_map_resolution="110m",
):
    vmax = np.percentile(np.abs(scores), 99)
    vmax = max(vmax, 1e-6)

    fig, ax, transform = make_pc_map_axes(
        add_world_map=add_world_map,
        world_map_resolution=world_map_resolution,
    )

    scatter_kwargs = {}
    if transform is not None:
        scatter_kwargs["transform"] = transform

    sc = ax.scatter(
        lon,
        lat,
        c=scores,
        s=2,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0,
        zorder=2,
        **scatter_kwargs,
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("PC score", fontsize=14, labelpad=12)
    cbar.ax.tick_params(labelsize=12)

    if transform is None:
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude", fontsize=18)
        ax.set_ylabel("Latitude", fontsize=18)
    else:
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.3,
            color="gray",
            alpha=0.35,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 14}
        gl.ylabel_style = {"size": 14}

    ax.tick_params(axis="both", labelsize=14)
    ax.set_title(title, fontsize=25, pad=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_mean_pcs(
    acts_dir,
    pca_components_path,
    pca_mean_path,
    out_dir="plots",
    n_top_pcs=5,
    use_last_pcs=False,
    scramble_activations=False,
    add_world_map=False,
    world_map_resolution="110m",
):
    os.makedirs(out_dir, exist_ok=True)

    pca_components = np.load(pca_components_path)
    pca_mean = np.load(pca_mean_path)

    if use_last_pcs:
        pcs = pca_components[-n_top_pcs:]
        pc_labels = range(pca_components.shape[0] - n_top_pcs + 1, pca_components.shape[0] + 1)
    else:
        pcs = pca_components[:n_top_pcs]
        pc_labels = range(1, n_top_pcs + 1)

    #npy_files = sorted(glob(os.path.join(acts_dir, "*.npy")))
    pattern = "layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t20*.npy"
    #pattern = "layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t2021-01-01T00.npy"
    npy_files = sorted(glob(os.path.join(acts_dir, pattern)))

    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {acts_dir}")

    lat, lon = get_mesh_latlon(splits=6)
    n_top_pcs = min(n_top_pcs, pca_components.shape[0])

    score_sum = None
    valid_count = 0

    rng = np.random.default_rng()

    for f in npy_files:
        X = load_activations(f)
        if scramble_activations:
            X_scrambled = np.empty_like(X)
            n_nodes, n_features = X.shape

            for node_idx in range(n_nodes):
                perm = rng.permutation(n_features)
                X_scrambled[node_idx] = X[node_idx, perm]
            X = X_scrambled

        if np.isnan(X).any():
            print(f"WARNING: skipping {os.path.basename(f)} because it contains NaNs")
            continue

        if X.shape[0] != len(lat):
            raise ValueError(
                f"Node mismatch in {os.path.basename(f)}: {X.shape[0]} vs mesh {len(lat)}"
            )

        scores = (X - pca_mean) @ pcs.T

        if score_sum is None:
            score_sum = np.zeros_like(scores, dtype=np.float64)

        score_sum += scores
        valid_count += 1

    if valid_count == 0:
        raise ValueError("No valid files available for averaging")

    mean_scores = score_sum / valid_count
    np.save(os.path.join(out_dir, "mean_pc_scores_year.npy"), mean_scores)

    for pc_idx, pc_num in enumerate(pc_labels):
        plot_pc_map(
            mean_scores[:, pc_idx],
            lat,
            lon,
            os.path.join(out_dir, f"pc{pc_num}_mean_activation_map_year.png"),
            #os.path.join(out_dir, f"pc{pc_num}_activation_map_Jan1.png"),
            #title=f"PC1 of t2021-01-01T00",
            title=f"Year-mean PC{pc_num} activation map",
            add_world_map=add_world_map,
            world_map_resolution=world_map_resolution,
        )

    print(f"Saved yearly mean PC maps from {valid_count} files to {out_dir}")

def plot_cumulative_explained_variance(ipca, out_dir, max_components=None, output_tag="pca"):
    cumulative = np.cumsum(ipca.explained_variance_ratio_)

    if max_components is not None:
        cumulative = cumulative[:max_components]

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", linewidth=2)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.ylim(0, 1.01)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pca_cumulative_explained_variance_{output_tag}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def parse_timestamp_from_path(path: str) -> str:
    name = os.path.basename(path)
    return name.split("_t")[-1].replace(".npy", "")


def collect_activation_files(acts_dirs, pattern):
    """
    Collect matching activation files from one or more directories.

    acts_dirs can be:
      - one string/path
      - list of strings/paths
    """
    if isinstance(acts_dirs, (str, os.PathLike)):
        acts_dirs = [acts_dirs]

    all_files = []

    for acts_dir in acts_dirs:
        files = sorted(glob(os.path.join(str(acts_dir), pattern)))
        print(f"Found {len(files)} files in {acts_dir}")
        all_files.extend(files)

    if not all_files:
        raise FileNotFoundError(f"No files found for pattern {pattern} in {acts_dirs}")

    # Sort by timestamp, then filename, so 2019/2020 are in chronological order.
    all_files = sorted(all_files, key=lambda p: (parse_timestamp_from_path(p), p))

    return all_files


def run_pca(
    acts_dir: str,
    n_components: int = 20,
    batch_size: int = 10,
    out_dir: str = "/share/prj-4d/graphcast_shared/data/pca_components/", 
    output_tag: str = "2019_2020_layer8",
    layer_pattern: str = "layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t2019*.npy",
):
    """
    Fit IncrementalPCA on activation files, skip files with NaNs, and plot top PCs.
    
    Args:
        acts_dir: Directory containing .npy activation files.
        n_components: Number of PCA components to compute.
        batch_size: Number of files to process per batch.
        out_dir: Output directory for plots and saved matrices.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Find all .npy files
    pattern = layer_pattern
    npy_files = collect_activation_files(acts_dir, pattern)

    print(f"Found {len(npy_files)} activation files in total")


    time_counts = Counter()
    layer_counts = Counter()

    pat = re.compile(r"layer(\d+)_mesh_gnn_post_res_nodes_mesh_nodes_t(.+)\.npy")

    for f in npy_files:
        name = os.path.basename(f)
        m = pat.match(name)
        if not m:
            print("No regex match:", name)
            continue

        layer = int(m.group(1))
        timestamp = m.group(2)

        layer_counts[layer] += 1
        time_counts[timestamp] += 1

    print("\nFiles per layer:")
    for layer, count in sorted(layer_counts.items()):
        print(f"layer{layer:04d}: {count}")

    print("\nTimestamps with incomplete layer coverage:")
    for timestamp, count in sorted(time_counts.items()):
        if count not in (1, 16):
            print(timestamp, count)

    print("\nAll timestamps found:")
    for timestamp, count in sorted(time_counts.items()):
        print(timestamp, count)

    # Fit incremental PCA
    ipca = IncrementalPCA(n_components=n_components)
    
    # Track skipped files
    skipped_files = []

    for i in range(0, len(npy_files), batch_size):
        batch_files = npy_files[i : i + batch_size]
        batch_data = []
        batch_files_valid = []

        for f in batch_files:
            data = load_activations(f)
            
            # Check for NaNs
            nan_count = int(np.isnan(data).sum())
            if nan_count > 0:
                print(f"WARNING: Skipping {os.path.basename(f)} — contains {nan_count} NaN values")
                skipped_files.append((os.path.basename(f), nan_count))
                continue
            
            batch_data.append(data)
            batch_files_valid.append(f)

        if not batch_data:
            print(f"Batch {i // batch_size + 1}: All files skipped due to NaNs")
            continue

        batch_data = np.vstack(batch_data)
        print(f"Fitting batch {i // batch_size + 1}: {batch_data.shape} ({len(batch_files_valid)}/{len(batch_files)} files)")
        ipca.partial_fit(batch_data)

    # Print skipped files summary
    if skipped_files:
        print(f"\n{'='*60}")
        print(f"SKIPPED FILES ({len(skipped_files)} total):")
        print(f"{'='*60}")
        for fname, nan_count in skipped_files:
            print(f"  {fname:60s} — {nan_count} NaNs")
        print(f"{'='*60}\n")

    # Print PCA basis info
    print(f"\nPCA components matrix shape: {ipca.components_.shape}")
    print(f"PCA mean vector shape: {ipca.mean_.shape}")

    # Save PCA basis for later reuse
    np.save(os.path.join(out_dir, f"pca_components_{output_tag}.npy"), ipca.components_)
    np.save(os.path.join(out_dir, f"pca_mean_{output_tag}.npy"), ipca.mean_)
    print(f"Saved PCA basis to {out_dir}/")

    # Plot cumulative explained variance
    plot_cumulative_explained_variance(ipca, out_dir, output_tag=output_tag)

    # Print explained variance
    print("\nExplained variance ratio:")
    print(ipca.explained_variance_ratio_)
    print("Cumulative explained variance:")
    print(np.cumsum(ipca.explained_variance_ratio_))

    # Save for later plotting
    variance_path = out_dir / f"{output_tag}_cumulative_explained_variance.npy"
    np.save(np.cumsum(ipca.explained_variance_ratio_))

    print(f"Saved cumulative explained variance to: {variance_path}")



    return ipca

if __name__ == "__main__":
    ACTS_DIR = ["/share/prj-4d/graphcast_shared/data/graphcast_activation_2019", "/share/prj-4d/graphcast_shared/data/graphcast_activation_2020"]  # can also pass in a list for running ipca on multiple years
    PCA_DIR = "/share/prj-4d/graphcast_shared/data/pca_components/512_PCs/layer8_only/rerun_for_cumulative_explained_variance_plot"
    LAYER_PATTERN = "layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t*.npy"
    #PLOTS_OUT    = "plots/2021_projected_on_2021"

    ipca = run_pca(
        acts_dir=ACTS_DIR,
        n_components=512,
        batch_size=10,
        out_dir=PCA_DIR,
        output_tag="2019_2020_layer8",
        layer_pattern=LAYER_PATTERN
  
    )
    

    # plot_yearly_mean_pcs(
    #     acts_dir="/share/prj-4d/graphcast_shared/data/graphcast_activation_2021",
    #     pca_components_path='/share/prj-4d/graphcast_shared/data/pca_components/512_PCs/layer8_only/pca_components_2019_2020_layer8.npy',
    #     pca_mean_path='/share/prj-4d/graphcast_shared/data/pca_components/512_PCs/layer8_only/pca_mean_2019_2020_layer8.npy',
    #     out_dir="plots/2019_2020_pca_projected_on_2021/Jan1_plot",
    #     n_top_pcs=1,
    #     use_last_pcs=False,
    #     scramble_activations=False, # Set to True to scramble activations before projection -> should yield no meaningful spatial patterns in the PC maps, confirming that the original patterns are not artifacts of the PCA basis alone.
    #     add_world_map=False,
    # )