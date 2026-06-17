import os
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
from graphcast import icosahedral_mesh


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


def plot_pc_map(scores, lat, lon, out_path, title):
    vmax = np.percentile(np.abs(scores), 99)
    vmax = max(vmax, 1e-6)
    plt.figure(figsize=(12, 6))
    sc = plt.scatter(lon, lat, c=scores, s=2, cmap="coolwarm", vmin=-vmax, vmax=vmax, linewidths=0)
    plt.colorbar(sc, label="PC score")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_yearly_mean_pcs(
    acts_dir,
    pca_components_path,
    pca_mean_path,
    out_dir="plots",
    n_top_pcs=5,
    scramble_activations=False,
):
    os.makedirs(out_dir, exist_ok=True)

    pca_components = np.load(pca_components_path)
    pca_mean = np.load(pca_mean_path)

    npy_files = sorted(glob(os.path.join(acts_dir, "*.npy")))
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

        scores = (X - pca_mean) @ pca_components[:n_top_pcs].T

        if score_sum is None:
            score_sum = np.zeros_like(scores, dtype=np.float64)

        score_sum += scores
        valid_count += 1

    if valid_count == 0:
        raise ValueError("No valid files available for averaging")

    mean_scores = score_sum / valid_count
    np.save(os.path.join(out_dir, "mean_pc_scores_year.npy"), mean_scores)

    for pc_idx in range(n_top_pcs):
        plot_pc_map(
            mean_scores[:, pc_idx],
            lat,
            lon,
            os.path.join(out_dir, f"pc{pc_idx + 1}_mean_activation_map_year.png"),
            title=f"Year-mean PC{pc_idx + 1} activation map",
        )

    print(f"Saved yearly mean PC maps from {valid_count} files to {out_dir}")

def plot_cumulative_explained_variance(ipca, out_dir, max_components=None):
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
    plt.savefig(os.path.join(out_dir, "pca_cumulative_explained_variance_2021.png"), dpi=300, bbox_inches="tight")
    plt.close()

def run_pca(
    acts_dir: str,
    n_components: int = 20,
    batch_size: int = 10,
    out_dir: str = "/share/prj-4d/graphcast_shared/data/pca_components/"
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
    npy_files = sorted(glob(os.path.join(acts_dir, "*.npy")))

    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {acts_dir}")

    print(f"Found {len(npy_files)} activation files")

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
    np.save(os.path.join(out_dir, "pca_components_2021.npy"), ipca.components_)
    np.save(os.path.join(out_dir, "pca_mean_2021.npy"), ipca.mean_)
    print(f"Saved PCA basis to {out_dir}/")

    # Plot cumulative explained variance
    plot_cumulative_explained_variance(ipca, out_dir)

    # Print explained variance
    print("\nExplained variance ratio:")
    print(ipca.explained_variance_ratio_)
    print("Cumulative explained variance:")
    print(np.cumsum(ipca.explained_variance_ratio_))


    return ipca

if __name__ == "__main__":
    ACTS_DIR = "/share/prj-4d/graphcast_shared/data/graphcast_activation_2021"
    PCA_DIR = "/share/prj-4d/graphcast_shared/data/pca_components_test"
    PLOTS_OUT    = "plots/2021_pca_projected_on_2021"

    ipca = run_pca(
        acts_dir=ACTS_DIR,
        n_components=400,
        batch_size=10,
        out_dir=PCA_DIR,
  
    )

    plot_cumulative_explained_variance()
    

    # plot_yearly_mean_pcs(
    #     acts_dir=ACTS_DIR,
    #     pca_components_path='/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021.npy',
    #     pca_mean_path='/share/prj-4d/graphcast_shared/data/pca_components/pca_mean_2021.npy',
    #     out_dir="plots/2021_pca_projected_on_2021_20pcs",
    #     n_top_pcs=20,
    #     scramble_activations=False, # Set to True to scramble activations before projection -> should yield no meaningful spatial patterns in the PC maps, confirming that the original patterns are not artifacts of the PCA basis alone.
    # )