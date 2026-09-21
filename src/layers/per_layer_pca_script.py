import os
import re
from collections import defaultdict
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import IncrementalPCA


# Matches, e.g.:
# layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t2019-03-18T00.npy
FILE_RE = re.compile(
    r"layer(\d+)_mesh_gnn_post_res_nodes_mesh_nodes_t(.+)\.npy$"
)


def load_activations(path: str) -> np.ndarray:
    """Load one activation file as float32 [mesh_nodes, latent_features]."""
    x = np.load(path, mmap_mode="r")

    # Needed if files were saved as raw float16 / void-2 dtype.
    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    if x.ndim != 2:
        raise ValueError(f"{path}: expected [nodes, features], got {x.shape}")

    return x.astype(np.float32, copy=False)


def discover_files(acts_dir: str):
    """
    Return:
      files_by_layer[layer][timestamp] = absolute_path

    A dictionary is used so later comparisons only use timestamps that exist
    for both layers.
    """
    files_by_layer = defaultdict(dict)

    for path in sorted(glob(os.path.join(acts_dir, "*.npy"))):
        match = FILE_RE.match(os.path.basename(path))
        if match is None:
            continue

        layer = int(match.group(1))
        timestamp = match.group(2)

        if timestamp in files_by_layer[layer]:
            raise ValueError(
                f"Duplicate activation for layer {layer}, timestamp {timestamp}"
            )

        files_by_layer[layer][timestamp] = path

    if not files_by_layer:
        raise FileNotFoundError(
            f"No activation files matching the expected GraphCast naming "
            f"convention were found in {acts_dir}"
        )

    return dict(files_by_layer)


def fit_layer_pca(
    layer: int,
    files_by_timestamp: dict,
    n_components: int,
    batch_size_files: int,
    out_dir: str,
):
    """Fit IncrementalPCA for one layer and save its basis."""
    timestamps = sorted(files_by_timestamp)
    ipca = IncrementalPCA(n_components=n_components)
    skipped = []

    for start in range(0, len(timestamps), batch_size_files):
        batch_timestamps = timestamps[start:start + batch_size_files]
        valid_arrays = []

        for timestamp in batch_timestamps:
            path = files_by_timestamp[timestamp]
            x = load_activations(path)

            if np.isnan(x).any() or np.isinf(x).any():
                print(f"Skipping invalid file: {os.path.basename(path)}")
                skipped.append(timestamp)
                continue

            valid_arrays.append(x)

        if not valid_arrays:
            continue

        batch = np.vstack(valid_arrays)

        if batch.shape[0] < n_components:
            raise ValueError(
                f"Layer {layer}: batch has only {batch.shape[0]} samples, "
                f"but n_components={n_components}."
            )

        print(
            f"Layer {layer:04d}: fitting files "
            f"{start + 1}-{start + len(batch_timestamps)} / {len(timestamps)} "
            f"on {batch.shape}"
        )
        ipca.partial_fit(batch)

    if not hasattr(ipca, "components_"):
        raise RuntimeError(f"Layer {layer} PCA was not fitted.")

    tag = f"layer{layer:04d}"
    np.save(os.path.join(out_dir, f"pca_components_{tag}.npy"), ipca.components_)
    np.save(os.path.join(out_dir, f"pca_mean_{tag}.npy"), ipca.mean_)
    np.save(
        os.path.join(out_dir, f"pca_explained_variance_ratio_{tag}.npy"),
        ipca.explained_variance_ratio_,
    )

    with open(os.path.join(out_dir, f"skipped_{tag}.txt"), "w") as f:
        for timestamp in skipped:
            f.write(f"{timestamp}\n")

    return ipca, set(skipped)


def plot_explained_variance(all_pcas: dict, out_dir: str):
    """One explained-variance curve per layer."""
    plt.figure(figsize=(9, 6))

    for layer, ipca in sorted(all_pcas.items()):
        cumulative = np.cumsum(ipca.explained_variance_ratio_)
        plt.plot(
            np.arange(1, len(cumulative) + 1),
            cumulative,
            label=f"Layer {layer}",
            linewidth=1.5,
        )

    plt.xlabel("Number of PCs")
    plt.ylabel("Cumulative explained variance")
    plt.ylim(0, 1.01)
    plt.title("Per-layer PCA cumulative explained variance")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, "per_layer_cumulative_explained_variance.png"),
        dpi=300,
    )
    plt.close()


def score_correlation_between_layers(
    ref_pca,
    other_pca,
    ref_files: dict,
    other_files: dict,
    n_compare_pcs: int,
):
    """
    Calculate correlations between PC scores from two layers.

    Rows are all common (timestamp, mesh-node) pairs. Statistics are accumulated
    streaming-style, so no large score matrix is held in memory.
    """
    common_timestamps = sorted(set(ref_files) & set(other_files))
    if not common_timestamps:
        raise ValueError("The two layers have no timestamps in common.")

    k = min(
        n_compare_pcs,
        ref_pca.components_.shape[0],
        other_pca.components_.shape[0],
    )

    n_rows = 0
    sum_ref = np.zeros(k, dtype=np.float64)
    sum_other = np.zeros(k, dtype=np.float64)
    sum_ref_sq = np.zeros(k, dtype=np.float64)
    sum_other_sq = np.zeros(k, dtype=np.float64)
    sum_cross = np.zeros((k, k), dtype=np.float64)

    for timestamp in common_timestamps:
        x_ref = load_activations(ref_files[timestamp])
        x_other = load_activations(other_files[timestamp])

        if (
            np.isnan(x_ref).any()
            or np.isnan(x_other).any()
            or np.isinf(x_ref).any()
            or np.isinf(x_other).any()
        ):
            print(f"Skipping invalid paired timestamp: {timestamp}")
            continue

        if x_ref.shape[0] != x_other.shape[0]:
            raise ValueError(
                f"Node-count mismatch at {timestamp}: "
                f"{x_ref.shape[0]} vs {x_other.shape[0]}"
            )

        # PCA scores for every mesh node at this timestamp.
        scores_ref = (x_ref - ref_pca.mean_) @ ref_pca.components_[:k].T
        scores_other = (x_other - other_pca.mean_) @ other_pca.components_[:k].T

        n_rows += scores_ref.shape[0]
        sum_ref += scores_ref.sum(axis=0)
        sum_other += scores_other.sum(axis=0)
        sum_ref_sq += np.square(scores_ref).sum(axis=0)
        sum_other_sq += np.square(scores_other).sum(axis=0)
        sum_cross += scores_ref.T @ scores_other

    if n_rows == 0:
        raise ValueError("No valid paired samples for correlation analysis.")

    mean_ref = sum_ref / n_rows
    mean_other = sum_other / n_rows

    covariance = (
        sum_cross / n_rows
        - np.outer(mean_ref, mean_other)
    )
    std_ref = np.sqrt(np.maximum(sum_ref_sq / n_rows - mean_ref**2, 1e-12))
    std_other = np.sqrt(np.maximum(sum_other_sq / n_rows - mean_other**2, 1e-12))

    correlation = covariance / np.outer(std_ref, std_other)
    return correlation, common_timestamps


def save_alignment_to_reference(
    reference_layer: int,
    all_pcas: dict,
    valid_files: dict,
    n_compare_pcs: int,
    out_dir: str,
):
    """
    Compare every layer's PC score space against reference_layer.

    Outputs:
      - correlation matrices
      - PC matching CSV
      - heatmaps
      - subspace singular values
    """
    ref_pca = all_pcas[reference_layer]
    ref_files = valid_files[reference_layer]

    summary_rows = []

    for layer, pca in sorted(all_pcas.items()):
        if layer == reference_layer:
            continue

        corr, common_timestamps = score_correlation_between_layers(
            ref_pca=ref_pca,
            other_pca=pca,
            ref_files=ref_files,
            other_files=valid_files[layer],
            n_compare_pcs=n_compare_pcs,
        )

        np.save(
            os.path.join(
                out_dir,
                f"pc_score_correlation_layer{reference_layer:04d}"
                f"_vs_layer{layer:04d}.npy",
            ),
            corr,
        )

        # Sign is arbitrary in PCA. Match components using absolute correlation.
        ref_indices, layer_indices = linear_sum_assignment(-np.abs(corr))

        csv_path = os.path.join(
            out_dir,
            f"pc_matching_layer{reference_layer:04d}_vs_layer{layer:04d}.csv",
        )
        with open(csv_path, "w") as f:
            f.write(
                "reference_pc,matched_layer_pc,signed_correlation,"
                "absolute_correlation,sign_to_align\n"
            )
            for ref_idx, layer_idx in zip(ref_indices, layer_indices):
                signed_corr = corr[ref_idx, layer_idx]
                sign = 1 if signed_corr >= 0 else -1
                f.write(
                    f"{ref_idx + 1},{layer_idx + 1},{signed_corr:.8f},"
                    f"{abs(signed_corr):.8f},{sign}\n"
                )

                summary_rows.append(
                    (
                        layer,
                        ref_idx + 1,
                        layer_idx + 1,
                        signed_corr,
                        abs(signed_corr),
                    )
                )

        # Singular values summarize similarity between the first-k score subspaces.
        # Values near 1 indicate closely aligned subspaces.
        subspace_singular_values = np.linalg.svd(corr, compute_uv=False)
        np.save(
            os.path.join(
                out_dir,
                f"subspace_singular_values_layer{reference_layer:04d}"
                f"_vs_layer{layer:04d}.npy",
            ),
            subspace_singular_values,
        )

        plt.figure(figsize=(8, 7))
        plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        plt.colorbar(label="PC score correlation")
        plt.xlabel(f"Layer {layer} PC")
        plt.ylabel(f"Layer {reference_layer} PC")
        plt.title(
            f"PC-score correlation: layer {reference_layer} vs layer {layer}\n"
            f"({len(common_timestamps)} common timestamps)"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                out_dir,
                f"pc_score_correlation_layer{reference_layer:04d}"
                f"_vs_layer{layer:04d}.png",
            ),
            dpi=300,
        )
        plt.close()

    with open(
        os.path.join(out_dir, f"all_matches_to_layer{reference_layer:04d}.csv"),
        "w",
    ) as f:
        f.write(
            "layer,reference_pc,matched_layer_pc,signed_correlation,"
            "absolute_correlation\n"
        )
        for row in summary_rows:
            f.write(
                f"{row[0]},{row[1]},{row[2]},{row[3]:.8f},{row[4]:.8f}\n"
            )


def run_layerwise_pca(
    acts_dir: str,
    out_dir: str,
    n_components: int = 100,
    batch_size_files: int = 10,
    reference_layer: int = 8,
    n_compare_pcs: int = 20,
):
    os.makedirs(out_dir, exist_ok=True)

    files_by_layer = discover_files(acts_dir)
    layers = sorted(files_by_layer)

    print(f"Detected layers: {layers}")
    for layer in layers:
        print(f"Layer {layer:04d}: {len(files_by_layer[layer])} files")

    if reference_layer not in files_by_layer:
        raise ValueError(
            f"Reference layer {reference_layer} was not found. "
            f"Available layers: {layers}"
        )

    all_pcas = {}
    valid_files = {}

    for layer in layers:
        print(f"\n{'=' * 80}\nFitting PCA for layer {layer:04d}\n{'=' * 80}")

        pca, skipped = fit_layer_pca(
            layer=layer,
            files_by_timestamp=files_by_layer[layer],
            n_components=n_components,
            batch_size_files=batch_size_files,
            out_dir=out_dir,
        )

        all_pcas[layer] = pca
        valid_files[layer] = {
            timestamp: path
            for timestamp, path in files_by_layer[layer].items()
            if timestamp not in skipped
        }

    plot_explained_variance(all_pcas, out_dir)

    save_alignment_to_reference(
        reference_layer=reference_layer,
        all_pcas=all_pcas,
        valid_files=valid_files,
        n_compare_pcs=n_compare_pcs,
        out_dir=out_dir,
    )

    print(f"\nFinished. Results written to: {out_dir}")


if __name__ == "__main__":
    ACTS_DIR = (
        "/share/prj-4d/graphcast_shared/data/"
        "graphcast_activations_all_layers_2019"
    )
    OUT_DIR = (
        "/share/prj-4d/graphcast_shared/data/pca_components/512_PCs/per_layer/"
        "layerwise_2019"
    )

    run_layerwise_pca(
        acts_dir=ACTS_DIR,
        out_dir=OUT_DIR,
        n_components=512,       # PCA basis saved for each layer
        batch_size_files=10,    # 10 activation files per IncrementalPCA update
        reference_layer=8,      # compare all layers to your existing focus layer
        n_compare_pcs=20,       # stability analysis for the leading PCs
    )