#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA


def load_activations(path: Path) -> np.ndarray:
    """Load one activation file as [nodes, features] float32."""
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


def timestamp_from_path(path: Path) -> str:
    return path.stem.split("_t")[-1]


def collect_activation_files(activation_dirs, pattern: str):
    files = []

    for directory in activation_dirs:
        directory = Path(directory)
        found = sorted(directory.glob(pattern))
        print(f"Found {len(found)} files in {directory}")
        files.extend(found)

    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    return sorted(files, key=lambda p: (timestamp_from_path(p), str(p)))


def fit_pca(files, n_components: int, batch_size: int):
    ipca = IncrementalPCA(n_components=n_components)

    skipped = []
    used_files = []
    n_samples = 0

    for start in range(0, len(files), batch_size):
        batch_files = files[start:start + batch_size]
        batch_data = []

        for path in batch_files:
            data = load_activations(path)

            if not np.isfinite(data).all():
                skipped.append(str(path))
                print(f"Skipping non-finite activation file: {path.name}")
                continue

            batch_data.append(data)
            used_files.append(str(path))
            n_samples += data.shape[0]

        if not batch_data:
            continue

        batch = np.vstack(batch_data)
        print(
            f"Fitting batch {start // batch_size + 1}: "
            f"{batch.shape}, files={len(batch_data)}"
        )
        ipca.partial_fit(batch)

    if not used_files:
        raise RuntimeError("No valid activation files were available for PCA.")

    return ipca, used_files, skipped, n_samples

def run_pca(
    activation_dirs,
    out_dir,
    tag="2019_2020_layer8",
    n_components=512,
    batch_size=10,
    layer_pattern="layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t*.npy",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_activation_files(
        activation_dirs,
        layer_pattern,
    )
    print(f"\nTotal activation files: {len(files)}")

    ipca, used_files, skipped_files, n_samples = fit_pca(
        files=files,
        n_components=n_components,
        batch_size=batch_size,
    )

    components_path = out_dir / f"pca_components_{tag}.npy"
    mean_path = out_dir / f"pca_mean_{tag}.npy"
    variance_path = out_dir / f"pca_explained_variance_ratio_{tag}.npy"
    metadata_path = out_dir / f"pca_metadata_{tag}.json"

    np.save(components_path, ipca.components_.astype(np.float32))
    np.save(mean_path, ipca.mean_.astype(np.float32))
    np.save(variance_path, ipca.explained_variance_ratio_)

    metadata = {
        "n_components": n_components,
        "n_samples": n_samples,
        "n_files_used": len(used_files),
        "n_files_skipped": len(skipped_files),
        "activation_dirs": [str(path) for path in activation_dirs],
        "layer_pattern": layer_pattern,
        "used_files": used_files,
        "skipped_files": skipped_files,
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved PCA outputs:")
    print(f"  Components: {components_path}")
    print(f"  Mean:       {mean_path}")
    print(f"  Variance:   {variance_path}")
    print(f"  Metadata:   {metadata_path}")

    print("\nCumulative explained variance:")
    print(np.cumsum(ipca.explained_variance_ratio_))

    return ipca

if __name__ == "__main__":
    run_pca(
        activation_dirs=[
            "/share/prj-4d/graphcast_shared/data/graphcast_activation_2019",
            "/share/prj-4d/graphcast_shared/data/graphcast_activation_2020",
        ],
        out_dir=(
            "/share/prj-4d/graphcast_shared/data/"
            "pca_components/512_PCs/layer8_only"
        ),
        tag="2019_2020_layer8",
        n_components=512,
        batch_size=10,
        layer_pattern="layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t*.npy",
    )