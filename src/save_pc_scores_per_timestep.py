import os
from glob import glob
import numpy as np


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


def project_activations_to_pcs_incremental(
    acts_dir: str,
    pca_components_path: str,
    pca_mean_path: str,
    out_path: str,
    n_pcs: int = 400,
    skip_nan_files: bool = True,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    pca_components = np.load(pca_components_path).astype(np.float32)[:n_pcs]
    pca_mean = np.load(pca_mean_path).astype(np.float32)

    files = sorted(glob(os.path.join(acts_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {acts_dir}")

    print(f"Found {len(files)} activation files")
    print(f"PCA components: {pca_components.shape}")
    print(f"PCA mean: {pca_mean.shape}")

    # First pass: determine valid files and shape
    valid_files = []
    skipped_files = []
    n_nodes = None

    for f in files:
        X = load_activations(f)

        if X.shape[1] != pca_mean.shape[0]:
            raise ValueError(
                f"Feature mismatch in {os.path.basename(f)}: "
                f"activation features={X.shape[1]}, PCA mean={pca_mean.shape[0]}"
            )

        if n_nodes is None:
            n_nodes = X.shape[0]
        elif X.shape[0] != n_nodes:
            raise ValueError(
                f"Node mismatch in {os.path.basename(f)}: "
                f"{X.shape[0]} vs expected {n_nodes}"
            )

        nan_count = int(np.isnan(X).sum())
        if nan_count > 0:
            msg = f"{os.path.basename(f)} contains {nan_count} NaNs"
            if skip_nan_files:
                print(f"Skipping: {msg}")
                skipped_files.append((f, nan_count))
                continue
            else:
                raise ValueError(msg)

        valid_files.append(f)

    if not valid_files:
        raise ValueError("No valid activation files found.")

    print(f"Valid files: {len(valid_files)}")
    print(f"Output shape: ({len(valid_files)}, {n_nodes}, {n_pcs})")

    # Create .npy file on disk and write incrementally
    pc_scores = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(valid_files), n_nodes, n_pcs),
    )

    for t_idx, f in enumerate(valid_files):
        X = load_activations(f)

        scores = (X - pca_mean) @ pca_components.T
        pc_scores[t_idx] = scores.astype(np.float32)

        if (t_idx + 1) % 10 == 0 or t_idx == 0 or t_idx == len(valid_files) - 1:
            pc_scores.flush()
            print(
                f"[{t_idx + 1}/{len(valid_files)}] saved "
                f"{os.path.basename(f)} -> {scores.shape}"
            )

    pc_scores.flush()

    meta_path = out_path.replace(".npy", "_files.txt")
    with open(meta_path, "w") as f:
        for path in valid_files:
            f.write(path + "\n")

    skipped_path = out_path.replace(".npy", "_skipped_files.txt")
    with open(skipped_path, "w") as f:
        for path, nan_count in skipped_files:
            f.write(f"{path}\t{nan_count}\n")

    print("\nDone.")
    print(f"Saved PC scores to: {out_path}")
    print(f"Shape: ({len(valid_files)}, {n_nodes}, {n_pcs}) = [time, nodes, pcs]")
    print(f"Saved timestep file order to: {meta_path}")
    print(f"Saved skipped files list to: {skipped_path}")


if __name__ == "__main__":
    ACTS_DIR = "/share/prj-4d/graphcast_shared/data/graphcast_activation_2021"
    PCA_COMPONENTS_PATH = "/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021.npy"
    PCA_MEAN_PATH = "/share/prj-4d/graphcast_shared/data/pca_components/pca_mean_2021.npy"

    OUT_PATH = "/share/prj-4d/graphcast_shared/data/pc_scores_2021_per_timestep.npy"

    project_activations_to_pcs_incremental(
        acts_dir=ACTS_DIR,
        pca_components_path=PCA_COMPONENTS_PATH,
        pca_mean_path=PCA_MEAN_PATH,
        out_path=OUT_PATH,
        n_pcs=400,
        skip_nan_files=True,
    )