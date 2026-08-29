import os
from glob import glob

import numpy as np


ACTS_DIR = (
    "/share/prj-4d/graphcast_shared/data/"
    "graphcast_activation_2019"
)

PCA_COMPONENTS_PATH = (
    "/share/prj-4d/graphcast_shared/data/"
    "pca_components/512_PCs/layer8_only/"
    "pca_components_2019_2020_layer8.npy"
)

PCA_MEAN_PATH = (
    "/share/prj-4d/graphcast_shared/data/"
    "pca_components/512_PCs/layer8_only/"
    "pca_mean_2019_2020_layer8.npy"
)

OUT_PATH = (
    "/share/prj-4d/graphcast_shared/data/"
    "pc_scores_per_timestep/"
    "pc_scores_2019_from_2019_2020_pca_per_timestep.npy"
)

FILES_TXT_OUT = (
    "/share/prj-4d/graphcast_shared/data/"
    "pc_scores_per_timestep/"
    "pc_scores_2019_from_2019_2020_pca_per_timestep_files.txt"
)

N_COMPONENTS = 512


def load_activations(path):
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    if x.ndim != 2:
        raise ValueError(
            f"Expected [nodes, features], got {x.shape} "
            f"for {path}"
        )

    return x.astype(np.float32)


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    components = np.load(PCA_COMPONENTS_PATH).astype(np.float32)
    mean = np.load(PCA_MEAN_PATH).astype(np.float32)

    n_components = min(N_COMPONENTS, components.shape[0])
    components = components[:n_components]

    pattern = (
        "layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t*.npy"
    )

    files = sorted(glob(os.path.join(ACTS_DIR, pattern)))

    if not files:
        raise FileNotFoundError(
            f"No activation files found with pattern:\n"
            f"{os.path.join(ACTS_DIR, pattern)}"
        )

    print(f"Found {len(files)} activation files")

    valid_files = []

    # First pass: find valid files
    for path in files:
        X = load_activations(path)

        if X.shape[1] != mean.shape[0]:
            raise ValueError(
                f"Feature mismatch in {os.path.basename(path)}: "
                f"{X.shape[1]} versus PCA mean {mean.shape[0]}"
            )

        if np.isnan(X).any():
            print(
                f"[SKIP] {os.path.basename(path)} contains NaNs"
            )
            continue

        valid_files.append(path)

    if not valid_files:
        raise ValueError("No valid activation files found")

    X0 = load_activations(valid_files[0])

    output_shape = (
        len(valid_files),
        X0.shape[0],
        n_components,
    )

    print("Output shape:", output_shape)

    scores_out = np.lib.format.open_memmap(
        OUT_PATH,
        mode="w+",
        dtype=np.float32,
        shape=output_shape,
    )

    # Second pass: transform and write directly to disk
    for i, path in enumerate(valid_files):
        X = load_activations(path)

        scores_out[i] = (
            (X - mean) @ components.T
        ).astype(np.float32)

        if (i + 1) % 100 == 0:
            scores_out.flush()
            print(
                f"Processed and saved "
                f"{i + 1}/{len(valid_files)} files"
            )

    scores_out.flush()
    del scores_out

    with open(FILES_TXT_OUT, "w") as f:
        for path in valid_files:
            f.write(path + "\n")

    print("Saved scores:", OUT_PATH)
    print("Saved file list:", FILES_TXT_OUT)


if __name__ == "__main__":
    main()