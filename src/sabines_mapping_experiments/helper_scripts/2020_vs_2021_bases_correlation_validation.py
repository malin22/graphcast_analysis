from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


PCA_2020_PATH = Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2020_layer8.npy")
PCA_2021_PATH = Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021_layer8.npy")

OUT_DIR = Path(
    "/home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/mapping_experiments/2020_vs_2021_PC_validation"
)


def load_components(path: Path) -> np.ndarray:
    X = np.load(path)
    if X.ndim != 2:
        raise ValueError(f"Expected [n_pcs, n_features] array in {path}, got {X.shape}")
    return X.astype(np.float64)


def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = normalize_rows(A)
    B = normalize_rows(B)
    return A @ B.T


def basis_overlap_summary(sim: np.ndarray) -> dict[str, float]:
    """
    For a quick sanity check:
      - same_index_abs_mean: mean |corr| along the diagonal
      - best_match_abs_mean: mean of each 2020 PC's best absolute match in 2021
      - matched_abs_mean: mean abs corr after greedy same-index reading

    Note:
      The most mathematically sound *subspace* comparison is the singular values
      of the overlap matrix. That is included too, but for a full 400-PC basis
      the diagonal and best-match views are usually more interpretable.
    """
    abs_sim = np.abs(sim)
    diag_abs = np.abs(np.diag(sim))
    best_match_abs = abs_sim.max(axis=1)

    # Greedy same-index view, but keep the row-wise best for sanity.
    same_index_abs_mean = float(diag_abs.mean())
    best_match_abs_mean = float(best_match_abs.mean())

    # Subspace overlap singular values. This is the strongest "same subspace?" check.
    svals = np.linalg.svd(sim, compute_uv=False)
    principal_angle_mean_cos = float(svals.mean())
    principal_angle_min_cos = float(svals.min())

    return {
        "same_index_abs_mean": same_index_abs_mean,
        "best_match_abs_mean": best_match_abs_mean,
        "principal_angle_mean_cos": principal_angle_mean_cos,
        "principal_angle_min_cos": principal_angle_min_cos,
    }


def make_heatmap(sim: np.ndarray, out_path: Path) -> None:
    abs_sim = np.abs(sim)
    vmax = max(float(np.nanmax(abs_sim)), 1e-6)

    plt.figure(figsize=(12, 10))
    im = plt.imshow(abs_sim, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    plt.colorbar(im, label="Absolute cosine similarity")
    plt.xlabel("2021 layer 8 PC")
    plt.ylabel("2020 layer 8 PC")
    plt.title("2020 vs 2021 PCA basis overlap, layer 8")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X20 = load_components(PCA_2020_PATH)
    X21 = load_components(PCA_2021_PATH)

    if X20.shape[1] != X21.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: 2020={X20.shape[1]} vs 2021={X21.shape[1]}"
        )

    n_pcs = min(X20.shape[0], X21.shape[0], 400)
    X20 = X20[:n_pcs]
    X21 = X21[:n_pcs]

    sim = cosine_matrix(X20, X21)
    abs_sim = np.abs(sim)

    summary = basis_overlap_summary(sim)

    # Per-PC summary table
    best_j = abs_sim.argmax(axis=1)
    rows = []
    for i in range(n_pcs):
        j = int(best_j[i])
        rows.append(
            {
                "pc_2020": i + 1,
                "best_pc_2021": j + 1,
                "signed_cosine_same_index": float(sim[i, i]),
                "abs_cosine_same_index": float(abs(sim[i, i])),
                "signed_cosine_best_match": float(sim[i, j]),
                "abs_cosine_best_match": float(abs_sim[i, j]),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "pca_basis_2020_vs_2021_layer8_table.csv"
    md_path = OUT_DIR / "pca_basis_2020_vs_2021_layer8_table.md"
    summary_path = OUT_DIR / "pca_basis_2020_vs_2021_layer8_summary.txt"
    heatmap_path = OUT_DIR / "pca_basis_2020_vs_2021_layer8_abs_cosine_heatmap.png"
    npy_path = OUT_DIR / "pca_basis_2020_vs_2021_layer8_abs_cosine_matrix.npy"

    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    summary_text = (
        "2020 vs 2021 layer-8 PCA basis validation\n"
        f"2020 shape: {X20.shape}\n"
        f"2021 shape: {X21.shape}\n"
        f"same_index_abs_mean: {summary['same_index_abs_mean']:.6f}\n"
        f"best_match_abs_mean: {summary['best_match_abs_mean']:.6f}\n"
        f"principal_angle_mean_cos: {summary['principal_angle_mean_cos']:.6f}\n"
        f"principal_angle_min_cos: {summary['principal_angle_min_cos']:.6f}\n"
    )
    summary_path.write_text(summary_text, encoding="utf-8")
    np.save(npy_path, abs_sim)
    make_heatmap(sim, heatmap_path)

    print(summary_text)
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {heatmap_path}")


if __name__ == "__main__":
    main()
