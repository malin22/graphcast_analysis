#!/usr/bin/env python3
import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PCA_COMPONENTS_PATH = Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021.npy")
PCA_MEAN_PATH = Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_mean_2021.npy")

DEFAULT_ACTIVATION_TEMPLATE = "/share/prj-4d/graphcast_shared/data/graphcast_activation_{year}"
DEFAULT_OUTPUT_DIR = Path("plots/sabines_experiments/pc_diurnal_similarity")

HOURS = [0, 6, 12, 18]


def to_float32(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == np.dtype("|V2"):
        arr = arr.view(np.float16)
    return np.asarray(arr, dtype=np.float32)


def load_activation_matrix(path: Path) -> np.ndarray:
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"Expected activation matrix [nodes, features], got {x.shape}")

    return x


def parse_center_time(path: Path) -> np.datetime64:
    center_str = path.stem.split("_t")[-1]
    return np.datetime64(center_str, "h")


def timestamp_hour(timestamp: np.datetime64) -> int:
    text = np.datetime_as_string(timestamp, unit="h")
    return int(text[-2:])


def timestamp_day(timestamp: np.datetime64) -> np.datetime64:
    return np.datetime64(timestamp, "D")


def list_activation_files(year: int, activation_template: str):
    act_dir = Path(activation_template.format(year=year))
    files = sorted(act_dir.glob("layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t*.npy"))

    if not files:
        raise FileNotFoundError(f"No layer0008 activation files found in {act_dir}")

    rows = [(parse_center_time(path), path) for path in files]
    return sorted(rows, key=lambda x: x[0])


def select_days(rows, max_days=None, start_day=None):
    if start_day is not None:
        start_day = np.datetime64(start_day, "D")
        rows = [(t, p) for t, p in rows if timestamp_day(t) >= start_day]

    days = sorted(set(timestamp_day(t) for t, _ in rows))

    if max_days is not None:
        selected_days = set(days[:max_days])
        rows = [(t, p) for t, p in rows if timestamp_day(t) in selected_days]

    return rows


def project_activation_to_pcs(activations, pca_mean, pca_components, pc_indices):
    if activations.shape[1] != pca_mean.shape[0]:
        raise ValueError(
            f"Activation feature dimension {activations.shape[1]} does not match "
            f"PCA mean dimension {pca_mean.shape[0]}"
        )

    centered = activations - pca_mean
    scores = centered @ pca_components[pc_indices].T  # [nodes, n_pcs]
    return scores.T.astype(np.float32)                # [n_pcs, nodes]


def center_and_normalize(pc_maps, eps=1e-8):
    """
    pc_maps: [n_samples, n_pcs, n_nodes]
    """
    X = pc_maps.astype(np.float64, copy=False)
    X = X - np.nanmean(X, axis=2, keepdims=True)
    norms = np.sqrt(np.nansum(X * X, axis=2, keepdims=True))
    return (X / np.maximum(norms, eps)).astype(np.float32)


def load_hour_grouped_maps(years, activation_template, pca_mean, pca_components, pc_indices, max_days, start_day):
    by_hour = {h: [] for h in HOURS}
    used_files = 0

    for year in years:
        print(f"Processing {year}")
        rows = list_activation_files(year, activation_template)
        rows = select_days(rows, max_days=max_days, start_day=start_day)

        print(f"  selected files: {len(rows)}")

        for timestamp, path in rows:
            hour = timestamp_hour(timestamp)
            if hour not in by_hour:
                continue

            activations = load_activation_matrix(path)
            pc_maps = project_activation_to_pcs(
                activations,
                pca_mean,
                pca_components,
                pc_indices,
            )

            by_hour[hour].append(pc_maps)
            used_files += 1

            del activations, pc_maps
            gc.collect()

    for hour in HOURS:
        if not by_hour[hour]:
            raise ValueError(f"No maps found for hour {hour:02d}")

        by_hour[hour] = center_and_normalize(
            np.stack(by_hour[hour], axis=0)
        )  # [n_samples, n_pcs, n_nodes]

        print(f"Hour {hour:02d}: {by_hour[hour].shape[0]} samples")

    print(f"Used files: {used_files}")
    return by_hour


def mean_pairwise_cosine(A, B):
    """
    A: [n_a, n_pcs, n_nodes]
    B: [n_b, n_pcs, n_nodes]

    Returns:
      [n_pcs] mean cosine similarity over all A/B pairs.
    """
    # Average vector dot products over all sample pairs without materializing
    # [n_a, n_b, n_pcs].
    n_pcs = A.shape[1]
    out = np.zeros(n_pcs, dtype=np.float64)

    for pc in range(n_pcs):
        # Since vectors are normalized, dot product is cosine similarity.
        dots = A[:, pc, :] @ B[:, pc, :].T
        out[pc] = np.nanmean(dots)

    return out.astype(np.float32)


def compute_hour_similarity_matrix(by_hour):
    n_pcs = next(iter(by_hour.values())).shape[1]
    sim = np.full((n_pcs, 4, 4), np.nan, dtype=np.float32)

    for i, h1 in enumerate(HOURS):
        for j, h2 in enumerate(HOURS):
            print(f"Comparing {h1:02d} vs {h2:02d}", flush=True)
            sim[:, i, j] = mean_pairwise_cosine(by_hour[h1], by_hour[h2])

    return sim


def diurnal_scores(sim):
    """
    sim: [n_pcs, 4, 4]
    """
    same_hour = np.mean(np.diagonal(sim, axis1=1, axis2=2), axis=1)

    opposite_pairs = [
        (0, 2),  # 00 vs 12
        (2, 0),  # 12 vs 00
        (1, 3),  # 06 vs 18
        (3, 1),  # 18 vs 06
    ]
    opposite = np.mean(
        np.stack([sim[:, i, j] for i, j in opposite_pairs], axis=1),
        axis=1,
    )

    contrast = same_hour - opposite

    return same_hour, opposite, contrast


def plot_pc_heatmap(sim_pc, pc_label, output_path):
    plt.figure(figsize=(5.5, 4.8))
    im = plt.imshow(sim_pc, vmin=-1, vmax=1, cmap="coolwarm")

    plt.xticks(range(4), [f"{h:02d}" for h in HOURS])
    plt.yticks(range(4), [f"{h:02d}" for h in HOURS])
    plt.xlabel("UTC hour")
    plt.ylabel("UTC hour")
    plt.title(f"{pc_label}: hour-pair cosine similarity")

    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{sim_pc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    plt.colorbar(im, label="Centered cosine similarity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_diurnal_ranking(pc_labels, same_hour, opposite, contrast, output_path):
    order = np.argsort(contrast)[::-1]

    labels = [pc_labels[i] for i in order]
    contrast_sorted = contrast[order]
    same_sorted = same_hour[order]
    opp_sorted = opposite[order]

    plt.figure(figsize=(14, 5))
    x = np.arange(len(labels))

    plt.bar(x, contrast_sorted, label="same-hour minus opposite-hour")
    plt.plot(x, same_sorted, color="black", linewidth=1, marker="o", label="same-hour")
    plt.plot(x, opp_sorted, color="gray", linewidth=1, marker="o", label="opposite-hour")

    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Cosine similarity")
    plt.title("PC diurnal-cycle contrast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Check diurnal cycle in PC activation spatial patterns."
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2021])
    parser.add_argument("--activation-template", default=DEFAULT_ACTIVATION_TEMPLATE)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--pc-indices", type=int, nargs="*", default=None)
    parser.add_argument("--max-days", type=int, default=7)
    parser.add_argument("--start-day", default=None, help="Optional YYYY-MM-DD start day")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pca_components = np.load(PCA_COMPONENTS_PATH)
    pca_mean = np.load(PCA_MEAN_PATH)

    if args.pc_indices:
        pc_indices = np.asarray(args.pc_indices, dtype=np.int64)
    else:
        pc_indices = np.arange(args.n_pcs, dtype=np.int64)

    pc_labels = [f"PC{i + 1}" for i in pc_indices]

    by_hour = load_hour_grouped_maps(
        years=args.years,
        activation_template=args.activation_template,
        pca_mean=pca_mean,
        pca_components=pca_components,
        pc_indices=pc_indices,
        max_days=args.max_days,
        start_day=args.start_day,
    )

    sim = compute_hour_similarity_matrix(by_hour)
    same_hour, opposite, contrast = diurnal_scores(sim)

    np.savez(
        args.output_dir / "pc_diurnal_similarity.npz",
        sim=sim,
        same_hour=same_hour,
        opposite=opposite,
        contrast=contrast,
        pc_indices=pc_indices,
        hours=np.asarray(HOURS),
        years=np.asarray(args.years),
    )

    plot_diurnal_ranking(
        pc_labels,
        same_hour,
        opposite,
        contrast,
        args.output_dir / "pc_diurnal_contrast_ranking.png",
    )

    top = np.argsort(contrast)[-10:][::-1]
    for idx in top:
        plot_pc_heatmap(
            sim[idx],
            pc_labels[idx],
            args.output_dir / f"{pc_labels[idx]}_diurnal_heatmap.png",
        )

    print("\nTop PCs by diurnal contrast:")
    for idx in top:
        print(
            f"{pc_labels[idx]}: "
            f"same={same_hour[idx]:.3f}, "
            f"opposite={opposite[idx]:.3f}, "
            f"contrast={contrast[idx]:.3f}"
        )

    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()