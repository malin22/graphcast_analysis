#!/usr/bin/env python3
import argparse
import gc
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PCA_COMPONENTS_PATH = Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_components_2021.npy")
PCA_MEAN_PATH = Path("/share/prj-4d/graphcast_shared/data/pca_components/pca_mean_2021.npy")

DEFAULT_ACTIVATION_TEMPLATE = "/share/prj-4d/graphcast_shared/data/graphcast_activation_{year}"
DEFAULT_OUTPUT_DIR = Path("plots/sabines_experiments/pc_temporal_similarity")


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
    return np.datetime64(path.stem.split("_t")[-1])


def list_activation_files_for_year(year: int, activation_template: str):
    act_dir = Path(activation_template.format(year=year))
    files = sorted(act_dir.glob("layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t*.npy"))
    if not files:
        raise FileNotFoundError(f"No layer0008 mesh-node activation files found for {year} in {act_dir}")
    return sorted([(parse_center_time(path), path) for path in files], key=lambda x: x[0])


def project_activation_to_pcs(activations, pca_mean, pca_components, pc_indices):
    if activations.shape[1] != pca_mean.shape[0]:
        raise ValueError(
            f"Activation feature dimension {activations.shape[1]} does not match "
            f"PCA mean dimension {pca_mean.shape[0]}"
        )
    centered = activations - pca_mean
    scores = centered @ pca_components[pc_indices].T
    return scores.T.astype(np.float32)


def daily_pc_maps_for_year(year, activation_template, pca_mean, pca_components, pc_indices):
    rows = list_activation_files_for_year(year, activation_template)
    by_day = defaultdict(list)
    by_day_paths = defaultdict(list)

    for timestamp, path in rows:
        day = np.datetime64(timestamp, "D")
        activations = load_activation_matrix(path)
        pc_maps = project_activation_to_pcs(
            activations=activations,
            pca_mean=pca_mean,
            pca_components=pca_components,
            pc_indices=pc_indices,
        )

        by_day[day].append(pc_maps)
        by_day_paths[day].append((timestamp, path, pc_maps.shape))

        del activations, pc_maps
        gc.collect()

    daily_dates = sorted(by_day.keys())
    daily_maps = []

    for day in daily_dates:
        shapes = [m.shape for m in by_day[day]]
        if len(set(shapes)) != 1:
            print(f"Shape mismatch on {day}: {sorted(set(shapes))}")
            for timestamp, path, shape in by_day_paths[day]:
                print(f"  {timestamp} {shape} {path}")
            raise ValueError(f"Shape mismatch on {day}")

        maps = np.stack(by_day[day], axis=0)
        daily_maps.append(np.nanmean(maps, axis=0).astype(np.float32))

    return np.asarray(daily_dates), np.stack(daily_maps, axis=0)


def day_of_year_365(date):
    date_d = np.datetime64(date, "D")
    year = int(str(date_d)[:4])

    if str(date_d)[5:10] == "02-29":
        return None

    year_start = np.datetime64(f"{year}-01-01", "D")
    doy = int((date_d - year_start).astype(int))

    is_leap = (np.datetime64(f"{year + 1}-01-01", "D") - year_start).astype(int) == 366
    if is_leap and date_d > np.datetime64(f"{year}-02-29", "D"):
        doy -= 1

    return doy


def build_dayofyear_climatology(years, activation_template, pca_mean, pca_components, pc_indices):
    doy_maps = defaultdict(list)

    for year in years:
        print(f"\nProcessing {year} for day-of-year climatology")
        dates, daily_maps = daily_pc_maps_for_year(
            year=year,
            activation_template=activation_template,
            pca_mean=pca_mean,
            pca_components=pca_components,
            pc_indices=pc_indices,
        )

        print(f"  daily_maps shape: {daily_maps.shape}")

        for date, maps in zip(dates, daily_maps):
            doy = day_of_year_365(date)
            if doy is not None:
                doy_maps[doy].append(maps)

    available = sorted(doy_maps.keys())
    if len(available) < 300:
        raise ValueError(f"Only {len(available)} day-of-year bins available")

    n_pcs, n_nodes = doy_maps[available[0]][0].shape
    clim_maps = np.full((365, n_pcs, n_nodes), np.nan, dtype=np.float32)
    doy_counts = np.zeros(365, dtype=np.int32)

    for doy in range(365):
        if doy not in doy_maps:
            continue
        maps = np.stack(doy_maps[doy], axis=0)
        clim_maps[doy] = np.nanmean(maps, axis=0).astype(np.float32)
        doy_counts[doy] = maps.shape[0]

    return clim_maps, doy_counts


def load_all_daily_maps(years, activation_template, pca_mean, pca_components, pc_indices):
    all_dates = []
    all_maps = []

    for year in years:
        print(f"\nProcessing {year} for all-days circular sequence")
        dates, daily_maps = daily_pc_maps_for_year(
            year=year,
            activation_template=activation_template,
            pca_mean=pca_mean,
            pca_components=pca_components,
            pc_indices=pc_indices,
        )

        print(f"  daily_maps shape: {daily_maps.shape}")
        all_dates.append(dates)
        all_maps.append(daily_maps)

    all_dates = np.concatenate(all_dates, axis=0)
    all_maps = np.concatenate(all_maps, axis=0)

    order = np.argsort(all_dates)
    return all_dates[order], all_maps[order]


def normalize_spatial_maps(maps, center=True, eps=1e-8):
    X = maps.astype(np.float64, copy=False)
    if center:
        X = X - np.nanmean(X, axis=2, keepdims=True)
    norms = np.sqrt(np.nansum(X * X, axis=2, keepdims=True))
    return (X / np.maximum(norms, eps)).astype(np.float32)


def circular_similarity_by_lag(normalized_maps, max_lag_days):
    n_days, n_pcs, _ = normalized_maps.shape
    max_lag_days = min(max_lag_days, n_days - 1)

    lags = np.arange(max_lag_days + 1, dtype=np.int32)
    sim = np.full((n_pcs, len(lags)), np.nan, dtype=np.float32)
    pair_counts = np.full(len(lags), n_days, dtype=np.int32)

    for lag_idx, lag in enumerate(lags):
        shifted = np.roll(normalized_maps, shift=-lag, axis=0)
        dots = np.sum(normalized_maps * shifted, axis=2)
        sim[:, lag_idx] = np.nanmean(dots, axis=0).astype(np.float32)

    return lags, sim, pair_counts


def plot_circular_lag_similarity(lags, sim, pair_counts, pc_labels, output_path):
    fig, ax1 = plt.subplots(figsize=(14, 6))

    for pc_idx, label in enumerate(pc_labels):
        ax1.plot(lags, sim[pc_idx], linewidth=2, label=label)

    ax1.axhline(0.0, color="black", linewidth=1, alpha=0.4)

    for marker in [182, 365, 547, 730]:
        if marker <= lags[-1]:
            linestyle = "--" if marker in [365, 730] else ":"
            ax1.axvline(marker, color="gray", linestyle=linestyle, linewidth=1, alpha=0.45)

    ax1.set_xlabel("Circular lag [days]")
    ax1.set_ylabel("Mean centered cosine similarity")
    ax1.set_title("Circular lag recurrence of PC spatial patterns")
    ax1.legend(ncol=2, fontsize=8, loc="upper right")

    ax2 = ax1.twinx()
    ax2.plot(lags, pair_counts, color="black", alpha=0.15, linewidth=1.5)
    ax2.set_ylabel("Number of day pairs")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_dayofyear_counts(doy_counts, output_path):
    plt.figure(figsize=(13, 3))
    plt.bar(np.arange(365), doy_counts, width=1.0)
    plt.xlabel("Day of year")
    plt.ylabel("Number of years available")
    plt.title("Samples per day-of-year bin")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute circular PC spatial-pattern recurrence."
    )
    parser.add_argument("--years", type=int, nargs="+", default=[2020, 2021])
    parser.add_argument("--activation-template", default=DEFAULT_ACTIVATION_TEMPLATE)
    parser.add_argument("--pc-indices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--max-lag-days", type=int, default=730)
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--also-plot-climatology-counts",
        action="store_true",
        help="Also build day-of-year climatology and plot sample counts.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pca_components = np.load(PCA_COMPONENTS_PATH)
    pca_mean = np.load(PCA_MEAN_PATH)

    pc_indices = np.asarray(args.pc_indices, dtype=np.int64)
    pc_labels = [f"PC{i + 1}" for i in pc_indices]

    all_dates, all_maps = load_all_daily_maps(
        years=args.years,
        activation_template=args.activation_template,
        pca_mean=pca_mean,
        pca_components=pca_components,
        pc_indices=pc_indices,
    )

    normalized_all_maps = normalize_spatial_maps(
        all_maps,
        center=not args.no_center,
    )

    lags, sim, pair_counts = circular_similarity_by_lag(
        normalized_all_maps,
        max_lag_days=args.max_lag_days,
    )

    np.savez(
        args.output_dir / "pc_circular_over_all_years.npz",
        dates=all_dates,
        lags=lags,
        sim=sim,
        pair_counts=pair_counts,
        pc_indices=pc_indices,
        years=np.asarray(args.years),
    )

    plot_circular_lag_similarity(
        lags,
        sim,
        pair_counts,
        pc_labels,
        args.output_dir / "pc_circular_over_all_years.png",
    )

    if args.also_plot_climatology_counts:
        _, doy_counts = build_dayofyear_climatology(
            years=args.years,
            activation_template=args.activation_template,
            pca_mean=pca_mean,
            pca_components=pca_components,
            pc_indices=pc_indices,
        )
        plot_dayofyear_counts(
            doy_counts,
            args.output_dir / "pc_dayofyear_counts.png",
        )

    print(f"\nSaved circular recurrence plot to {args.output_dir}")


if __name__ == "__main__":
    main()