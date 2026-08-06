#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

'''
python pc_correlation_dropoff_diagnostics.py \
  --input-json /home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/mapping_experiments/top_512_pcs/correlation_pc_era5_mesh_m6_screening_cache.json \
  --out-dir /home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/mapping_experiments/top_512_pcs/dropoff_diagnostics

'''


DEFAULT_INPUT = Path(
    "/home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/"
    "mapping_experiments/top_512_pcs/correlation_pc_era5_mesh_m6_screening_cache.json"
)

DEFAULT_OUTPUT = Path("/home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/mapping_experiments/top_512_pcs/dropoff_diagnostics")


def pc_sort_key(pc_name):
    m = re.search(r"(\d+)", pc_name)
    return int(m.group(1)) if m else 10**9


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def rolling_mean(x, window):
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or len(x) < window:
        return x

    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def extract_topk_rows(data, top_k=3):
    rows = []

    for pc_name in sorted(data.keys(), key=pc_sort_key):
        pc_idx = pc_sort_key(pc_name)
        pc_data = data[pc_name]

        ranked = pc_data.get("ranked_variables", pc_data.get("top_variables", []))
        ranked = sorted(
            ranked,
            key=lambda r: float(r.get("mean_abs_r", np.nan)),
            reverse=True,
        )

        top = ranked[:top_k]

        row = {
            "pc_name": pc_name,
            "pc_index": pc_idx,
            "total_timesteps": pc_data.get("total_timesteps"),
        }

        for k in range(top_k):
            if k < len(top):
                row[f"top{k+1}_variable"] = top[k].get("variable")
                row[f"top{k+1}_mean_abs_r"] = float(top[k].get("mean_abs_r", np.nan))
                row[f"top{k+1}_wins"] = int(top[k].get("wins", 0))
                row[f"top{k+1}_count"] = int(top[k].get("count", 0))
            else:
                row[f"top{k+1}_variable"] = None
                row[f"top{k+1}_mean_abs_r"] = np.nan
                row[f"top{k+1}_wins"] = 0
                row[f"top{k+1}_count"] = 0

        rows.append(row)

    return rows


def find_dropoff_pc(pc_indices, top1, threshold=0.2, window=15):
    """
    First PC where rolling top1 correlation falls below threshold and stays
    below threshold for at least `window` PCs.
    """
    top1 = np.asarray(top1, dtype=np.float64)
    smoothed = rolling_mean(top1, window)

    for i in range(len(smoothed) - window + 1):
        if np.all(smoothed[i:i + window] < threshold):
            return int(pc_indices[i]), float(smoothed[i])

    return None, None


def interesting_pcs(rows, high_threshold=0.4, late_start=100, top_n=30):
    """
    Return PCs that are globally high or surprisingly high late in the spectrum.
    """
    ranked = sorted(
        rows,
        key=lambda r: float(r["top1_mean_abs_r"]),
        reverse=True,
    )

    high = [
        r for r in rows
        if np.isfinite(r["top1_mean_abs_r"])
        and r["top1_mean_abs_r"] >= high_threshold
    ]

    late = [
        r for r in rows
        if r["pc_index"] >= late_start
        and np.isfinite(r["top1_mean_abs_r"])
    ]
    late = sorted(late, key=lambda r: r["top1_mean_abs_r"], reverse=True)[:top_n]

    return {
        "high_threshold": high,
        "top_overall": ranked[:top_n],
        "top_late": late,
    }


def save_summary_json(rows, output_path, thresholds):
    pc_indices = np.asarray([r["pc_index"] for r in rows], dtype=int)
    top1 = np.asarray([r["top1_mean_abs_r"] for r in rows], dtype=float)
    top2 = np.asarray([r["top2_mean_abs_r"] for r in rows], dtype=float)
    top3 = np.asarray([r["top3_mean_abs_r"] for r in rows], dtype=float)

    summary = {
        "n_pcs": int(len(rows)),
        "max_top1_mean_abs_r": float(np.nanmax(top1)),
        "median_top1_mean_abs_r": float(np.nanmedian(top1)),
        "median_top1_last_100_pcs": float(np.nanmedian(top1[-100:])),
        "median_top3_last_100_pcs": float(np.nanmedian(top3[-100:])),
        "threshold_counts": {},
        "dropoff_estimates": {},
        "interesting_pcs": interesting_pcs(rows),
    }

    for threshold in thresholds:
        summary["threshold_counts"][str(threshold)] = {
            "top1_above": int(np.sum(top1 >= threshold)),
            "top2_above": int(np.sum(top2 >= threshold)),
            "top3_above": int(np.sum(top3 >= threshold)),
        }

        drop_pc, smooth_value = find_dropoff_pc(
            pc_indices,
            top1,
            threshold=threshold,
            window=15,
        )
        summary["dropoff_estimates"][str(threshold)] = {
            "first_pc_where_rolling_top1_stays_below_threshold": drop_pc,
            "rolling_value_at_dropoff": smooth_value,
            "window": 15,
        }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {output_path}")


def save_rows_csv(rows, output_path, top_k=3):
    with open(output_path, "w") as f:
        columns = ["pc_name", "pc_index", "total_timesteps"]
        for k in range(top_k):
            columns.extend([
                f"top{k+1}_variable",
                f"top{k+1}_mean_abs_r",
                f"top{k+1}_wins",
                f"top{k+1}_count",
            ])

        f.write(",".join(columns) + "\n")

        for r in rows:
            vals = [r.get(c, "") for c in columns]
            f.write(",".join(str(v) for v in vals) + "\n")

    print(f"Saved {output_path}")


def plot_pc_dropoff(rows, output_path, rolling_window=15, thresholds=(0.2, 0.3, 0.4)):
    pc_indices = np.asarray([r["pc_index"] for r in rows], dtype=int)
    top1 = np.asarray([r["top1_mean_abs_r"] for r in rows], dtype=float)
    top2 = np.asarray([r["top2_mean_abs_r"] for r in rows], dtype=float)
    top3 = np.asarray([r["top3_mean_abs_r"] for r in rows], dtype=float)

    plt.figure(figsize=(16, 6))

    plt.plot(pc_indices, top1, linewidth=1.2, alpha=0.9, label="Top 1")
    plt.plot(pc_indices, top2, linewidth=1.0, alpha=0.65, label="Top 2")
    plt.plot(pc_indices, top3, linewidth=1.0, alpha=0.65, label="Top 3")

    plt.plot(
        pc_indices,
        rolling_mean(top1, rolling_window),
        color="black",
        linewidth=2.2,
        label=f"Top 1 rolling mean ({rolling_window})",
    )

    for threshold in thresholds:
        plt.axhline(
            threshold,
            color="gray",
            linestyle="--",
            linewidth=0.9,
            alpha=0.55,
        )
        plt.text(
            pc_indices[-1] + 3,
            threshold,
            f"{threshold:.2f}",
            va="center",
            fontsize=9,
            color="gray",
        )

    plt.xlabel("PC index")
    plt.ylabel("Mean absolute spatial Pearson r")
    plt.title("Top ERA5 correlation strength across PCA components")
    plt.ylim(0, min(1.0, np.nanmax(top1) * 1.12))
    plt.xlim(pc_indices[0], pc_indices[-1] + 25)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_path}")


def plot_histogram(rows, output_path, bins=40):
    top1 = np.asarray([r["top1_mean_abs_r"] for r in rows], dtype=float)
    top2 = np.asarray([r["top2_mean_abs_r"] for r in rows], dtype=float)
    top3 = np.asarray([r["top3_mean_abs_r"] for r in rows], dtype=float)

    plt.figure(figsize=(9, 5))
    plt.hist(top1, bins=bins, alpha=0.65, label="Top 1")
    plt.hist(top2, bins=bins, alpha=0.45, label="Top 2")
    plt.hist(top3, bins=bins, alpha=0.45, label="Top 3")

    plt.xlabel("Mean absolute spatial Pearson r")
    plt.ylabel("Number of PCs")
    plt.title("Distribution of top ERA5 correlations across PCs")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_path}")


def plot_late_outliers(rows, output_path, late_start=100, top_n=30):
    late = [
        r for r in rows
        if r["pc_index"] >= late_start and np.isfinite(r["top1_mean_abs_r"])
    ]
    late = sorted(late, key=lambda r: r["top1_mean_abs_r"], reverse=True)[:top_n]
    late = sorted(late, key=lambda r: r["top1_mean_abs_r"])

    labels = [
        f"{r['pc_name']}: {r['top1_variable']}"
        for r in late
    ]
    values = [r["top1_mean_abs_r"] for r in late]

    plt.figure(figsize=(10, max(6, 0.28 * len(late))))
    plt.barh(np.arange(len(late)), values)
    plt.yticks(np.arange(len(late)), labels, fontsize=8)
    plt.xlabel("Top mean absolute spatial Pearson r")
    plt.title(f"Highest-correlating later PCs (PC >= {late_start})")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose where PC-to-ERA5 correlations drop into noise."
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rolling-window", type=int, default=15)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.2, 0.3, 0.4])
    parser.add_argument("--late-start", type=int, default=100)
    parser.add_argument("--late-top-n", type=int, default=30)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(args.input_json)
    rows = extract_topk_rows(data, top_k=args.top_k)

    save_rows_csv(
        rows,
        args.out_dir / "pc_top_correlations.csv",
        top_k=args.top_k,
    )

    save_summary_json(
        rows,
        args.out_dir / "pc_correlation_dropoff_summary.json",
        thresholds=args.thresholds,
    )

    plot_pc_dropoff(
        rows,
        args.out_dir / "pc_correlation_dropoff_top3_by_pc.png",
        rolling_window=args.rolling_window,
        thresholds=args.thresholds,
    )

    plot_histogram(
        rows,
        args.out_dir / "pc_top_correlation_histogram.png",
    )

    plot_late_outliers(
        rows,
        args.out_dir / "late_pc_high_correlation_outliers.png",
        late_start=args.late_start,
        top_n=args.late_top_n,
    )


if __name__ == "__main__":
    main()