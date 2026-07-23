#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


'''
To run without the slurm script: 

python -u /home/student/s/sascholle/share/graphcast_analysis/src/plotting_script_for_correlation_and_regression.py \
  --correlation-json plots/sabines_experiments/mapping_experiments/correlation_regression_json_results_depreciated/pc_era5_mesh_m5_screening_cache.json \
  --out-dir plots/sabines_experiments/mapping_experiments/histograms \
  --pcs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --per-pc \
  --per-pc-layout grid \
  --aggregation mean

'''


PRESSURE_LEVELS_HPA = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
    225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
    775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
]

KEEP_PRESSURE_LEVELS_HPA = {
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
}

ATMOSPHERIC_BASES = {
    "geopotential": "Geopotential",
    "specific_humidity": "Specific humidity",
    "temperature": "Temperature",
    "u_component_of_wind": "U wind",
    "v_component_of_wind": "V wind",
    "vertical_velocity": "Vertical velocity",
}

SINGLE_LEVEL_LABELS = {
    "2m_temperature": "2m temperature",
    "10m_u_component_of_wind": "10m U wind",
    "10m_v_component_of_wind": "10m V wind",
    "mean_sea_level_pressure": "Mean sea-level pressure",
    "total_precipitation_6hr": "Total precipitation 6h",
    "toa_incident_solar_radiation": "TOA solar radiation",
}

STATIC_LABELS = {
    "geopotential_at_surface": "Surface geopotential",
    "land_sea_mask": "Land-sea mask",
    "latitude": "Latitude",
    "latitude_sin": "Latitude sin",
    "longitude_sin": "Longitude sin",
    "longitude_cos": "Longitude cos",
    "local_time_sin": "Local time sin",
    "local_time_cos": "Local time cos",
    "year_progress_sin": "Year progress sin",
    "year_progress_cos": "Year progress cos",
}

CATEGORY_ORDER = [
    "Static/context",
    "Surface/single-level",
    "Geopotential",
    "Temperature",
    "Specific humidity",
    "U wind",
    "V wind",
    "Vertical velocity",
]

CATEGORY_COLORS = {
    "Static/context": "#6b7280",
    "Surface/single-level": "#0f766e",
    "Geopotential": "#7c2d12",
    "Temperature": "#dc2626",
    "Specific humidity": "#2563eb",
    "U wind": "#9333ea",
    "V wind": "#c026d3",
    "Vertical velocity": "#ea580c",
}

#combine latlon and time sin cos into a max value for plotting 
COMBINE_FEATURE_GROUPS = {
    "position_context": {
        "members": [
            "latitude",
            "latitude_sin",
            "longitude_sin",
            "longitude_cos",
        ],
        "label": "Position context",
        "category": "Static/context",
        "combine": "max",
    },
    "local_time_context": {
        "members": [
            "local_time_sin",
            "local_time_cos",
        ],
        "label": "Local time context",
        "category": "Static/context",
        "combine": "max",
    },
    "year_progress_context": {
        "members": [
            "year_progress_sin",
            "year_progress_cos",
        ],
        "label": "Year progress context",
        "category": "Static/context",
        "combine": "max",
    },
}

def lev_to_hpa(lev_idx):
    if lev_idx < 0 or lev_idx >= len(PRESSURE_LEVELS_HPA):
        return None
    return PRESSURE_LEVELS_HPA[lev_idx]

def combine_feature_groups(feature_scores):
    """
    Collapse sin/cos/context feature channels into single plotting rows.

    Keeps the original feature_scores untouched for the actual analysis.
    For plotting, each group score is the max absolute/importance score
    among its member features by default.
    """
    combined = dict(feature_scores)

    for group_name, spec in COMBINE_FEATURE_GROUPS.items():
        members = spec["members"]
        values = [
            float(feature_scores[m])
            for m in members
            if m in feature_scores and np.isfinite(feature_scores[m])
        ]

        if not values:
            continue

        if spec.get("combine", "max") == "mean":
            combined[group_name] = float(np.mean(values))
        elif spec.get("combine", "max") == "sum":
            combined[group_name] = float(np.sum(values))
        else:
            combined[group_name] = float(np.max(values))

        for member in members:
            combined.pop(member, None)

    return combined

def parse_feature_name(name):
    """
    Returns dict with:
      keep, category, label, sort_key
    """
    if name in COMBINE_FEATURE_GROUPS:
        spec = COMBINE_FEATURE_GROUPS[name]
        return {
            "keep": True,
            "category": spec["category"],
            "label": spec["label"],
            "sort_key": (
                CATEGORY_ORDER.index(spec["category"]),
                list(COMBINE_FEATURE_GROUPS).index(name),
            ),
        }
    
    for base, base_label in ATMOSPHERIC_BASES.items():
        prefix = f"{base}_lev"
        if name.startswith(prefix):
            m = re.match(rf"^{re.escape(base)}_lev(\d+)$", name)
            if not m:
                return {"keep": False}

            lev_idx = int(m.group(1))
            hpa = lev_to_hpa(lev_idx)
            if hpa not in KEEP_PRESSURE_LEVELS_HPA:
                return {"keep": False}

            return {
                "keep": True,
                "category": base_label,
                "label": f"{base_label} {hpa} hPa",
                "sort_key": (
                    CATEGORY_ORDER.index(base_label),
                    hpa,
                ),
            }

    if name in SINGLE_LEVEL_LABELS:
        return {
            "keep": True,
            "category": "Surface/single-level",
            "label": SINGLE_LEVEL_LABELS[name],
            "sort_key": (
                CATEGORY_ORDER.index("Surface/single-level"),
                list(SINGLE_LEVEL_LABELS).index(name),
            ),
        }

    if name in STATIC_LABELS:
        return {
            "keep": True,
            "category": "Static/context",
            "label": STATIC_LABELS[name],
            "sort_key": (
                CATEGORY_ORDER.index("Static/context"),
                list(STATIC_LABELS).index(name),
            ),
        }

    return {"keep": False}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def pc_sort_key(pc_name):
    m = re.search(r"(\d+)", pc_name)
    return int(m.group(1)) if m else 10**9


def extract_correlation_scores(data, score_key="mean_abs_r"):
    """
    Returns:
      scores_by_pc: dict[pc_name][feature_name] = score
    """
    scores_by_pc = {}

    for pc_name, pc_data in data.items():
        rows = pc_data.get("ranked_variables", pc_data.get("top_variables", []))
        feature_scores = {}

        for row in rows:
            feature = row.get("variable")
            if feature is None:
                continue
            if score_key not in row:
                continue
            feature_scores[feature] = float(row[score_key])

        scores_by_pc[pc_name] = feature_scores

    return scores_by_pc


def find_regression_coefficients(pc_data):
    """
    Return regression coefficients as either:
      - dict[feature_name] = coefficient
      - array of coefficients plus feature_names elsewhere
    """

    # Best case: your JSON stores a complete feature -> coefficient mapping.
    dict_keys = [
        "coef_standardized",
        "standardized_coefficients_by_feature",
        "coefficients_by_feature",
    ]

    for key in dict_keys:
        if key in pc_data and isinstance(pc_data[key], dict):
            return {
                feature: float(coef)
                for feature, coef in pc_data[key].items()
            }, key

    # Array-style outputs.
    array_keys = [
        "standardized_coefficients",
        "coefficients",
        "coef",
        "coefs",
        "model_coefficients",
    ]

    for key in array_keys:
        if key in pc_data:
            return np.asarray(pc_data[key], dtype=np.float64), key

    # Row-style outputs.
    row_keys = [
        "ranked_features_standardized",
        "ranked_coefficients",
        "top_coefficients",
        "top_features",
        "selected_features",
    ]

    for key in row_keys:
        if key in pc_data and isinstance(pc_data[key], list):
            feature_scores = {}

            for row in pc_data[key]:
                if not isinstance(row, dict):
                    continue

                feature = row.get("feature", row.get("variable"))
                coef = row.get("coefficient", row.get("abs_coefficient"))

                if feature is not None and coef is not None:
                    feature_scores[feature] = float(coef)

            if feature_scores:
                return feature_scores, key

    raise KeyError(
        "Could not find regression coefficients. Expected coef_standardized, "
        "standardized_coefficients/coefficients arrays, or ranked feature rows."
    )


def extract_regression_scores(data, use_abs=True, normalize=True):
    """
    Returns:
      scores_by_pc: dict[pc_name][feature_name] = score
    """
    scores_by_pc = {}

    for pc_name, pc_data in data.items():
        feature_names = pc_data.get("feature_names")
        coef_obj, coef_key = find_regression_coefficients(pc_data)

        if isinstance(coef_obj, dict):
            features = list(coef_obj.keys())
            coef = np.asarray([coef_obj[f] for f in features], dtype=np.float64)

            if use_abs:
                coef = np.abs(coef)

            if normalize:
                denom = np.nanmax(np.abs(coef))
                if denom > 0:
                    coef = coef / denom

            feature_scores = {
                feature: float(value)
                for feature, value in zip(features, coef)
            }

        else:
            if feature_names is None:
                raise KeyError(f"{pc_name}: coefficients array but no feature_names")

            if len(feature_names) != len(coef_obj):
                raise ValueError(
                    f"{pc_name}: feature_names length {len(feature_names)} != "
                    f"{coef_key} length {len(coef_obj)}"
                )

            coef = np.asarray(coef_obj, dtype=np.float64)

            if use_abs:
                coef = np.abs(coef)

            if normalize:
                denom = np.nanmax(np.abs(coef))
                if denom > 0:
                    coef = coef / denom

            feature_scores = {
                feature: float(value)
                for feature, value in zip(feature_names, coef)
            }

        scores_by_pc[pc_name] = feature_scores

    return scores_by_pc


def aggregate_scores(scores_by_pc, pcs=None, aggregation="mean"):
    """
    Aggregate feature importance across PCs.
    """
    if pcs is None:
        pcs = sorted(scores_by_pc.keys(), key=pc_sort_key)

    values = {}

    for pc in pcs:
        for feature, score in scores_by_pc.get(pc, {}).items():
            values.setdefault(feature, []).append(float(score))

    out = {}
    for feature, vals in values.items():
        arr = np.asarray(vals, dtype=np.float64)
        if aggregation == "mean":
            out[feature] = float(np.nanmean(arr))
        elif aggregation == "max":
            out[feature] = float(np.nanmax(arr))
        elif aggregation == "sum":
            out[feature] = float(np.nansum(arr))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    return out


def build_plot_rows(feature_scores):
    feature_scores = combine_feature_groups(feature_scores)
    rows = []

    for feature, score in feature_scores.items():
        parsed = parse_feature_name(feature)
        if not parsed.get("keep", False):
            continue

        rows.append(
            {
                "feature": feature,
                "label": parsed["label"],
                "category": parsed["category"],
                "sort_key": parsed["sort_key"],
                "score": float(score),
            }
        )

    return sorted(rows, key=lambda x: x["sort_key"])


def smooth_line(y, window=5):
    y = np.asarray(y, dtype=np.float64)
    if len(y) < window:
        return y

    pad = window // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def annotate_top_k_bars(ax, scores, y, k=3, x_max=None, color="black"):
    """
    Mark the top-k bars by value using rank labels #1, #2, #3.
    This marks descriptive rank, not statistical significance.
    """
    scores = np.asarray(scores, dtype=np.float64)

    valid = np.isfinite(scores)
    if not valid.any():
        return

    valid_indices = np.where(valid)[0]
    ranked = valid_indices[np.argsort(scores[valid_indices])[::-1]]
    top = ranked[:k]

    if x_max is None:
        x_max = np.nanmax(scores)
        if x_max <= 0:
            x_max = 1.0

    for rank, idx in enumerate(top, start=1):
        ax.text(
            scores[idx] + x_max * 0.015,
            y[idx],
            f"#{rank}",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
            color=color,
        )


def plot_rows(rows, title, score_label, output_path, add_smooth=True):
    if not rows:
        raise ValueError("No rows to plot after filtering variables")

    labels = [r["label"] for r in rows]
    scores = np.asarray([r["score"] for r in rows], dtype=np.float64)
    categories = [r["category"] for r in rows]
    colors = [CATEGORY_COLORS.get(cat, "#9ca3af") for cat in categories]

    y = np.arange(len(rows))

    fig_height = max(8, 0.25 * len(rows))
    fig, ax = plt.subplots(figsize=(13.5, fig_height))

    ax.barh(y, scores, color=colors, alpha=0.85)

    if add_smooth:
        smoothed = smooth_line(scores, window=5)
        ax.plot(
            smoothed,
            y,
            color="black",
            linewidth=1.5,
            alpha=0.75,
            label="smoothed trend",
        )
        ax.legend(loc="lower right")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(score_label)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)

    x_max = np.nanmax(scores)
    if x_max <= 0:
        x_max = 1.0

    ax.set_xlim(0, x_max * 1.32)

    # Category separators and right-side grouped labels.
    for cat in CATEGORY_ORDER:
        idxs = [i for i, c in enumerate(categories) if c == cat]
        if not idxs:
            continue

        start = min(idxs)
        end = max(idxs)
        mid = (start + end) / 2

        color = CATEGORY_COLORS.get(cat, "#9ca3af")

        # Light horizontal band across this category.
        ax.axhspan(
            start - 0.5,
            end + 0.5,
            color=color,
            alpha=0.055,
            linewidth=0,
            zorder=0,
        )

        # Separator lines.
        ax.axhline(start - 0.5, color="black", linewidth=0.6, alpha=0.35)
        ax.axhline(end + 0.5, color="black", linewidth=0.6, alpha=0.35)

        # Right-side category label.
        ax.text(
            x_max * 1.08,
            mid,
            cat,
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    annotate_top_k_bars(ax, scores, y, k=3, x_max=x_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")

def rows_for_pc(scores_by_pc, pc):
    return build_plot_rows(scores_by_pc.get(pc, {}))


def get_selected_pcs(scores_by_pc, pcs=None):
    if pcs is not None:
        return [pc for pc in pcs if pc in scores_by_pc]
    return sorted(scores_by_pc.keys(), key=pc_sort_key)


def plot_per_pc_grid(
    scores_by_pc,
    pcs,
    title,
    score_label,
    output_path,
    ncols=4,
):
    """
    Small-multiple horizontal bar charts, one subplot per PC.
    Uses a shared variable order built from the mean score across selected PCs.
    """
    pcs = get_selected_pcs(scores_by_pc, pcs)
    if not pcs:
        raise ValueError("No PCs available for per-PC plotting")

    # Shared y-axis variable order from mean importance across selected PCs.
    shared_scores = aggregate_scores(scores_by_pc, pcs=pcs, aggregation="mean")
    shared_rows = build_plot_rows(shared_scores)
    if not shared_rows:
        raise ValueError("No rows to plot after filtering variables")

    features = [r["feature"] for r in shared_rows]
    labels = [r["label"] for r in shared_rows]
    categories = [r["category"] for r in shared_rows]
    colors = [CATEGORY_COLORS.get(cat, "#9ca3af") for cat in categories]

    nrows = int(np.ceil(len(pcs) / ncols))
    fig_height = max(7, 0.22 * len(features) * nrows)
    fig_width = 5.2 * ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)

    # Shared x limit.
    all_scores = []
    for pc in pcs:
        pc_scores = combine_feature_groups(scores_by_pc.get(pc, {}))
        all_scores.extend([abs(float(pc_scores.get(f, 0.0))) for f in features])

    x_max = np.nanmax(all_scores) if all_scores else 1.0
    if x_max <= 0:
        x_max = 1.0

    y = np.arange(len(features))

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(pcs):
            ax.axis("off")
            continue

        pc = pcs[ax_idx]
        pc_scores = combine_feature_groups(scores_by_pc.get(pc, {}))
        values = np.asarray([float(pc_scores.get(f, 0.0)) for f in features])

        ax.barh(y, values, color=colors, alpha=0.85)
        annotate_top_k_bars(ax, values, y, k=3, x_max=x_max)
        ax.set_title(pc, fontsize=12)
        ax.set_xlim(0, x_max * 1.08)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()

        if ax_idx % ncols == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=7)
        else:
            ax.tick_params(axis="y", labelleft=False)

        # Light category bands plus category labels.
        for cat in CATEGORY_ORDER:
            idxs = [i for i, c in enumerate(categories) if c == cat]
            if not idxs:
                continue

            start = min(idxs)
            end = max(idxs)
            mid = (start + end) / 2
            color = CATEGORY_COLORS.get(cat, "#9ca3af")

            ax.axhspan(
                start - 0.5,
                end + 0.5,
                color=color,
                alpha=0.045,
                linewidth=0,
            )
            ax.axhline(start - 0.5, color="black", linewidth=0.4, alpha=0.25)

            # Add the group label on the left of each category band.
            # x is in axes coordinates, y is in data coordinates.
            if ax_idx % ncols == 0:
                ax.text(
                -0.71,
                mid,
                cat,
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="center",
                fontsize=13,
                fontweight="bold",
                color=color,
                #rotation=90,
                clip_on=False,
            )

    fig.suptitle(title, fontsize=30, y=0.985)
    fig.supxlabel(score_label, fontsize=20, y=0.015)

    plt.tight_layout(rect=[0.12, 0.02, 1.0, 0.985]) #rect=[left, bottom, right, top]
    #plt.subplots_adjust(hspace=0.1, wspace=0.18)
    plt.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def plot_per_pc_separate(
    scores_by_pc,
    pcs,
    title_prefix,
    score_label,
    out_dir,
    filename_prefix,
    add_smooth=True,
):
    """
    Save one full-size plot per PC.
    """
    pcs = get_selected_pcs(scores_by_pc, pcs)
    if not pcs:
        raise ValueError("No PCs available for per-PC plotting")

    pc_dir = out_dir / f"{filename_prefix}_per_pc"
    pc_dir.mkdir(parents=True, exist_ok=True)

    for pc in pcs:
        rows = rows_for_pc(scores_by_pc, pc)
        if not rows:
            print(f"Skipping {pc}: no filtered rows")
            continue

        safe_pc = pc.replace("/", "_")
        plot_rows(
            rows,
            title=f"{title_prefix}: {pc}",
            score_label=score_label,
            output_path=pc_dir / f"{filename_prefix}_{safe_pc}.png",
            add_smooth=add_smooth,
        )
        write_csv(rows, pc_dir / f"{filename_prefix}_{safe_pc}.csv") 


def write_csv(rows, output_path):
    with open(output_path, "w") as f:
        f.write("feature,label,category,score\n")
        for r in rows:
            f.write(
                f"{r['feature']},{r['label']},{r['category']},{r['score']:.8g}\n"
            )
    print(f"Saved {output_path}")


def parse_pc_list(pc_args):
    if not pc_args:
        return None
    return [pc if pc.startswith("PC_") else f"PC_{pc}" for pc in pc_args]


def main():
    parser = argparse.ArgumentParser(
        description="Plot comparable variable-importance summaries from correlation and regression JSON files."
    )
    parser.add_argument("--correlation-json", type=Path, default=None)
    parser.add_argument("--regression-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pcs", nargs="*", default=None, help="Optional PCs, e.g. --pcs 1 2 3 or PC_1 PC_2")
    parser.add_argument("--aggregation", choices=["mean", "max", "sum"], default="mean")
    parser.add_argument("--no-smooth", action="store_true")
    parser.add_argument("--regression-no-normalize", action="store_true", help="Use raw absolute regression coefficients instead of normalizing each PC by max abs coefficient.")
    parser.add_argument(
        "--per-pc",
        action="store_true",
        help="Also plot variable importance separately for each selected PC.",
    )
    parser.add_argument(
        "--per-pc-layout",
        choices=["grid", "separate", "both"],
        default="grid",
        help="How to save per-PC plots.",
    )
    parser.add_argument(
        "--per-pc-grid-cols",
        type=int,
        default=4,
        help="Number of columns for per-PC grid plots.",
    )    

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pcs = parse_pc_list(args.pcs)

    # correlation block
    if args.correlation_json is None and args.regression_json is None:
        raise ValueError("Provide at least --correlation-json or --regression-json")

    if args.correlation_json is not None:
        data = load_json(args.correlation_json)
        scores_by_pc = extract_correlation_scores(data, score_key="mean_abs_r")
        agg = aggregate_scores(scores_by_pc, pcs=pcs, aggregation=args.aggregation)
        rows = build_plot_rows(agg)

        suffix = "selected_pcs" if pcs else "all_pcs"
        out_png = args.out_dir / f"correlation_variable_importance_{args.aggregation}_{suffix}.png"
        out_csv = args.out_dir / f"correlation_variable_importance_{args.aggregation}_{suffix}.csv"

        plot_rows(
            rows,
            title=f"Correlation variable importance ({args.aggregation} across PCs)",
            score_label="Mean absolute spatial Pearson r",
            output_path=out_png,
            add_smooth=not args.no_smooth,
        )
        write_csv(rows, out_csv)

        if args.per_pc:
            if args.per_pc_layout in ("grid", "both"):
                plot_per_pc_grid(
                    scores_by_pc=scores_by_pc,
                    pcs=pcs,
                    title="Correlation variable importance per PC",
                    score_label="Mean absolute spatial Pearson r",
                    output_path=args.out_dir / f"correlation_variable_importance_per_pc_grid_{suffix}.png",
                    ncols=args.per_pc_grid_cols,
                )

            if args.per_pc_layout in ("separate", "both"):
                plot_per_pc_separate(
                    scores_by_pc=scores_by_pc,
                    pcs=pcs,
                    title_prefix="Correlation variable importance",
                    score_label="Mean absolute spatial Pearson r",
                    out_dir=args.out_dir,
                    filename_prefix="correlation_variable_importance",
                    add_smooth=not args.no_smooth,
                )

    
    #regression block
    if args.regression_json is not None:
        data = load_json(args.regression_json)
        scores_by_pc = extract_regression_scores(
            data,
            use_abs=True,
            normalize=not args.regression_no_normalize,
        )
        agg = aggregate_scores(scores_by_pc, pcs=pcs, aggregation=args.aggregation)
        rows = build_plot_rows(agg)

        suffix = "selected_pcs" if pcs else "all_pcs"
        norm_label = "normalized" if not args.regression_no_normalize else "raw"
        out_png = args.out_dir / f"regression_variable_importance_{norm_label}_{args.aggregation}_{suffix}.png"
        out_csv = args.out_dir / f"regression_variable_importance_{norm_label}_{args.aggregation}_{suffix}.csv"

        plot_rows(
            rows,
            title=f"Regression variable importance ({norm_label}, {args.aggregation} across PCs)",
            score_label="Normalized absolute regression weight" if not args.regression_no_normalize else "Absolute regression weight",
            output_path=out_png,
            add_smooth=not args.no_smooth,
        )
        write_csv(rows, out_csv)

        if args.per_pc:
            if args.per_pc_layout in ("grid", "both"):
                plot_per_pc_grid(
                    scores_by_pc=scores_by_pc,
                    pcs=pcs,
                    title=f"Regression variable importance per PC ({norm_label})",
                    score_label="Normalized absolute regression weight" if not args.regression_no_normalize else "Absolute regression weight",
                    output_path=args.out_dir / f"regression_variable_importance_{norm_label}_per_pc_grid_{suffix}.png",
                    ncols=args.per_pc_grid_cols,
                )

            if args.per_pc_layout in ("separate", "both"):
                plot_per_pc_separate(
                    scores_by_pc=scores_by_pc,
                    pcs=pcs,
                    title_prefix=f"Regression variable importance ({norm_label})",
                    score_label="Normalized absolute regression weight" if not args.regression_no_normalize else "Absolute regression weight",
                    out_dir=args.out_dir,
                    filename_prefix=f"regression_variable_importance_{norm_label}",
                    add_smooth=not args.no_smooth,
                )


if __name__ == "__main__":
    main()


