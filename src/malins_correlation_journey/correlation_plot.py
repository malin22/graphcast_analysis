#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


"""
Plot ERA5/context ↔ PC correlations.

Expected input CSV columns:

    feature
    feature_kind
    pc
    correlation
    abs_correlation
    n_valid

These are produced by:

    all_era5_context_pc_correlations_long.csv

Example
-------

python -u plot_era5_pc_correlations.py \
  --correlation-csv \
    plots/malins_experiments/2020_2021_correlation/PCA_matrix_multiply/l6_nodes/all_era5_context_pc_correlations_long.csv \
  --out-dir \
    plots/malins_experiments/2020_2021_correlation/PCA_matrix_multiply/l6_nodes/correlation_plots \
  --pcs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --per-pc \
  --per-pc-layout grid \
  --aggregation mean \
  --score abs

Score modes
-----------

--score abs

    Plot absolute Pearson correlations |r|.

    This is recommended for variable-importance-style figures because
    strong negative and strong positive correlations are both treated as
    important.

--score signed

    Plot signed Pearson correlations r.

    Positive bars extend right and negative bars extend left.
"""


# ============================================================
# VARIABLE DEFINITIONS
# ============================================================
# =====================
# DEFAULT PATHS
# =====================

# =====================
# CONFIG
# =====================

CORRELATION_CSV = Path(
    "plots/malins_experiments/2021_correlation_on_2020_19/"
    "PCA/l6_nodes/"
    "all_era5_context_pc_correlations_long.csv"
)

OUT_DIR = Path(
    "plots/malins_experiments/2021_correlation_on_2020_19/"
    "PCA/l6_nodes/"
    "correlation_plots"
)

PCS = list(range(1, 513))          # e.g. None or ["PC_1", "PC_2", ..., "PC_20"]
AGGREGATION = "mean"
SCORE_MODE = "abs"  # "abs" or "signed" (correlation values)

ADD_SMOOTH = False # smooth curve added

PER_PC = False
PER_VARIABLE = True
PER_PC_LAYOUT = "grid"   # "grid", "separate", "both"
PER_PC_GRID_COLS = 5

PRESSURE_LEVELS_HPA = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]


KEEP_PRESSURE_LEVELS_HPA = {
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
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


# Combine related context channels into one plotting row.
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
    },
    "local_time_context": {
        "members": [
            "local_time_sin",
            "local_time_cos",
        ],
        "label": "Local time context",
        "category": "Static/context",
    },
    "year_progress_context": {
        "members": [
            "year_progress_sin",
            "year_progress_cos",
        ],
        "label": "Year progress context",
        "category": "Static/context",
    },
}


# ============================================================
# INPUT HELPERS
# ============================================================

def normalize_feature_name(name: str) -> str:
    """
    Convert names such as:

        context__latitude
        context__local_time_sin

    to:

        latitude
        local_time_sin
    """
    name = str(name)

    if name.startswith("context__"):
        return name[len("context__"):]

    return name


def normalize_pc_name(value) -> str:
    """
    Convert all of these forms to PC_1:

        1
        "1"
        "PC1"
        "PC_1"
    """
    match = re.search(r"(\d+)", str(value))

    if not match:
        raise ValueError(
            f"Could not parse a PC number from {value!r}"
        )

    return f"PC_{int(match.group(1))}"


def pc_sort_key(pc_name: str) -> int:
    match = re.search(r"(\d+)", str(pc_name))

    if not match:
        return 10**9

    return int(match.group(1))


def parse_pc_list(pc_args):
    if not pc_args:
        return None

    return [
        normalize_pc_name(pc)
        for pc in pc_args
    ]


def load_correlation_scores(
    csv_path: Path,
    score_mode: str,
):
    """
    Load the feature-PC correlation table.

    Input orientation:

        feature | pc | correlation

    Output orientation:

        scores_by_pc[pc][feature] = score

    This reversal is the main change required for the new analysis.
    """
    frame = pd.read_csv(csv_path)

    required_columns = {
        "feature",
        "pc",
        "correlation",
    }

    missing_columns = required_columns - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}\n"
            f"Available columns: {list(frame.columns)}"
        )

    frame["feature"] = frame["feature"].map(
        normalize_feature_name
    )

    frame["pc_name"] = frame["pc"].map(
        normalize_pc_name
    )

    if score_mode == "abs":

        if "abs_correlation" in frame.columns:
            frame["score"] = pd.to_numeric(
                frame["abs_correlation"],
                errors="coerce",
            )

        else:
            frame["score"] = np.abs(
                pd.to_numeric(
                    frame["correlation"],
                    errors="coerce",
                )
            )

    elif score_mode == "signed":

        frame["score"] = pd.to_numeric(
            frame["correlation"],
            errors="coerce",
        )

    else:
        raise ValueError(
            f"Unknown score mode: {score_mode}"
        )

    # Duplicate feature-PC rows should not normally exist.
    # If they do, average them.
    frame = (
        frame.groupby(
            ["pc_name", "feature"],
            as_index=False,
        )["score"]
        .mean()
    )

    scores_by_pc = {}

    for _, row in frame.iterrows():
        pc_name = row["pc_name"]
        feature_name = row["feature"]
        score = float(row["score"])

        scores_by_pc.setdefault(
            pc_name,
            {},
        )[feature_name] = score

    return scores_by_pc


# ============================================================
# FEATURE PARSING
# ============================================================

def lev_to_hpa(lev_idx: int):
    if lev_idx < 0:
        return None

    if lev_idx >= len(PRESSURE_LEVELS_HPA):
        return None

    return PRESSURE_LEVELS_HPA[lev_idx]


def parse_feature_name(name: str):
    """
    Return:

        keep
        category
        label
        sort_key
    """
    if name in COMBINE_FEATURE_GROUPS:

        spec = COMBINE_FEATURE_GROUPS[name]

        return {
            "keep": True,
            "category": spec["category"],
            "label": spec["label"],
            "sort_key": (
                CATEGORY_ORDER.index(
                    spec["category"]
                ),
                list(
                    COMBINE_FEATURE_GROUPS
                ).index(name),
            ),
        }

    for base, base_label in ATMOSPHERIC_BASES.items():

        match = re.fullmatch(
            rf"{re.escape(base)}_lev(\d+)",
            name,
        )

        if not match:
            continue

        lev_idx = int(match.group(1))
        hpa = lev_to_hpa(lev_idx)

        if hpa not in KEEP_PRESSURE_LEVELS_HPA:
            return {
                "keep": False,
            }

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
                CATEGORY_ORDER.index(
                    "Surface/single-level"
                ),
                list(
                    SINGLE_LEVEL_LABELS
                ).index(name),
            ),
        }

    if name in STATIC_LABELS:

        return {
            "keep": True,
            "category": "Static/context",
            "label": STATIC_LABELS[name],
            "sort_key": (
                CATEGORY_ORDER.index(
                    "Static/context"
                ),
                list(
                    STATIC_LABELS
                ).index(name),
            ),
        }

    return {
        "keep": False,
    }


# ============================================================
# SCORE PROCESSING
# ============================================================

def combine_feature_groups(
    feature_scores,
    score_mode,
):
    """
    Combine context channels for plotting.

    Absolute mode:
        Keep the maximum |r| among group members.

    Signed mode:
        Keep the member with the largest absolute correlation,
        while preserving its sign.
    """
    combined = dict(feature_scores)

    for group_name, spec in COMBINE_FEATURE_GROUPS.items():

        values = np.asarray(
            [
                float(feature_scores[member])
                for member in spec["members"]
                if (
                    member in feature_scores
                    and np.isfinite(feature_scores[member])
                )
            ],
            dtype=np.float64,
        )

        if values.size == 0:
            continue

        if score_mode == "signed":

            strongest_index = np.argmax(
                np.abs(values)
            )

            combined[group_name] = float(
                values[strongest_index]
            )

        else:

            combined[group_name] = float(
                np.max(values)
            )

        for member in spec["members"]:
            combined.pop(
                member,
                None,
            )

    return combined


def aggregate_scores(
    scores_by_pc,
    pcs,
    aggregation,
):
    """
    Aggregate ERA5-feature correlations across selected PCs.

    For signed scores, mean or sum can produce cancellation.
    Absolute scores are recommended for importance summaries.
    """
    values_by_feature = {}

    for pc in pcs:

        for feature, score in scores_by_pc.get(
            pc,
            {},
        ).items():

            values_by_feature.setdefault(
                feature,
                [],
            ).append(float(score))

    output = {}

    for feature, values in values_by_feature.items():

        values = np.asarray(
            values,
            dtype=np.float64,
        )

        if aggregation == "mean":

            output[feature] = float(
                np.nanmean(values)
            )

        elif aggregation == "sum":

            output[feature] = float(
                np.nansum(values)
            )

        elif aggregation == "max":

            strongest_index = np.nanargmax(
                np.abs(values)
            )

            output[feature] = float(
                values[strongest_index]
            )

        else:

            raise ValueError(
                f"Unknown aggregation: {aggregation}"
            )

    return output


def build_plot_rows(
    feature_scores,
    score_mode,
):
    feature_scores = combine_feature_groups(
        feature_scores,
        score_mode,
    )

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

    return sorted(
        rows,
        key=lambda row: row["sort_key"],
    )


def get_selected_pcs(
    scores_by_pc,
    requested_pcs,
):
    available_pcs = sorted(
        scores_by_pc,
        key=pc_sort_key,
    )

    if requested_pcs is None:
        return available_pcs

    missing_pcs = [
        pc
        for pc in requested_pcs
        if pc not in scores_by_pc
    ]

    if missing_pcs:
        print(
            "Warning: requested PCs not found and skipped: "
            f"{missing_pcs}"
        )

    return [
        pc
        for pc in requested_pcs
        if pc in scores_by_pc
    ]


# ============================================================
# PLOTTING HELPERS
# ============================================================

def smooth_line(
    values,
    window=5,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if len(values) < window:
        return values

    padding = window // 2

    padded = np.pad(
        values,
        (padding, padding),
        mode="edge",
    )

    kernel = (
        np.ones(
            window,
            dtype=np.float64,
        )
        / window
    )

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )


def annotate_top_k_bars(
    ax,
    scores,
    y,
    score_mode,
    axis_extent,
    k=3,
):
    """
    Mark the strongest correlations by absolute magnitude.

    These ranks do not indicate statistical significance.
    """
    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    valid_indices = np.flatnonzero(
        np.isfinite(scores)
    )

    if valid_indices.size == 0:
        return

    ranked_indices = valid_indices[
        np.argsort(
            np.abs(scores[valid_indices])
        )[::-1]
    ]

    for rank, index in enumerate(
        ranked_indices[:k],
        start=1,
    ):

        value = scores[index]
        offset = axis_extent * 0.015

        if (
            score_mode == "signed"
            and value < 0
        ):
            text_x = value - offset
            horizontal_alignment = "right"

        else:
            text_x = value + offset
            horizontal_alignment = "left"

        ax.text(
            text_x,
            y[index],
            f"#{rank}",
            va="center",
            ha=horizontal_alignment,
            fontsize=8,
            fontweight="bold",
            color="black",
        )


def add_category_bands(
    ax,
    categories,
    show_labels,
    label_x,
):
    for category in CATEGORY_ORDER:

        indices = [
            index
            for index, value in enumerate(categories)
            if value == category
        ]

        if not indices:
            continue

        start = min(indices)
        end = max(indices)
        middle = (start + end) / 2

        color = CATEGORY_COLORS.get(
            category,
            "#9ca3af",
        )

        ax.axhspan(
            start - 0.5,
            end + 0.5,
            color=color,
            alpha=0.055,
            linewidth=0,
            zorder=0,
        )

        ax.axhline(
            start - 0.5,
            color="black",
            linewidth=0.5,
            alpha=0.3,
        )

        ax.axhline(
            end + 0.5,
            color="black",
            linewidth=0.5,
            alpha=0.3,
        )

        if show_labels:

            ax.text(
                label_x,
                middle,
                category,
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="center",
                fontsize=10,
                fontweight="bold",
                color=color,
                clip_on=False,
            )


def apply_x_limits(
    ax,
    scores,
    score_mode,
    padding=1.15,
):
    finite_scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    finite_scores = finite_scores[
        np.isfinite(finite_scores)
    ]

    if finite_scores.size == 0:
        extent = 1.0

    else:
        extent = float(
            np.max(
                np.abs(finite_scores)
            )
        )

    if extent <= 0:
        extent = 1.0

    limit = min(
        1.0,
        extent * padding,
    )

    if score_mode == "signed":

        ax.set_xlim(
            -limit,
            limit,
        )

        ax.axvline(
            0.0,
            color="black",
            linewidth=0.8,
            alpha=0.6,
        )

    else:

        ax.set_xlim(
            0.0,
            limit,
        )

    return extent


# ============================================================
# FULL-SIZE PLOT
# ============================================================

def plot_rows(
    rows,
    title,
    score_label,
    output_path,
    score_mode,
    add_smooth=True,
):
    if not rows:
        raise ValueError(
            "No recognized variables remained after filtering"
        )

    labels = [
        row["label"]
        for row in rows
    ]

    scores = np.asarray(
        [
            row["score"]
            for row in rows
        ],
        dtype=np.float64,
    )

    categories = [
        row["category"]
        for row in rows
    ]

    colors = [
        CATEGORY_COLORS.get(
            category,
            "#9ca3af",
        )
        for category in categories
    ]

    y = np.arange(
        len(rows)
    )

    figure_height = max(
        8,
        0.25 * len(rows),
    )

    fig, ax = plt.subplots(
        figsize=(
            13.5,
            figure_height,
        )
    )

    ax.barh(
        y,
        scores,
        color=colors,
        alpha=0.85,
    )

    if add_smooth:

        ax.plot(
            smooth_line(scores),
            y,
            color="black",
            linewidth=1.5,
            alpha=0.75,
            label="Smoothed trend",
        )

        ax.legend(
            loc="lower right"
        )

    ax.set_yticks(y)

    ax.set_yticklabels(
        labels,
        fontsize=8,
    )

    ax.set_xlabel(
        score_label
    )

    ax.set_title(
        title
    )

    ax.invert_yaxis()

    ax.grid(
        axis="x",
        alpha=0.25,
    )

    extent = apply_x_limits(
        ax,
        scores,
        score_mode,
        padding=1.28,
    )

    add_category_bands(
        ax,
        categories,
        show_labels=True,
        label_x=1.11,
    )

    annotate_top_k_bars(
        ax,
        scores,
        y,
        score_mode,
        extent,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved {output_path}"
    )


# ============================================================
# PER-PC GRID
# ============================================================

def plot_per_pc_grid(
    scores_by_pc,
    pcs,
    title,
    score_label,
    output_path,
    score_mode,
    ncols=4,
):
    if not pcs:
        raise ValueError(
            "No PCs available for per-PC plotting"
        )

    # Use mean absolute correlation to establish one shared
    # variable order across all PC panels.
    ordering_scores = {
        pc: {
            feature: abs(value)
            for feature, value in scores.items()
        }
        for pc, scores in scores_by_pc.items()
    }

    shared_scores = aggregate_scores(
        ordering_scores,
        pcs,
        aggregation="mean",
    )

    shared_rows = build_plot_rows(
        shared_scores,
        score_mode="abs",
    )

    if not shared_rows:
        raise ValueError(
            "No recognized features available for plotting"
        )

    features = [
        row["feature"]
        for row in shared_rows
    ]

    labels = [
        row["label"]
        for row in shared_rows
    ]

    categories = [
        row["category"]
        for row in shared_rows
    ]

    colors = [
        CATEGORY_COLORS.get(
            category,
            "#9ca3af",
        )
        for category in categories
    ]

    combined_by_pc = {
        pc: combine_feature_groups(
            scores_by_pc[pc],
            score_mode,
        )
        for pc in pcs
    }

    all_values = np.asarray(
        [
            combined_by_pc[pc].get(
                feature,
                0.0,
            )
            for pc in pcs
            for feature in features
        ],
        dtype=np.float64,
    )

    extent = float(
        np.max(
            np.abs(all_values)
        )
    )

    if extent <= 0:
        extent = 1.0

    limit = min(
        1.0,
        extent * 1.12,
    )

    nrows = int(
        np.ceil(
            len(pcs) / ncols
        )
    )

    figure_height = max(
        7,
        0.22 * len(features) * nrows,
    )

    figure_width = 5.2 * ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            figure_width,
            figure_height,
        ),
        sharey=True,
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    y = np.arange(
        len(features)
    )

    for axis_index, ax in enumerate(axes):

        if axis_index >= len(pcs):

            ax.axis("off")
            continue

        pc = pcs[axis_index]

        values = np.asarray(
            [
                combined_by_pc[pc].get(
                    feature,
                    0.0,
                )
                for feature in features
            ],
            dtype=np.float64,
        )

        ax.barh(
            y,
            values,
            color=colors,
            alpha=0.85,
        )

        annotate_top_k_bars(
            ax,
            values,
            y,
            score_mode,
            extent,
        )

        ax.set_title(
            pc.replace("_", " "),
            fontsize=12,
        )

        ax.grid(
            axis="x",
            alpha=0.25,
        )

        ax.invert_yaxis()

        if score_mode == "signed":

            ax.set_xlim(
                -limit,
                limit,
            )

            ax.axvline(
                0.0,
                color="black",
                linewidth=0.7,
                alpha=0.6,
            )

        else:

            ax.set_xlim(
                0.0,
                limit,
            )

        if axis_index % ncols == 0:

            ax.set_yticks(y)

            ax.set_yticklabels(
                labels,
                fontsize=7,
            )

        else:

            ax.tick_params(
                axis="y",
                labelleft=False,
            )

        add_category_bands(
            ax,
            categories,
            show_labels=(
                axis_index % ncols == 0
            ),
            label_x=-0.71,
        )

    fig.suptitle(
        title,
        fontsize=30,
        y=0.985,
    )

    fig.supxlabel(
        score_label,
        fontsize=20,
        y=0.015,
    )

    plt.tight_layout(
        rect=[
            0.12,
            0.02,
            1.0,
            0.985,
        ]
    )

    plt.savefig(
        output_path,
        dpi=450,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved {output_path}"
    )


# ============================================================
# OUTPUT HELPERS
# ============================================================

def write_csv(
    rows,
    output_path,
):
    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "feature",
                "label",
                "category",
                "score",
            ],
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    "feature": row["feature"],
                    "label": row["label"],
                    "category": row["category"],
                    "score": f"{row['score']:.8g}",
                }
            )

    print(
        f"Saved {output_path}"
    )


def plot_per_pc_separate(
    scores_by_pc,
    pcs,
    output_dir,
    score_label,
    score_mode,
    add_smooth,
):
    pc_output_dir = (
        output_dir
        / f"era5_pc_correlation_{score_mode}_per_pc"
    )

    pc_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for pc in pcs:

        rows = build_plot_rows(
            scores_by_pc[pc],
            score_mode,
        )

        if not rows:
            print(
                f"Skipping {pc}: no recognized variables"
            )
            continue

        plot_rows(
            rows=rows,
            title=(
                "ERA5/context correlation: "
                f"{pc.replace('_', ' ')}"
            ),
            score_label=score_label,
            output_path=(
                pc_output_dir
                / f"{pc}.png"
            ),
            score_mode=score_mode,
            add_smooth=add_smooth,
        )

        write_csv(
            rows,
            pc_output_dir / f"{pc}.csv",
        )



def plot_feature_group_grid(
    scores_by_pc,
    pcs,
    feature_base,
    feature_label,
    output_dir,
    score_mode,
    ncols=4,
):
    """
    One figure for one atmospheric feature group.

    Example:
        Temperature

    Each subplot corresponds to one pressure level.
    Within each subplot:
        x-axis = PCs
        y-axis = correlation
    """

    # ----------------------------------------------------------
    # Find all pressure-level features belonging to this group
    # ----------------------------------------------------------

    feature_levels = []

    for feature in {
        feature
        for pc in pcs
        for feature in scores_by_pc.get(pc, {})
    }:

        match = re.fullmatch(
            rf"{re.escape(feature_base)}_lev(\d+)",
            feature,
        )

        if not match:
            continue

        lev_idx = int(match.group(1))
        hpa = lev_to_hpa(lev_idx)

        if hpa not in KEEP_PRESSURE_LEVELS_HPA:
            continue

        feature_levels.append(
            (hpa, feature)
        )

    if not feature_levels:
        print(f"No features found for {feature_label}")
        return

    # Sort from upper atmosphere to surface
    feature_levels.sort(
        key=lambda item: item[0]
    )

    # ----------------------------------------------------------
    # Layout
    # ----------------------------------------------------------

    n_features = len(feature_levels)

    nrows = int(
        np.ceil(n_features / ncols)
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            4.5 * ncols,
            3.5 * nrows,
        ),
        sharex=True,
        sharey=True,
    )

    axes = np.asarray(
        axes
    ).reshape(-1)

    pc_numbers = np.asarray(
        [pc_sort_key(pc) for pc in pcs]
    )

    # ----------------------------------------------------------
    # Determine common y-axis
    # ----------------------------------------------------------

    all_values = []

    for hpa, feature in feature_levels:

        for pc in pcs:

            value = scores_by_pc.get(
                pc, {}
            ).get(
                feature, np.nan
            )

            if np.isfinite(value):
                all_values.append(value)

    all_values = np.asarray(
        all_values,
        dtype=np.float64,
    )

    if all_values.size == 0:
        print(f"No valid values for {feature_label}")
        plt.close(fig)
        return

    extent = np.nanmax(
        np.abs(all_values)
    )

    limit = min(
        1.0,
        extent * 1.15,
    )

    # ----------------------------------------------------------
    # Plot pressure levels
    # ----------------------------------------------------------

    for axis_index, ax in enumerate(axes):

        if axis_index >= n_features:
            ax.axis("off")
            continue

        hpa, feature = feature_levels[axis_index]

        values = np.asarray(
            [
                scores_by_pc.get(
                    pc, {}
                ).get(
                    feature,
                    np.nan,
                )
                for pc in pcs
            ],
            dtype=np.float64,
        )

        ax.bar(
            pc_numbers,
            values,
            alpha=0.85,
        )

        ax.set_title(
            f"{hpa} hPa",
            fontsize=11,
            fontweight="bold",
        )

        ax.grid(
            axis="y",
            alpha=0.25,
        )

        # Same scale for every pressure level
        if score_mode == "abs":

            ax.set_ylim(
                0.0,
                limit,
            )

        else:

            ax.set_ylim(
                -limit,
                limit,
            )

            ax.axhline(
                0.0,
                color="black",
                linewidth=0.7,
                alpha=0.6,
            )

        # Don't show every PC label if there are many PCs
        ax.set_xticks(
            pc_numbers
        )

        ax.set_xticklabels(
            [str(pc) for pc in pc_numbers],
            rotation=90,
            fontsize=7,
        )

    # ----------------------------------------------------------
    # Labels
    # ----------------------------------------------------------

    fig.suptitle(
        f"{feature_label}: correlation across PCs",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )

    fig.supxlabel(
        "Principal component",
        fontsize=13,
    )

    if score_mode == "abs":

        fig.supylabel(
            "Absolute pooled space-time Pearson correlation |r|",
            fontsize=13,
        )

    else:

        fig.supylabel(
            "Pooled space-time Pearson correlation r",
            fontsize=13,
        )

    plt.tight_layout(
        rect=[
            0.03,
            0.03,
            1.0,
            0.97,
        ]
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    output_path = (
        output_dir
        / (
            f"{feature_base}_"
            f"correlation_across_pcs_grid.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved {output_path}")

# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    requested_pcs = parse_pc_list(PCS)

    scores_by_pc = load_correlation_scores(
        CORRELATION_CSV,
        score_mode=SCORE_MODE,
    )

    selected_pcs = get_selected_pcs(
        scores_by_pc,
        requested_pcs,
    )

    if not selected_pcs:
        raise ValueError("None of the requested PCs were found")

    print(f"Loaded {len(scores_by_pc)} PCs")
    print(f"Plotting {len(selected_pcs)} PCs")
    print(f"Score mode: {SCORE_MODE}")

    aggregate = aggregate_scores(
        scores_by_pc,
        selected_pcs,
        aggregation=AGGREGATION,
    )

    aggregate_rows = build_plot_rows(
        aggregate,
        score_mode=SCORE_MODE,
    )

    selection_suffix = (
        "selected_pcs"
        if requested_pcs
        else "all_pcs"
    )

    if SCORE_MODE == "abs":
        score_label = "Absolute pooled space-time Pearson correlation |r|"
        aggregate_title = (
            f"ERA5/context correlation strength "
            f"({AGGREGATION} across PCs)"
        )
    else:
        score_label = "Pooled space-time Pearson correlation r"
        aggregate_title = (
            f"Signed ERA5/context correlation "
            f"({AGGREGATION} across PCs)"
        )

    aggregate_prefix = (
        f"era5_pc_correlation_"
        f"{SCORE_MODE}_"
        f"{AGGREGATION}_"
        f"{selection_suffix}"
    )

    plot_rows(
        rows=aggregate_rows,
        title=aggregate_title,
        score_label=score_label,
        output_path=OUT_DIR / f"{aggregate_prefix}.png",
        score_mode=SCORE_MODE,
        add_smooth=ADD_SMOOTH,
    )

    write_csv(
        aggregate_rows,
        OUT_DIR / f"{aggregate_prefix}.csv",
    )

    if PER_PC:

        if PER_PC_LAYOUT in ("grid", "both"):

            plot_per_pc_grid(
                scores_by_pc=scores_by_pc,
                pcs=selected_pcs,
                title="ERA5/context correlations per PC",
                score_label=score_label,
                output_path=(
                    OUT_DIR
                    / f"era5_pc_correlation_{SCORE_MODE}_per_pc_grid_{selection_suffix}.png"
                ),
                score_mode=SCORE_MODE,
                ncols=PER_PC_GRID_COLS,
            )

        if PER_PC_LAYOUT in ("separate", "both"):

            plot_per_pc_separate(
                scores_by_pc=scores_by_pc,
                pcs=selected_pcs,
                output_dir=OUT_DIR,
                score_label=score_label,
                score_mode=SCORE_MODE,
                add_smooth=ADD_SMOOTH,
            )


    if PER_VARIABLE:
        for feature_base, feature_label in ATMOSPHERIC_BASES.items():

            out_dir = OUT_DIR / f"grouped_by_feature"
            os.makedirs(out_dir, exist_ok=True)

            plot_feature_group_grid(
                scores_by_pc=scores_by_pc,
                pcs=selected_pcs,
                feature_base=feature_base,
                feature_label=feature_label,
                output_dir=out_dir,
                score_mode=SCORE_MODE,
                ncols=4,
            )


if __name__ == "__main__":
    main()