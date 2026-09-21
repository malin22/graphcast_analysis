#!/usr/bin/env python3

import os
import glob

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label as connected_components


# ============================================================
# CONFIG
# ============================================================

TC_DIR = (
    "/share/prj-4d/graphcast_shared/data/"
    "ClimateNetLarge/TC_labels_cleaned"
)

YEAR = 2021

# A ClimateNet mask must be this close to a 6-hour GraphCast
# initialization to count as an initialization mask.
MAX_INIT_DISTANCE_HOURS = 3

# Look for subsequent masks over this period.
N_DAYS = 5

# Ignore tiny disconnected pieces.
# ClimateNet grid is 0.25°, so this mainly removes small artifacts.
MIN_COMPONENT_CELLS = 20

# Number of candidates to print.
TOP_N = 40


# ============================================================
# LOAD MASK FILES
# ============================================================

def load_mask_times(mask_dir):

    rows = []

    for path in glob.glob(os.path.join(mask_dir, "*.nc")):

        try:
            time = pd.Timestamp(
                os.path.basename(path).replace(".nc", "")
            )
        except Exception:
            continue

        rows.append({
            "time": time,
            "file": path,
        })

    if not rows:
        raise RuntimeError(f"No masks found in {mask_dir}")

    return (
        pd.DataFrame(rows)
        .sort_values("time")
        .reset_index(drop=True)
    )


# ============================================================
# ANNOTATOR INTERSECTION
# ============================================================

def load_intersection_mask(path):

    with xr.open_dataset(path) as ds:

        if ds.sizes["annotator"] < 2:
            raise ValueError(
                f"Expected >=2 annotators in {path}"
            )

        a0 = ds["label"].isel(annotator=0).values == 1
        a1 = ds["label"].isel(annotator=1).values == 1

        mask = a0 & a1

        lat = ds["latitude"].values
        lon = ds["longitude"].values

    return mask, lat, lon


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

def find_components(mask, lat, lon, min_cells=20):
    """
    Find connected components in the binary TC mask.

    Uses 8-neighbour connectivity.

    Longitude is periodic, so components touching 0° and 360°
    are merged when appropriate.
    """

    structure = np.ones((3, 3), dtype=int)

    labels, n_labels = connected_components(
        mask,
        structure=structure,
    )

    # --------------------------------------------------------
    # Merge components that connect across longitude boundary
    # --------------------------------------------------------

    parent = np.arange(n_labels + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):

        if a == 0 or b == 0:
            return

        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[rb] = ra

    n_lat = mask.shape[0]

    for i in range(n_lat):

        # Same latitude
        union(labels[i, 0], labels[i, -1])

        # Diagonal neighbours across periodic boundary
        if i > 0:
            union(labels[i, 0], labels[i - 1, -1])
            union(labels[i, -1], labels[i - 1, 0])

        if i < n_lat - 1:
            union(labels[i, 0], labels[i + 1, -1])
            union(labels[i, -1], labels[i + 1, 0])

    # --------------------------------------------------------
    # Group pixels by merged component
    # --------------------------------------------------------

    groups = {}

    for lab in range(1, n_labels + 1):

        root = find(lab)

        ys, xs = np.where(labels == lab)

        if root not in groups:
            groups[root] = [[], []]

        groups[root][0].extend(ys.tolist())
        groups[root][1].extend(xs.tolist())

    # --------------------------------------------------------
    # Component statistics
    # --------------------------------------------------------

    components = []

    for ys, xs in groups.values():

        ys = np.asarray(ys)
        xs = np.asarray(xs)

        n_cells = len(xs)

        if n_cells < min_cells:
            continue

        component_lats = lat[ys]
        component_lons = lon[xs]

        # Latitude centroid
        center_lat = float(np.mean(component_lats))

        # Circular mean for longitude
        angles = np.deg2rad(component_lons)

        center_lon = np.rad2deg(
            np.arctan2(
                np.mean(np.sin(angles)),
                np.mean(np.cos(angles)),
            )
        )

        center_lon = center_lon % 360

        components.append({
            "n_cells": n_cells,
            "lat": center_lat,
            "lon": center_lon,
        })

    # Largest first
    components.sort(
        key=lambda x: x["n_cells"],
        reverse=True,
    )

    return components


# ============================================================
# ANALYSE EACH CLIMATENET MASK ONCE
# ============================================================

def analyse_masks(mask_df):

    results = {}

    print(f"Analysing {len(mask_df)} ClimateNet masks...")

    for i, row in mask_df.iterrows():

        mask, lat, lon = load_intersection_mask(
            row["file"]
        )

        components = find_components(
            mask,
            lat,
            lon,
            min_cells=MIN_COMPONENT_CELLS,
        )

        total_positive = int(mask.sum())

        largest = (
            components[0]["n_cells"]
            if components
            else 0
        )

        # Fraction of all positive cells belonging to largest
        # connected TC region.
        dominance = (
            largest / total_positive
            if total_positive > 0
            else 0.0
        )

        results[row["time"]] = {
            "file": row["file"],
            "n_positive": total_positive,
            "n_components": len(components),
            "largest_component": largest,
            "dominance": dominance,
            "components": components,
        }

        if (i + 1) % 25 == 0:
            print(
                f"  processed {i + 1}/{len(mask_df)}"
            )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    masks = load_mask_times(TC_DIR)

    # Only need masks around the requested year.
    masks = masks[
        (masks["time"] >= pd.Timestamp(f"{YEAR}-01-01"))
        &
        (
            masks["time"]
            <= pd.Timestamp(f"{YEAR + 1}-01-06")
        )
    ].copy()

    analyses = analyse_masks(masks)

    # --------------------------------------------------------
    # Possible GraphCast initialization times
    # --------------------------------------------------------

    centers = pd.date_range(
        f"{YEAR}-01-01 00:00",
        f"{YEAR}-12-31 18:00",
        freq="6h",
    )

    records = []

    for center in centers:

        # ----------------------------------------------------
        # Find nearest ClimateNet mask to initialization
        # ----------------------------------------------------

        diffs = (
            masks["time"] - center
        ).abs()

        idx = diffs.idxmin()

        init_mask_time = masks.loc[idx, "time"]

        init_diff_h = (
            diffs.loc[idx]
            / pd.Timedelta(hours=1)
        )

        if init_diff_h > MAX_INIT_DISTANCE_HOURS:
            continue

        init_info = analyses[init_mask_time]

        # Require actual intersection TC pixels
        if init_info["n_positive"] == 0:
            continue

        # ----------------------------------------------------
        # Subsequent masks
        # ----------------------------------------------------

        end = center + pd.Timedelta(days=N_DAYS)

        future = masks[
            (masks["time"] > init_mask_time)
            &
            (masks["time"] <= end)
        ]

        # Only count masks that actually contain an
        # annotator-intersection TC.
        future_with_tc = [
            t
            for t in future["time"]
            if analyses[t]["n_positive"] > 0
        ]

        # ----------------------------------------------------
        # Describe initialization components
        # ----------------------------------------------------

        components = init_info["components"]

        if components:

            largest = components[0]

            largest_lat = largest["lat"]
            largest_lon = largest["lon"]
            largest_cells = largest["n_cells"]

        else:

            largest_lat = np.nan
            largest_lon = np.nan
            largest_cells = 0

        records.append({

            "center": center,

            "init_mask": init_mask_time,
            "diff_h": init_diff_h,

            "n_components": init_info[
                "n_components"
            ],

            "positive_cells": init_info[
                "n_positive"
            ],

            "largest_cells": largest_cells,

            "dominance": init_info[
                "dominance"
            ],

            "largest_lat": largest_lat,
            "largest_lon": largest_lon,

            "future_masks": len(future),

            "future_masks_with_tc": len(
                future_with_tc
            ),

            "future_times": ", ".join(
                t.strftime("%m-%d %H:%M")
                for t in future_with_tc
            ),
        })

    result = pd.DataFrame(records)

    if result.empty:
        print("No candidates found.")
        return

    # ========================================================
    # RANK
    #
    # Priority:
    #   1. one connected TC at initialization
    #   2. dominant largest component
    #   3. many subsequent TC-containing masks
    #   4. mask close to GraphCast initialization
    # ========================================================

    result["single_tc"] = (
        result["n_components"] == 1
    )

    result = result.sort_values(
        [
            "single_tc",
            "dominance",
            "future_masks_with_tc",
            "diff_h",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    columns = [
        "center",
        "init_mask",
        "diff_h",
        "n_components",
        "positive_cells",
        "largest_cells",
        "dominance",
        "largest_lat",
        "largest_lon",
        "future_masks_with_tc",
        "future_times",
    ]

    pd.set_option(
        "display.max_colwidth",
        200,
    )

    print("\n")
    print("=" * 120)
    print("TOP TC INITIALIZATION CANDIDATES")
    print("=" * 120)

    print(
        result[columns]
        .head(TOP_N)
        .to_string(
            index=False,
            formatters={
                "dominance": "{:.3f}".format,
                "largest_lat": "{:.1f}".format,
                "largest_lon": "{:.1f}".format,
                "diff_h": "{:.1f}".format,
            },
        )
    )

    # Save full ranking
    output = (
        "tc_initialization_candidates_"
        f"{YEAR}.csv"
    )

    result.to_csv(
        output,
        index=False,
    )

    print(f"\nSaved full ranking to: {output}")


if __name__ == "__main__":
    main()