#!/usr/bin/env python3
"""
Find candidate start times for "keeping an AR alive" experiments.

This script does NOT require GraphCast forecasts.
It scans ClimateNet AR masks and finds times where an AR is present at t0
but weakens/disappears after 24h / 48h / 72h.
"""

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr


# =====================
# CONFIG
# =====================

MASK_DIR = "/share/prj-4d/graphcast_shared/data/ClimateNetLarge/AR_labels_cleaned"

OUT_DIR = "plots/malins_experiments/pertubation_experiments/ar_candidate_search"
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = [2021]

DECAY_LEAD_HOURS = [24, 48, 72]

MIN_INITIAL_AREA_FRACTION = 0.002

USE_ANNOTATOR_INTERSECTION = True


# =====================
# HELPERS
# =====================

def parse_time_from_mask(path):
    name = os.path.basename(path).replace(".nc", "")
    return pd.Timestamp(name)


def discover_masks():
    rows = []

    for path in sorted(glob.glob(os.path.join(MASK_DIR, "*.nc"))):
        try:
            t = parse_time_from_mask(path)
        except Exception:
            continue

        if YEARS is not None and t.year not in YEARS:
            continue

        rows.append({"time": t, "file": path})

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    if df.empty:
        raise FileNotFoundError("No matching mask files found.")

    print(f"Found {len(df)} mask files.")
    print("Time range:", df["time"].min(), "to", df["time"].max())

    return df


def load_mask_area(path):
    ds = xr.open_dataset(path)
    label = ds["label"]

    if USE_ANNOTATOR_INTERSECTION:
        mask = label.min("annotator") > 0
    else:
        mask = label.max("annotator") > 0

    # area weighting if latitude exists
    if "latitude" in mask.coords:
        weights = np.cos(np.deg2rad(mask["latitude"])).broadcast_like(mask)
        area = weights.where(mask).sum(skipna=True) / weights.sum(skipna=True)
        return float(area.values)

    return float(mask.mean().values)


def find_nearest_mask(mask_df, target_time, max_diff_hours=3):
    diffs = abs(mask_df["time"] - target_time)
    idx = diffs.idxmin()

    if diffs.loc[idx] > pd.Timedelta(hours=max_diff_hours):
        return None

    return mask_df.loc[idx]


# =====================
# MAIN LOGIC
# =====================

def main():
    mask_df = discover_masks()

    records = []

    for _, row in mask_df.iterrows():
        t0 = row["time"]
        initial_file = row["file"]

        try:
            initial_area = load_mask_area(initial_file)
        except Exception as e:
            print(f"[SKIP] {t0}: could not load initial mask: {e}")
            continue

        if initial_area < MIN_INITIAL_AREA_FRACTION:
            continue

        for lead_h in DECAY_LEAD_HOURS:
            future_time = t0 + pd.Timedelta(hours=lead_h)
            future_row = find_nearest_mask(mask_df, future_time)

            if future_row is None:
                continue

            try:
                future_area = load_mask_area(future_row["file"])
            except Exception as e:
                print(f"[SKIP] {t0} lead {lead_h}: could not load future mask: {e}")
                continue

            area_drop = initial_area - future_area
            relative_drop = area_drop / initial_area if initial_area > 0 else np.nan

            score = initial_area * relative_drop

            records.append({
                "center_time": str(t0),
                "initial_mask_file": initial_file,
                "lead_hours": lead_h,
                "future_time": str(future_row["time"]),
                "future_mask_file": future_row["file"],
                "initial_ar_area_fraction": initial_area,
                "future_ar_area_fraction": future_area,
                "area_drop": area_drop,
                "relative_drop": relative_drop,
                "candidate_score": score,
            })

    ranking = pd.DataFrame(records)

    if ranking.empty:
        raise RuntimeError("No candidates found. Try lowering MIN_INITIAL_AREA_FRACTION.")

    ranking = ranking.sort_values(
        ["candidate_score", "area_drop", "initial_ar_area_fraction"],
        ascending=[False, False, False],
    )

    out_csv = os.path.join(OUT_DIR, "ar_mask_decay_candidate_ranking.csv")
    ranking.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)
    print("\nTop candidates:")
    print(
        ranking[
            [
                "center_time",
                "lead_hours",
                "initial_ar_area_fraction",
                "future_ar_area_fraction",
                "area_drop",
                "relative_drop",
                "candidate_score",
            ]
        ].head(30)
    )


if __name__ == "__main__":
    main()