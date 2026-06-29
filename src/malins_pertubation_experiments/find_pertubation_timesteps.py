#!/usr/bin/env python3

import os
import glob
import pandas as pd


WEATHER_FEATURE = "TC"
MASK_DIR = f"/share/prj-4d/graphcast_shared/data/ClimateNetLarge/{WEATHER_FEATURE}_labels_cleaned"

START_DATE = "2021-01-01"
END_DATE = "2021-12-31"

FORECAST_DAYS = 5
MAX_DIFF_HOURS = 3

OUT_CSV = f"good_{WEATHER_FEATURE}_starting_points_5day_masks.csv"


def load_mask_times(mask_dir):
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.nc")))

    rows = []
    for f in mask_files:
        try:
            t = pd.Timestamp(os.path.basename(f).replace(".nc", ""))
        except Exception:
            continue

        rows.append({
            "time": t,
            "file": f,
        })

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    if df.empty:
        raise FileNotFoundError(f"No valid mask files found in {mask_dir}")

    return df


def nearest_mask(mask_df, target_time):
    diffs = (mask_df["time"] - target_time).abs()
    idx = diffs.idxmin()

    return {
        "target_time": target_time,
        "mask_time": mask_df.loc[idx, "time"],
        "mask_file": mask_df.loc[idx, "file"],
        "diff_hours": diffs.loc[idx] / pd.Timedelta(hours=1),
    }


def main():
    mask_df = load_mask_times(MASK_DIR)

    candidate_centers = pd.date_range(
        START_DATE,
        END_DATE,
        freq="6h",
    )

    records = []

    for center_time in candidate_centers:
        check_times = {
            "start": center_time + pd.Timedelta(hours=6),
            #"middle": center_time + pd.Timedelta(days=FORECAST_DAYS / 2) + pd.Timedelta(hours=6),
            "end": center_time + pd.Timedelta(days=FORECAST_DAYS) + pd.Timedelta(hours=6),
        }

        matches = {
            name: nearest_mask(mask_df, t)
            for name, t in check_times.items()
        }

        ok = all(
            m["diff_hours"] <= MAX_DIFF_HOURS
            for m in matches.values()
        )

        if not ok:
            continue

        records.append({
            "center_time": center_time,
            "start_target": matches["start"]["target_time"],
            "start_mask_time": matches["start"]["mask_time"],
            "start_diff_h": matches["start"]["diff_hours"],
            #"middle_target": matches["middle"]["target_time"],
            #"middle_mask_time": matches["middle"]["mask_time"],
            #"middle_diff_h": matches["middle"]["diff_hours"],
            "end_target": matches["end"]["target_time"],
            "end_mask_time": matches["end"]["mask_time"],
            "end_diff_h": matches["end"]["diff_hours"],
            "start_mask_file": matches["start"]["mask_file"],
            #"middle_mask_file": matches["middle"]["mask_file"],
            "end_mask_file": matches["end"]["mask_file"],
        })

    out = pd.DataFrame(records)

    if out.empty:
        print("No suitable center times found.")
        return

    out = out.sort_values("center_time").reset_index(drop=True)
    #out.to_csv(OUT_CSV, index=False)

    print(f"Found {len(out)} suitable center times.")
    #print(f"Saved: {OUT_CSV}")
    print()
    print(out[[
        "center_time",
        "start_mask_time",
        #"middle_mask_time",
        "end_mask_time",
        "start_diff_h",
        #"middle_diff_h",
        "end_diff_h",
    ]].head(30))


if __name__ == "__main__":
    main()