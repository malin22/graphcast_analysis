#!/usr/bin/env python3

import os
import glob
import pandas as pd


WEATHER_FEATURE = "TC"
MASK_DIR = (
    f"/share/prj-4d/graphcast_shared/data/"
    f"ClimateNetLarge/{WEATHER_FEATURE}_labels_cleaned"
)

START_DATE = "2021-01-01"
END_DATE = "2021-12-31"

MAX_DIFF_HOURS = 3

#OUT_CSV = f"good_{WEATHER_FEATURE}_starting_points_5day_masks.csv"


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

    df = pd.DataFrame(rows)

    if df.empty:
        raise FileNotFoundError(
            f"No valid mask files found in {mask_dir}"
        )

    return df.sort_values("time").reset_index(drop=True)


def nearest_mask(mask_df, target_time):
    """Find the mask closest in time to target_time."""

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

    print(f"Loaded {len(mask_df)} masks from {MASK_DIR}")

    # Possible starting/center times
    candidate_centers = pd.date_range(
        START_DATE,
        END_DATE,
        freq="6h",
    )

    records = []

    for center_time in candidate_centers:

        # Times for which we want a mask
        check_times = {
            "6h": center_time + pd.Timedelta(hours=6),
            "1d": center_time + pd.Timedelta(days=1),
            "3d": center_time + pd.Timedelta(days=3),
            "5d": center_time + pd.Timedelta(days=5),
        }

        # Find closest available mask to each desired time
        matches = {
            name: nearest_mask(mask_df, target_time)
            for name, target_time in check_times.items()
        }

        # Require all four masks to be sufficiently close
        ok = all(
            match["diff_hours"] <= MAX_DIFF_HOURS
            for match in matches.values()
        )

        if not ok:
            continue

        records.append({
            "center_time": center_time,

            "target_6h": matches["6h"]["target_time"],
            "mask_6h": matches["6h"]["mask_time"],
            "diff_6h": matches["6h"]["diff_hours"],
            "file_6h": matches["6h"]["mask_file"],

            "target_1d": matches["1d"]["target_time"],
            "mask_1d": matches["1d"]["mask_time"],
            "diff_1d": matches["1d"]["diff_hours"],
            "file_1d": matches["1d"]["mask_file"],

            "target_3d": matches["3d"]["target_time"],
            "mask_3d": matches["3d"]["mask_time"],
            "diff_3d": matches["3d"]["diff_hours"],
            "file_3d": matches["3d"]["mask_file"],

            "target_5d": matches["5d"]["target_time"],
            "mask_5d": matches["5d"]["mask_time"],
            "diff_5d": matches["5d"]["diff_hours"],
            "file_5d": matches["5d"]["mask_file"],
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

    print(
        out[[
            "center_time",
            "mask_6h",
            "diff_6h",
            "mask_1d",
            "diff_1d",
            "mask_3d",
            "diff_3d",
            "mask_5d",
            "diff_5d",
        ]].head(30)
    )


if __name__ == "__main__":
    main()