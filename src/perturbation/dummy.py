#!/usr/bin/env python3

import os
import glob
import math

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

TC_DIR = (
    "/share/prj-4d/graphcast_shared/data/"
    "ClimateNetLarge/TC_labels_cleaned"
)

START_TIME = pd.Timestamp("2021-05-24 18:00:00")
N_DAYS = 3


# ============================================================
# LOAD FILES
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

    return (
        pd.DataFrame(rows)
        .sort_values("time")
        .reset_index(drop=True)
    )


masks = load_mask_times(TC_DIR)

end_time = START_TIME + pd.Timedelta(days=N_DAYS)

selected = masks[
    (masks["time"] >= START_TIME)
    & (masks["time"] <= end_time)
].copy()


print(f"Found {len(selected)} masks:")
print(selected[["time", "file"]].to_string(index=False))


# ============================================================
# PLOT EACH ANNOTATOR
# ============================================================

for annotator in [0, 1]:

    n = len(selected)

    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, 3.8 * nrows),
        squeeze=False,
    )

    axes = axes.flatten()

    for ax, (_, row) in zip(axes, selected.iterrows()):

        with xr.open_dataset(row["file"]) as ds:

            label = ds["label"].sel(
                annotator=annotator
            )

            lon = ds["longitude"].values
            lat = ds["latitude"].values

            mask = label.values

        ax.pcolormesh(
            lon,
            lat,
            mask,
            shading="auto",
        )

        n_positive = int(np.sum(mask == 1))

        hours = (
            row["time"] - START_TIME
        ) / pd.Timedelta(hours=1)

        ax.set_title(
            f"{row['time']}\n"
            f"+{hours:.0f} h | "
            f"{n_positive} positive cells"
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)

    # Remove unused panels
    for ax in axes[n:]:
        ax.remove()

    fig.suptitle(
        f"ClimateNet TC masks — Annotator {annotator}",
        fontsize=14,
    )

    fig.tight_layout()

    output_path = f"tc_masks_annotator_{annotator}.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")