import os

import numpy as np
import pandas as pd
import xarray as xr


def parse_mask_timestamp(path):
    fname = os.path.basename(path).replace(".nc", "")
    return pd.Timestamp(fname)


def nearest_graphcast_row(mask_time, graphcast_df, max_hours=3):
    diffs = np.abs(graphcast_df["time"] - mask_time)
    idx = int(diffs.argmin())

    if diffs.iloc[idx] > pd.Timedelta(hours=max_hours):
        return None

    return graphcast_df.iloc[idx]


def load_mask_at_nodes(mask_path, lat, lon, node_indices, label_mode="intersection"):
    ds = xr.open_dataset(mask_path)

    label = ds["label"]

    if label_mode == "intersection":
        mask = label.min("annotator")
    elif label_mode == "union":
        mask = label.max("annotator")
    elif label_mode == "soft":
        mask = label.mean("annotator")
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    node_lat = xr.DataArray(lat[node_indices], dims="sample")
    node_lon = xr.DataArray(lon[node_indices], dims="sample")

    mask_nodes = mask.interp(
        latitude=node_lat,
        longitude=node_lon,
        method="nearest",
    ).values

    return mask_nodes.astype(np.float32)