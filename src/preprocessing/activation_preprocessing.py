import os
import re
from glob import glob

import numpy as np
import pandas as pd


def parse_activation_timestamp(path):
    fname = os.path.basename(path)
    m = re.search(r"t(\d{4})-(\d{2})-(\d{2})T(\d{2})", fname)
    if not m:
        raise ValueError(f"Could not parse timestamp from {fname}")
    y, mo, d, h = map(int, m.groups())
    return pd.Timestamp(y, mo, d, h)


def load_timestamps(files_txt):
    with open(files_txt, "r") as f:
        files = [line.strip() for line in f if line.strip()]
    timestamps = pd.to_datetime([parse_activation_timestamp(p) for p in files])
    return files, timestamps

def load_activations(path):
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    if x.ndim != 2:
        raise ValueError(f"Expected [nodes, features], got shape {x.shape}")

    return x.astype(np.float32)


def load_raw_activation_years(acts_dirs):
    """Collect raw activation files and timestamps without loading all activations into RAM."""
    all_files = []

    for year, acts_dir in sorted(acts_dirs.items()):
        files = sorted(glob(os.path.join(acts_dir, "*.npy")))
        print(f"Found {len(files)} raw activation files for {year} in {acts_dir}")
        all_files.extend(files)

    valid_files = []
    for f in all_files:
        X_t = load_activations(f)
        if np.isnan(X_t).any():
            print(f"Skipping NaN activation file: {os.path.basename(f)}")
            continue
        valid_files.append(f)

    act_files = sorted(valid_files, key=parse_activation_timestamp)
    graphcast_times = pd.to_datetime([parse_activation_timestamp(p) for p in act_files])

    if len(act_files) == 0:
        raise ValueError("No valid raw activation files found.")

    example = load_activations(act_files[0])
    max_features = example.shape[1]

    print(f"Using {len(act_files)} valid raw activation files.")
    print("Raw activation feature dimension:", max_features)

    return act_files, graphcast_times, max_features


def load_pca_metadata(pc_score_paths, timestep_files_txts):
    pc_scores_by_year = {}
    timestamps_by_year = {}
    max_features = None

    for year in sorted(pc_score_paths):
        pc_scores = np.load(pc_score_paths[year], mmap_mode="r")
        _, timestamps = load_timestamps(timestep_files_txts[year])

        T, N, K = pc_scores.shape
        if len(timestamps) != T:
            raise ValueError(f"{len(timestamps)} timestamps but {T} PC-score timesteps for {year}")

        pc_scores_by_year[year] = pc_scores
        timestamps_by_year[year] = pd.to_datetime(timestamps)

        max_features = K if max_features is None else min(max_features, K)

        print(f"PC scores {year}:", pc_scores.shape)

    return pc_scores_by_year, timestamps_by_year, max_features



def build_graphcast_time_table(timestamps_by_year):
    rows = []

    for year, times in timestamps_by_year.items():
        for t_idx, t in enumerate(times):
            rows.append({
                "year": year,
                "t_idx": t_idx,
                "time": t,
            })

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df