#!/usr/bin/env python3
import argparse
import os
import re
from glob import glob
from pathlib import Path

import numpy as np


DEFAULT_PATTERN = "layer0008_mesh_gnn_post_res_nodes_mesh_nodes_t*.npy"


def load_activation_shape_and_nans(path):
    x = np.load(path, mmap_mode="r")

    if x.dtype == np.dtype("|V2"):
        x = x.view(np.float16)

    x = np.asarray(x)

    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0, :]

    if x.ndim != 2:
        return x.shape, None

    total_nans = int(np.isnan(x).sum())
    return x.shape, total_nans


def parse_time_from_name(path):
    name = os.path.basename(path)
    match = re.search(r"_t(\d{4}-\d{2}-\d{2}T\d{2})", name)
    if not match:
        return None
    return np.datetime64(match.group(1), "h")


def expected_times_for_year(year):
    start = np.datetime64(f"{year}-01-01T00", "h")
    end = np.datetime64(f"{year + 1}-01-01T00", "h")
    return np.arange(start, end, np.timedelta64(6, "h"))


def main():
    parser = argparse.ArgumentParser(
        description="Find NaNs, shape issues, and missing 6-hourly activation files."
    )
    parser.add_argument(
        "acts_dir",
        nargs="?",
        default="/share/prj-4d/graphcast_shared/data/graphcast_activation_2020",
        help="Activation directory",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
        help="Year to check",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern to check",
    )
    parser.add_argument(
        "--all-npy",
        action="store_true",
        help="Check all .npy files instead of layer0008 mesh-node files only",
    )

    args = parser.parse_args()

    acts_dir = Path(args.acts_dir)
    pattern = "*.npy" if args.all_npy else args.pattern

    files = sorted(glob(str(acts_dir / pattern)))

    print(f"Directory: {acts_dir}")
    print(f"Pattern: {pattern}")
    print(f"Found files: {len(files)}")

    if not files:
        return

    seen_times = {}
    shape_counts = {}
    nan_files = []
    unexpected_shape_files = []
    unparsable_files = []

    expected_shape = None

    for f in files:
        t = parse_time_from_name(f)
        if t is None:
            unparsable_files.append(f)
        else:
            seen_times.setdefault(t, []).append(f)

        shape, total_nans = load_activation_shape_and_nans(f)
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

        if expected_shape is None and total_nans is not None:
            expected_shape = shape

        if total_nans is None:
            unexpected_shape_files.append((f, shape))
            print(f"{os.path.basename(f)}: unexpected shape {shape}; skipping NaN count")
            continue

        if expected_shape is not None and shape != expected_shape:
            unexpected_shape_files.append((f, shape))
            print(
                f"{os.path.basename(f)}: shape mismatch {shape}; "
                f"expected {expected_shape}"
            )

        if total_nans:
            nan_files.append((f, total_nans, shape))
            print(f"{os.path.basename(f)}: NaNs={total_nans}, shape={shape}")

    print("\nShape summary:")
    for shape, count in sorted(shape_counts.items(), key=lambda kv: str(kv[0])):
        print(f"  {shape}: {count}")

    if unparsable_files:
        print(f"\nFiles with no parseable timestamp: {len(unparsable_files)}")
        for f in unparsable_files[:20]:
            print(f"  {os.path.basename(f)}")
        if len(unparsable_files) > 20:
            print("  ...")

    expected = expected_times_for_year(args.year)
    expected_set = set(expected.tolist())
    seen_set = set(seen_times.keys())

    missing = sorted(expected_set - seen_set)
    extra = sorted(t for t in seen_set if t < expected[0] or t >= expected[-1] + np.timedelta64(6, "h"))

    duplicate_times = {
        t: paths
        for t, paths in seen_times.items()
        if len(paths) > 1
    }

    print("\nTime coverage summary:")
    print(f"  Expected 6-hourly timesteps in {args.year}: {len(expected)}")
    print(f"  Seen unique timestamps: {len(seen_set)}")
    print(f"  Missing timestamps: {len(missing)}")
    print(f"  Duplicate timestamps: {len(duplicate_times)}")
    print(f"  NaN files: {len(nan_files)}")
    print(f"  Unexpected shape files: {len(unexpected_shape_files)}")

    if missing:
        print("\nMissing timestamps:")
        for t in missing:
            print(f"  {np.datetime_as_string(t, unit='h')}")

    if duplicate_times:
        print("\nDuplicate timestamps:")
        for t, paths in sorted(duplicate_times.items()):
            print(f"  {np.datetime_as_string(t, unit='h')}: {len(paths)} files")
            for p in paths:
                print(f"    {os.path.basename(p)}")

    if extra:
        print("\nTimestamps outside requested year:")
        for t in extra[:50]:
            print(f"  {np.datetime_as_string(t, unit='h')}")
        if len(extra) > 50:
            print("  ...")

    # Per-day summary: useful for spotting missing chunks.
    day_counts = {}
    for t in seen_set:
        day = np.datetime64(t, "D")
        day_counts[day] = day_counts.get(day, 0) + 1

    expected_days = np.arange(
        np.datetime64(f"{args.year}-01-01", "D"),
        np.datetime64(f"{args.year + 1}-01-01", "D"),
        np.timedelta64(1, "D"),
    )

    incomplete_days = []
    for day in expected_days:
        count = day_counts.get(day, 0)
        if count != 4:
            incomplete_days.append((day, count))

    print("\nIncomplete day summary:")
    print(f"  Days with != 4 files: {len(incomplete_days)}")
    for day, count in incomplete_days:
        print(f"  {np.datetime_as_string(day, unit='D')}: {count}/4 files")


if __name__ == "__main__":
    main()