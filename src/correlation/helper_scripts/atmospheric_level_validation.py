#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import xarray as xr


NC_PATH = Path("/share/prj-4d/graphcast_shared/data/era5_daily_nc/era5_2021-01-15.nc")
MESH_ROOT = Path("/share/prj-4d/graphcast_shared/data/era5_daily_mesh/2021/mesh_l6")
CENTER_STR = "2021-01-15T12"

PRESSURE_LEVELS_HPA_EXPECTED = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200,
    225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
    775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
]


def inspect_nc_levels(path):
    print("\n=== NetCDF level inspection ===")
    print(path)

    ds = xr.open_dataset(path, decode_times=False)

    print("\nDims:")
    print(dict(ds.sizes))

    print("\nCoords:")
    print(list(ds.coords))

    level_coord = None
    for candidate in ["level", "pressure_level", "isobaricInhPa"]:
        if candidate in ds.coords:
            level_coord = candidate
            break

    if level_coord is None:
        print("Could not find a level coordinate.")
        print("Available coords:", list(ds.coords))
        return None

    levels = np.asarray(ds[level_coord].values)
    print(f"\nLevel coord: {level_coord}")
    print("Levels:")
    for i, lev in enumerate(levels):
        print(f"  index {i:02d}: {lev}")

    expected = np.asarray(PRESSURE_LEVELS_HPA_EXPECTED)

    if len(levels) == len(expected):
        same = np.allclose(levels.astype(float), expected.astype(float))
        reversed_same = np.allclose(levels.astype(float), expected[::-1].astype(float))

        print("\nMatches expected lev00=1hPa ... lev36=1000hPa:", same)
        print("Matches reversed lev00=1000hPa ... lev36=1hPa:", reversed_same)

        if same:
            print("OK: NetCDF level index 0 is 1 hPa, index 36 is 1000 hPa.")
        elif reversed_same:
            print("WARNING: NetCDF levels appear reversed relative to your levXX mapping.")
        else:
            print("WARNING: NetCDF levels differ from expected GraphCast pressure levels.")
    else:
        print(f"WARNING: NetCDF has {len(levels)} levels, expected {len(expected)}")

    ds.close()
    return levels


def load_time_index(mesh_root):
    time_values = np.load(mesh_root / "time_values.npy", allow_pickle=False)
    return {
        np.datetime_as_string(np.datetime64(t), unit="h"): i
        for i, t in enumerate(time_values)
    }


def inspect_mesh_profiles(mesh_root, center_str):
    print("\n=== Mesh .npy vertical profile inspection ===")
    print(mesh_root)

    time_index = load_time_index(mesh_root)
    if center_str not in time_index:
        raise ValueError(f"{center_str} not found in {mesh_root / 'time_values.npy'}")

    t_idx = time_index[center_str]
    print(f"Using mesh timestep {center_str}, index {t_idx}")

    for variable in ["temperature", "specific_humidity", "geopotential"]:
        print(f"\n{variable}")
        print("lev   expected_hPa       mean           p05           p50           p95")
        print("-" * 78)

        for lev_idx, hpa in enumerate(PRESSURE_LEVELS_HPA_EXPECTED):
            path = mesh_root / "time_series" / f"{variable}_lev{lev_idx:02d}.npy"
            if not path.exists():
                print(f"lev{lev_idx:02d} {hpa:5d}: missing {path.name}")
                continue

            arr = np.load(path, mmap_mode="r")
            vals = np.asarray(arr[t_idx], dtype=np.float32)

            print(
                f"{lev_idx:02d}    {hpa:5d}     "
                f"{np.nanmean(vals):12.5g}  "
                f"{np.nanpercentile(vals, 5):12.5g}  "
                f"{np.nanpercentile(vals, 50):12.5g}  "
                f"{np.nanpercentile(vals, 95):12.5g}"
            )


def main():
    levels = inspect_nc_levels(NC_PATH)
    inspect_mesh_profiles(MESH_ROOT, CENTER_STR)

    print("\nInterpretation checks:")
    print("1. If NetCDF index 00 is 1 hPa and mesh lev00 is also treated as 1 hPa, naming is consistent.")
    print("2. Specific humidity should generally be tiny at low lev indices and much larger near lev36.")
    print("3. Geopotential should generally be largest at low lev indices and smallest near lev36.")
    print("4. Temperature is not monotonic through the full column, so use it as a weaker sanity check.")


if __name__ == "__main__":
    main()