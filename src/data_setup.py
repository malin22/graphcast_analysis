import os
import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_VARS = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
    "toa_incident_solar_radiation",
    "geopotential_at_surface",
    "land_sea_mask",
]

def open_era5(
    zarr_path="gs://weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr",
):
    ds = xr.open_zarr(
        zarr_path,
        consolidated=True,
        storage_options={
            "token": "anon",
            "session_kwargs": {"trust_env": True},
        },
    )

    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    if ds.lat[0] > ds.lat[-1]:
        ds = ds.reindex(lat=ds.lat[::-1])

    return ds


def keep_only_available_vars(ds: xr.Dataset, vars_keep=None) -> xr.Dataset:
    if vars_keep is None:
        vars_keep = DEFAULT_VARS
    keep = [v for v in vars_keep if v in ds.data_vars]
    return ds[keep]


def write_daily_era5_files(ds: xr.Dataset, start: str, end: str, out_dir: str):
    """
    Write one NetCDF file per day between start and end inclusive.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Inclusive date range by calendar day
    days = pd.date_range(start=start, end=end, freq="D")

    print(f"Writing {len(days)} daily NetCDF files...")

    for day in days:
        day64 = np.datetime64(day.date(), "D")
        next_day64 = day64 + np.timedelta64(1, "D")
        day_str = str(day.date())
        out_path = os.path.join(out_dir, f"era5_{day_str}.nc")

        if os.path.exists(out_path):
            print(f"Skipping existing file: {out_path}")
            continue

        ds_day = ds.sel(time=slice(day64, next_day64 - np.timedelta64(1, "ns")))

        if ds_day.sizes.get("time", 0) == 0:
            print(f"No data for {day_str}, skipping.")
            continue

        ds_day = ds_day.chunk({"time": 4})  # 6-hourly → 4 steps per day

        encoding = {
            var: {"zlib": True, "complevel": 4}
            for var in ds_day.data_vars
        }

        ds_day.to_netcdf(out_path, encoding=encoding)
        print(f"WRITTEN: {out_path}")


if __name__ == "__main__":
    ds = open_era5()
    ds = keep_only_available_vars(ds, DEFAULT_VARS)

    write_daily_era5_files(
        ds,
        start="2021-01-01",
        end="2021-12-31",
        out_dir="/share/prj-4d/graphcast_shared/data/era5_daily_nc",
    )

    ds.close()
    print("All files written and ds closed")