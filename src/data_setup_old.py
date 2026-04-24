import os
import numpy as np
import xarray as xr
import gcsfs

# Variables GraphCast expects as input
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


def open_era5(zarr_path = "gs://weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr",):


    ds = xr.open_zarr(
        zarr_path,
        consolidated=True,
        storage_options={
        "token": "anon",
        "session_kwargs": {"trust_env": True},
        },
    )

    # Normalize coordinate names
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    # Ensure ascending latitude
    if ds.lat[0] > ds.lat[-1]:
        ds = ds.reindex(lat=ds.lat[::-1])

    return ds


def subset_era5(ds: xr.Dataset,start: str,end: str,vars_keep=None,):
    """subsetting the data to only keep the timeframe and variables needed"""
    if vars_keep is None:
        vars_keep = DEFAULT_VARS

    start = np.datetime64(start)
    end = np.datetime64(end)

    # Time subset
    ds = ds.sel(time=slice(start, end))

    # Variable subset
    keep = [v for v in vars_keep if v in ds.data_vars]
    ds = ds[keep]
    print("subsetting data done")


    return ds


def write_daily_era5_files(ds: xr.Dataset, out_dir: str):
    """
    Write one NetCDF file per day. Each day is computed/written separately.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Extract unique days from time coordinate
    unique_days = np.unique(ds.time.dt.floor("D").values)

    print("writing NetCDF files now..")
    for day in unique_days:
        day = np.datetime64(day, "D")
        next_day = day + np.timedelta64(1, "D")
        day_str = np.datetime_as_string(day, unit="D")
        out_path = os.path.join(out_dir, f"era5_{day_str}.nc")

        # Select one day only
        ds_day = ds.sel(time=slice(day, next_day - np.timedelta64(1, "ns")))

        # Skip empty selections
        if ds_day.sizes.get("time", 0) == 0:
            continue

        # Trigger reading/writing only for this day
        ds_day.to_netcdf(out_path)

        print(f"WRITEN: {out_path}")


if __name__ == "__main__":
    ds = open_era5()

    ds = subset_era5(
        ds,
        start="2020-08-29",
        end="2020-09-01",
        vars_keep=DEFAULT_VARS,
    )

    write_daily_era5_files(
        ds,
        out_dir='/share/prj-4d/graphcast_shared/data/era5_daily_nc'
    )

    ds.close()  # xarray supports explicit resource cleanup
    print("All files written and ds closed")