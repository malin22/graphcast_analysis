import os
import numpy as np
import xarray as xr
import gcsfs
from google.cloud import storage

#the variables graphcast expects as input
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


def load_era5_into_memory(start: str, end: str, zarr_path: str = "gs://weatherbench2/datasets/era5/1959-2022-full_37-6h-0p25deg_derived.zarr", vars_keep=None,):
    """
    Load a slice of ERA5 data fully into memory.

    Parameters
    ----------
    start, end : str
        Date range (YYYY-MM-DD)
    zarr_path : str
        GCS or local Zarr path
    vars_keep : list[str] | None
        Variables to keep (None = all)

    Returns
    -------
    xarray.Dataset
        Fully-loaded dataset in memory
    """
    if vars_keep is None:
      vars_keep = DEFAULT_VARS

    start = np.datetime64(start)
    end = np.datetime64(end)

    # --- Open Zarr store ---
    if zarr_path.startswith("gs://"):
        fs = gcsfs.GCSFileSystem(token="anon")
        store = fs.get_mapper(zarr_path[5:])
        ds = xr.open_zarr(store, consolidated=True)
    else:
        ds = xr.open_zarr(zarr_path, consolidated=True)

    # --- Normalize coords ---
    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    if ds.lat[0] > ds.lat[-1]:
        ds = ds.reindex(lat=ds.lat[::-1])

    # --- Time slice ---
    ds = ds.sel(time=slice(start, end))

    # --- Variable selection ---
    if vars_keep is not None:
        ds = ds[[v for v in vars_keep if v in ds.data_vars]]

    # --- LOAD EVERYTHING INTO MEMORY ---
    ds = ds.load()

    return ds

def write_daily_era5_files(ds: xr.Dataset, out_dir: str):
    """
    Write an in-memory ERA5 Dataset to daily NetCDF files
    compatible with three_step_window().

    Assumes:
      - ds has a 'time' coordinate of type datetime64
      - 6-hourly (or finer) resolution
    """

    os.makedirs(out_dir, exist_ok=True)

    # Group by day
    for day, ds_day in ds.groupby("time.date"):
        day_str = np.datetime_as_string(np.datetime64(day), unit="D")
        out_path = os.path.join(out_dir, f"era5_{day_str}.nc")

        # Preserve original encoding as much as possible
        ds_day.to_netcdf(out_path)

        print(f"[WRITE] {out_path}")



if __name__ == "__main__":
    #
    ds = load_era5_into_memory(
        start="2021-08-29",
        end="2021-08-30"
    )

    write_daily_era5_files(
        ds,
        out_dir = '/share/prj-4d/graphcast_shared/data/era5_daily_nc'
    )