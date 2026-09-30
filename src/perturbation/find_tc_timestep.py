import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from scipy import ndimage


# ============================================================
# Configuration
# ============================================================

WEATHER_FEATURE = "TC"

MASK_DIR = Path(
    f"/share/prj-4d/graphcast_shared/data/"
    f"ClimateNetLarge/{WEATHER_FEATURE}_labels_cleaned"
)

ERA5_DATA_DIR = Path(
    "/share/prj-4d/graphcast_shared/data/era5_daily_nc"
)

OUTPUT_DIR = Path(
    "plots/perturbation/TC/tc_candidate_plots"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


START_DATE = "2021-01-01"
END_DATE = "2021-12-31 23:00"

MAX_DIFF_HOURS = 3


# ------------------------------------------------------------
# Tracking / diagnostic settings
# ------------------------------------------------------------

# Maximum distance the TC centre may move in one 6 h step.
TRACK_SEARCH_RADIUS_KM = 400.0

# Small distance penalty used when choosing among pressure minima.
# Cost = MSLP + weight * distance/search_radius
TRACK_DISTANCE_WEIGHT_HPA = 2.0

# Calculate Vmax and pmin within this radius of tracked centre.
INTENSITY_RADIUS_KM = 200.0

# Storm-centred map radius.
PLOT_RADIUS_KM = 800.0

# Ignore extremely tiny isolated ClimateNet components.
# Set to 1 if you want literally every positive component.
MIN_COMPONENT_PIXELS = 3


# ------------------------------------------------------------
# Diagnostic panels
# ------------------------------------------------------------

PLOT_LEADS = {
    "t": 0,
    "+6h": 6,
    "+1d": 24,
    "+2d": 48,
    "+3d": 72,
    "+4d": 96,
    "+5d": 120,
}


# ============================================================
# Generic helpers
# ============================================================

def get_lat_lon(obj):
    """Return latitude and longitude coordinates."""

    lat_name = None
    lon_name = None

    for name in ["latitude", "lat"]:
        if name in obj.coords:
            lat_name = name
            break

    for name in ["longitude", "lon"]:
        if name in obj.coords:
            lon_name = name
            break

    if lat_name is None:
        raise KeyError(
            "Could not find latitude coordinate. "
            f"Available: {list(obj.coords)}"
        )

    if lon_name is None:
        raise KeyError(
            "Could not find longitude coordinate. "
            f"Available: {list(obj.coords)}"
        )

    return obj[lat_name], obj[lon_name]


def find_name(ds, candidates):
    """Return first matching variable name."""

    for name in candidates:
        if name in ds:
            return name

    raise KeyError(
        f"None of {candidates} found.\n"
        f"Available variables:\n"
        f"{list(ds.variables)}"
    )


# ============================================================
# Geographic utilities
# ============================================================

def great_circle_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""

    R = 6371.0

    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * R
        * np.arcsin(
            np.sqrt(
                np.clip(a, 0.0, 1.0)
            )
        )
    )


def distance_field(field, center_lat, center_lon):
    """Distance of every grid point from centre."""

    lat_da, lon_da = get_lat_lon(field)

    lat = np.asarray(lat_da)
    lon = np.asarray(lon_da)

    lon2d, lat2d = np.meshgrid(lon, lat)

    return great_circle_distance(
        center_lat,
        center_lon,
        lat2d,
        lon2d,
    )


def set_local_extent(
    ax,
    center_lat,
    center_lon,
    radius_km,
):
    """Set local map extent around TC."""

    dlat = radius_km / 111.0

    coslat = max(
        abs(np.cos(np.deg2rad(center_lat))),
        0.2,
    )

    dlon = radius_km / (
        111.0 * coslat
    )

    ax.set_extent(
        [
            center_lon - dlon,
            center_lon + dlon,
            center_lat - dlat,
            center_lat + dlat,
        ],
        crs=ccrs.PlateCarree(),
    )


# ============================================================
# ClimateNet
# ============================================================

def load_mask_times(mask_dir):
    """Load timestamps of ClimateNet files."""

    mask_files = sorted(
        glob.glob(str(mask_dir / "*.nc"))
    )

    rows = []

    for f in mask_files:
        try:
            t = pd.Timestamp(Path(f).stem)
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

    return (
        df
        .sort_values("time")
        .reset_index(drop=True)
    )


def nearest_mask(mask_df, target_time):
    """Nearest ClimateNet file within MAX_DIFF_HOURS."""

    diffs = (
        mask_df["time"] - target_time
    ).abs()

    idx = diffs.idxmin()

    diff_hours = (
        diffs.loc[idx]
        / pd.Timedelta(hours=1)
    )

    if diff_hours > MAX_DIFF_HOURS:
        return None

    return {
        "target_time": target_time,
        "mask_time": mask_df.loc[idx, "time"],
        "mask_file": mask_df.loc[idx, "file"],
        "diff_hours": float(diff_hours),
    }


def load_tc_mask(
    mask_file,
    label_mode="intersection",
):
    """
    Load ClimateNet TC mask and combine annotators.

    Returns:
        2-D DataArray (latitude, longitude)
    """

    with xr.open_dataset(mask_file) as ds:

        label = ds["label"]

        if label_mode == "intersection":
            mask = label.min("annotator")

        elif label_mode == "union":
            mask = label.max("annotator")

        elif label_mode == "soft":
            mask = label.mean("annotator")

        else:
            raise ValueError(
                f"Unknown label_mode: {label_mode}"
            )

        mask = mask.load()

    return mask


def get_mask_components(mask):
    """
    Extract every connected TC component.

    Each component is treated as a separate candidate cyclone.
    """

    values = (
        np.asarray(mask) > 0.5
    )

    labels, n_components = ndimage.label(
        values
    )

    lat_da, lon_da = get_lat_lon(mask)

    lat = np.asarray(lat_da)
    lon = np.asarray(lon_da)

    components = []

    for component_id in range(
        1,
        n_components + 1,
    ):

        yy, xx = np.where(
            labels == component_id
        )

        if len(yy) < MIN_COMPONENT_PIXELS:
            continue

        component_lat = lat[yy]
        component_lon = lon[xx]

        # Circular mean longitude.
        lon_rad = np.deg2rad(
            component_lon
        )

        mean_lon = np.rad2deg(
            np.arctan2(
                np.mean(np.sin(lon_rad)),
                np.mean(np.cos(lon_rad)),
            )
        )

        components.append({
            "id": int(component_id),
            "lat": float(
                np.mean(component_lat)
            ),
            "lon": float(mean_lon),
            "n_pixels": int(len(yy)),
        })

    return components


# ============================================================
# ERA5
# ============================================================

def load_era5(target_time):
    """Load ERA5 timestep nearest target time."""

    target_time = pd.Timestamp(
        target_time
    )

    era5_file = (
        ERA5_DATA_DIR
        / f"era5_{target_time:%Y-%m-%d}.nc"
    )

    if not era5_file.exists():
        raise FileNotFoundError(
            f"ERA5 file not found: {era5_file}"
        )

    ds = xr.open_dataset(
        era5_file
    )

    time_name = find_name(
        ds,
        [
            "time",
            "valid_time",
            "datetime",
        ],
    )

    available_times = pd.to_datetime(
        ds[time_name].values
    )

    diffs = np.abs(
        available_times - target_time
    )

    idx = int(
        np.argmin(diffs)
    )

    selected_time = available_times[idx]

    diff_hours = abs(
        (selected_time - target_time)
        / pd.Timedelta(hours=1)
    )

    if diff_hours > 3:

        ds.close()

        raise ValueError(
            f"No ERA5 timestep close to "
            f"{target_time}. "
            f"Nearest: {selected_time}"
        )

    ds = (
        ds
        .isel({time_name: idx})
        .load()
    )

    return ds


def get_wind_field(ds):
    """ERA5 10-m wind speed."""

    u_name = find_name(
        ds,
        [
            "10m_u_component_of_wind",
            "u10",
            "10u",
        ],
    )

    v_name = find_name(
        ds,
        [
            "10m_v_component_of_wind",
            "v10",
            "10v",
        ],
    )

    u10 = ds[u_name].squeeze()
    v10 = ds[v_name].squeeze()

    return np.sqrt(
        u10 ** 2 + v10 ** 2
    )


def get_mslp_field(ds):
    """ERA5 mean sea-level pressure in hPa."""

    name = find_name(
        ds,
        [
            "mean_sea_level_pressure",
            "msl",
            "mslp",
            "MSL",
        ],
    )

    mslp = ds[name].squeeze()

    median_value = float(
        np.nanmedian(mslp.values)
    )

    # ERA5 normally stores MSLP in Pa.
    if median_value > 2000:
        mslp = mslp / 100.0

    return mslp


# ============================================================
# TC diagnostics
# ============================================================

def find_local_mslp_center(
    mslp,
    previous_lat,
    previous_lon,
    search_radius_km=TRACK_SEARCH_RADIUS_KM,
):
    """
    Track TC using MSLP with a weak distance penalty.

    cost =
        MSLP
        + TRACK_DISTANCE_WEIGHT_HPA
          * distance / search_radius

    This discourages jumping to another nearby low.
    """

    distance = distance_field(
        mslp,
        previous_lat,
        previous_lon,
    )

    values = np.asarray(mslp)

    valid = (
        np.isfinite(values)
        & (distance <= search_radius_km)
    )

    if not np.any(valid):
        return None

    cost = (
        values
        + TRACK_DISTANCE_WEIGHT_HPA
        * distance
        / search_radius_km
    )

    cost = np.where(
        valid,
        cost,
        np.inf,
    )

    flat_idx = np.argmin(
        cost
    )

    iy, ix = np.unravel_index(
        flat_idx,
        cost.shape,
    )

    lat_da, lon_da = get_lat_lon(
        mslp
    )

    lat = np.asarray(lat_da)
    lon = np.asarray(lon_da)

    return {
        "lat": float(lat[iy]),
        "lon": float(lon[ix]),
        "pmin_at_center": float(
            values[iy, ix]
        ),
        "step_distance_km": float(
            great_circle_distance(
                previous_lat,
                previous_lon,
                lat[iy],
                lon[ix],
            )
        ),
    }


def local_vmax(
    wind,
    center_lat,
    center_lon,
    radius_km=INTENSITY_RADIUS_KM,
):
    """Maximum V10 around tracked centre."""

    distance = distance_field(
        wind,
        center_lat,
        center_lon,
    )

    values = np.asarray(
        wind
    )

    valid = (
        np.isfinite(values)
        & (distance <= radius_km)
    )

    if not np.any(valid):
        return np.nan

    return float(
        np.nanmax(
            values[valid]
        )
    )


def local_pmin(
    mslp,
    center_lat,
    center_lon,
    radius_km=INTENSITY_RADIUS_KM,
):
    """Minimum MSLP around tracked centre."""

    distance = distance_field(
        mslp,
        center_lat,
        center_lon,
    )

    values = np.asarray(
        mslp
    )

    valid = (
        np.isfinite(values)
        & (distance <= radius_km)
    )

    if not np.any(valid):
        return np.nan

    return float(
        np.nanmin(
            values[valid]
        )
    )


# ============================================================
# Track one ClimateNet component
# ============================================================

def track_component(
    center_time,
    component,
):
    """
    Track one +6 h ClimateNet TC component through ERA5.

    +6 h ClimateNet gives the identity anchor.

    ERA5 MSLP is then used:
        backward to t
        forward every 6 h to +5 d.
    """

    anchor_time = (
        center_time
        + pd.Timedelta(hours=6)
    )

    anchor_lat = component["lat"]
    anchor_lon = component["lon"]

    track = {}

    # --------------------------------------------------------
    # Refine +6 h ClimateNet centre with ERA5 MSLP
    # --------------------------------------------------------

    ds = load_era5(
        anchor_time
    )

    wind = get_wind_field(
        ds
    ).load()

    mslp = get_mslp_field(
        ds
    ).load()

    refined = find_local_mslp_center(
        mslp,
        anchor_lat,
        anchor_lon,
    )

    ds.close()

    if refined is None:
        return None

    current_lat = refined["lat"]
    current_lon = refined["lon"]

    track[6] = {
        "time": anchor_time,
        "lat": current_lat,
        "lon": current_lon,
        "vmax": local_vmax(
            wind,
            current_lat,
            current_lon,
        ),
        "pmin": local_pmin(
            mslp,
            current_lat,
            current_lon,
        ),
        "wind": wind,
        "mslp": mslp,
        "step_distance_km":
            refined["step_distance_km"],
    }

    # --------------------------------------------------------
    # Track backward from +6 h to t
    # --------------------------------------------------------

    t0 = center_time

    ds = load_era5(
        t0
    )

    wind = get_wind_field(
        ds
    ).load()

    mslp = get_mslp_field(
        ds
    ).load()

    backward = find_local_mslp_center(
        mslp,
        current_lat,
        current_lon,
    )

    ds.close()

    if backward is None:
        return None

    track[0] = {
        "time": t0,
        "lat": backward["lat"],
        "lon": backward["lon"],
        "vmax": local_vmax(
            wind,
            backward["lat"],
            backward["lon"],
        ),
        "pmin": local_pmin(
            mslp,
            backward["lat"],
            backward["lon"],
        ),
        "wind": wind,
        "mslp": mslp,
        "step_distance_km":
            backward["step_distance_km"],
    }

    # --------------------------------------------------------
    # Forward track from +6 h to +5 d
    # --------------------------------------------------------

    current_lat = track[6]["lat"]
    current_lon = track[6]["lon"]

    for lead_hours in range(
        12,
        121,
        6,
    ):

        target_time = (
            center_time
            + pd.Timedelta(
                hours=lead_hours
            )
        )

        ds = load_era5(
            target_time
        )

        wind = get_wind_field(
            ds
        ).load()

        mslp = get_mslp_field(
            ds
        ).load()

        next_center = (
            find_local_mslp_center(
                mslp,
                current_lat,
                current_lon,
            )
        )

        ds.close()

        if next_center is None:
            return None

        current_lat = next_center["lat"]
        current_lon = next_center["lon"]

        track[lead_hours] = {
            "time": target_time,
            "lat": current_lat,
            "lon": current_lon,
            "vmax": local_vmax(
                wind,
                current_lat,
                current_lon,
            ),
            "pmin": local_pmin(
                mslp,
                current_lat,
                current_lon,
            ),
            "wind": wind,
            "mslp": mslp,
            "step_distance_km":
                next_center["step_distance_km"],
        }

    return track


# ============================================================
# ClimateNet diagnostic for arbitrary lead
# ============================================================

def get_mask_for_plot(
    mask_df,
    target_time,
):
    """
    Return nearby ClimateNet TC mask if one exists
    and actually contains TC pixels.

    This is ONLY for plotting/validation.
    It does not decide whether the candidate is accepted.
    """

    match = nearest_mask(
        mask_df,
        target_time,
    )

    if match is None:
        return None, None

    mask = load_tc_mask(
        match["mask_file"]
    )

    if not np.any(
        np.asarray(mask) > 0.5
    ):
        return match, None

    return match, mask


# ============================================================
# Plot one candidate
# ============================================================

def plot_candidate(
    center_time,
    component,
    track,
    mask_df,
    candidate_number,
):
    """
    Plot:
        t, +6h, +1d, +2d, +3d, +4d, +5d

    Every panel is centred on the ERA5 tracked TC.
    """

    plotted_leads = list(
        PLOT_LEADS.values()
    )

    local_maxima = [
        track[h]["vmax"]
        for h in plotted_leads
        if (
            h in track
            and np.isfinite(
                track[h]["vmax"]
            )
        )
    ]

    vmax_plot = max(
        max(local_maxima),
        10.0,
    )

    fig, axes = plt.subplots(
        1,
        len(PLOT_LEADS),
        figsize=(25, 5.4),
        subplot_kw={
            "projection":
                ccrs.PlateCarree()
        },
        constrained_layout=True,
    )

    mappable = None

    for ax, (
        label,
        lead_hours,
    ) in zip(
        axes,
        PLOT_LEADS.items(),
    ):

        item = track[
            lead_hours
        ]

        wind = item["wind"]
        mslp = item["mslp"]

        center_lat = item["lat"]
        center_lon = item["lon"]

        lat_da, lon_da = (
            get_lat_lon(wind)
        )

        # ----------------------------------------------------
        # Local storm-centred map
        # ----------------------------------------------------

        set_local_extent(
            ax,
            center_lat,
            center_lon,
            PLOT_RADIUS_KM,
        )

        mappable = ax.pcolormesh(
            lon_da,
            lat_da,
            wind,
            shading="auto",
            vmin=0,
            vmax=vmax_plot,
            cmap="viridis",
            transform=ccrs.PlateCarree(),
        )

        # ----------------------------------------------------
        # MSLP contours
        # ----------------------------------------------------

        mslp_lat, mslp_lon = (
            get_lat_lon(mslp)
        )

        distance = distance_field(
            mslp,
            center_lat,
            center_lon,
        )

        values = np.asarray(
            mslp
        )

        local = (
            np.isfinite(values)
            & (
                distance
                <= PLOT_RADIUS_KM
            )
        )

        if np.any(local):

            local_min = float(
                np.nanmin(
                    values[local]
                )
            )

            local_max = float(
                np.nanmax(
                    values[local]
                )
            )

            first_level = (
                np.floor(
                    local_min / 4.0
                )
                * 4.0
            )

            last_level = (
                np.ceil(
                    local_max / 4.0
                )
                * 4.0
            )

            levels = np.arange(
                first_level,
                last_level + 4.0,
                4.0,
            )

            if len(levels) >= 2:

                ax.contour(
                    mslp_lon,
                    mslp_lat,
                    mslp,
                    levels=levels,
                    colors="white",
                    linewidths=0.65,
                    alpha=0.8,
                    transform=ccrs.PlateCarree(),
                )

        # ----------------------------------------------------
        # ClimateNet overlay, if available
        # ----------------------------------------------------

        mask_match, mask = (
            get_mask_for_plot(
                mask_df,
                item["time"],
            )
        )

        if mask is not None:

            mask_lat, mask_lon = (
                get_lat_lon(mask)
            )

            ax.contour(
                mask_lon,
                mask_lat,
                mask,
                levels=[0.5],
                colors="red",
                linewidths=2.0,
                transform=ccrs.PlateCarree(),
            )

            mask_status = (
                "ClimateNet TC"
            )

        elif mask_match is not None:

            mask_status = (
                "ClimateNet: no TC"
            )

        else:

            mask_status = (
                "no ClimateNet mask"
            )

        # ----------------------------------------------------
        # Tracked ERA5 centre
        # ----------------------------------------------------

        ax.plot(
            center_lon,
            center_lat,
            marker="x",
            markersize=9,
            markeredgewidth=2.3,
            color="black",
            transform=ccrs.PlateCarree(),
            zorder=20,
        )

        # ----------------------------------------------------
        # Geography
        # ----------------------------------------------------

        ax.coastlines(
            linewidth=0.7,
            color="0.3",
        )

        ax.add_feature(
            cfeature.BORDERS,
            linewidth=0.3,
            edgecolor="0.5",
        )

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.3,
            alpha=0.4,
        )

        gl.top_labels = False
        gl.right_labels = False

        # ----------------------------------------------------
        # Title
        # ----------------------------------------------------

        ax.set_title(
            f"{label}\n"
            f"{item['time']:%Y-%m-%d %H}Z\n"
            f"Vmax={item['vmax']:.1f} m/s\n"
            f"pmin={item['pmin']:.1f} hPa\n"
            f"{mask_status}",
            fontsize=8.5,
        )

    cbar = fig.colorbar(
        mappable,
        ax=axes,
        orientation="vertical",
        fraction=0.018,
        pad=0.015,
    )

    cbar.set_label(
        r"ERA5 10-m wind speed [m s$^{-1}$]"
    )

    fig.suptitle(
        "ERA5 TC candidate "
        f"{candidate_number} — "
        f"initialization "
        f"{center_time:%Y-%m-%d %H} UTC\n"
        f"+6 h ClimateNet component "
        f"{component['id']} | "
        f"{component['n_pixels']} mask pixels",
        fontsize=13,
    )

    output_file = (
        OUTPUT_DIR
        / (
            f"TC_candidate_"
            f"{center_time:%Y%m%d_%H}_"
            f"component_{component['id']:02d}.png"
        )
    )

    fig.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_file


# ============================================================
# Main
# ============================================================

def main():

    mask_df = load_mask_times(
        MASK_DIR
    )

    print(
        f"Loaded {len(mask_df)} "
        "ClimateNet mask files."
    )

    candidate_centers = pd.date_range(
        START_DATE,
        END_DATE,
        freq="6h",
    )

    # --------------------------------------------------------
    # Find every actual +6 h TC component
    # --------------------------------------------------------

    candidate_components = []

    for center_time in candidate_centers:

        anchor_time = (
            center_time
            + pd.Timedelta(hours=6)
        )

        match = nearest_mask(
            mask_df,
            anchor_time,
        )

        if match is None:
            continue

        mask = load_tc_mask(
            match["mask_file"]
        )

        components = get_mask_components(
            mask
        )

        for component in components:

            candidate_components.append({
                "center_time":
                    center_time,

                "anchor_time":
                    anchor_time,

                "mask_time":
                    match["mask_time"],

                "mask_diff_hours":
                    match["diff_hours"],

                "component":
                    component,
            })

    print()
    print(
        f"Found {len(candidate_components)} "
        "individual +6 h ClimateNet "
        "TC components."
    )
    print()

    # --------------------------------------------------------
    # Track every component independently
    # --------------------------------------------------------

    records = []
    failures = []

    for candidate_number, candidate in enumerate(
        candidate_components,
        start=1,
    ):

        center_time = (
            candidate["center_time"]
        )

        component = (
            candidate["component"]
        )

        print(
            f"[{candidate_number}/"
            f"{len(candidate_components)}] "
            f"{center_time} | "
            f"component {component['id']} | "
            f"centre "
            f"({component['lat']:.1f}, "
            f"{component['lon']:.1f})"
        )

        try:

            track = track_component(
                center_time,
                component,
            )

            if track is None:

                print(
                    "  FAILED: ERA5 tracking"
                )

                failures.append({
                    "center_time":
                        center_time,
                    "component_id":
                        component["id"],
                    "reason":
                        "ERA5 tracking failed",
                })

                continue

            output_file = (
                plot_candidate(
                    center_time,
                    component,
                    track,
                    mask_df,
                    candidate_number,
                )
            )

            print(
                f"  -> {output_file}"
            )

            # -----------------------------------------------
            # Save diagnostics
            # -----------------------------------------------

            record = {
                "center_time":
                    center_time,

                "anchor_time":
                    candidate[
                        "anchor_time"
                    ],

                "mask_time":
                    candidate[
                        "mask_time"
                    ],

                "component_id":
                    component["id"],

                "component_pixels":
                    component[
                        "n_pixels"
                    ],

                "climatenet_lat_6h":
                    component["lat"],

                "climatenet_lon_6h":
                    component["lon"],
            }

            for label, lead_hours in (
                PLOT_LEADS.items()
            ):

                key = (
                    label
                    .replace("+", "")
                    .replace("d", "d")
                )

                item = track[
                    lead_hours
                ]

                record[
                    f"lat_{key}"
                ] = item["lat"]

                record[
                    f"lon_{key}"
                ] = item["lon"]

                record[
                    f"vmax_{key}"
                ] = item["vmax"]

                record[
                    f"pmin_{key}"
                ] = item["pmin"]

            records.append(
                record
            )

        except Exception as exc:

            print(
                f"  FAILED: {exc}"
            )

            failures.append({
                "center_time":
                    center_time,

                "component_id":
                    component["id"],

                "reason":
                    str(exc),
            })

    # --------------------------------------------------------
    # Save tables
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        records
    )

    failures_df = pd.DataFrame(
        failures
    )

    results_file = (
        OUTPUT_DIR
        / "tc_candidate_tracks.csv"
    )

    failures_file = (
        OUTPUT_DIR
        / "tc_candidate_failures.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    failures_df.to_csv(
        failures_file,
        index=False,
    )

    print()
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)

    print(
        f"Tracked candidates: "
        f"{len(results_df)}"
    )

    print(
        f"Tracking failures: "
        f"{len(failures_df)}"
    )

    print(
        f"Results: {results_file}"
    )

    print(
        f"Failures: {failures_file}"
    )


if __name__ == "__main__":
    main()