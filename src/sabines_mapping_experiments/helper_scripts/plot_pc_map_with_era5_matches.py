#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


HPA_BY_LEV = {
    0: 1, 1: 2, 2: 3, 3: 5, 4: 7, 5: 10,
    6: 20, 7: 30, 8: 50, 9: 70,
    10: 100, 11: 125, 12: 150,
    13: 175, 14: 200, 15: 225, 16: 250, 17: 300, 18: 350,
    19: 400, 20: 450, 21: 500, 22: 550, 23: 600, 24: 650,
    25: 700, 26: 750, 27: 775, 28: 800, 29: 825, 30: 850,
    31: 875, 32: 900, 33: 925, 34: 950, 35: 975, 36: 1000,
}


def atmospheric_band(lev_idx):
    if 0 <= lev_idx <= 5:
        return "upper stratosphere / mesosphere"
    if 6 <= lev_idx <= 9:
        return "middle-to-lower stratosphere"
    if 10 <= lev_idx <= 12:
        return "tropopause / upper troposphere"
    if 13 <= lev_idx <= 18:
        return "upper troposphere"
    if 19 <= lev_idx <= 24:
        return "mid troposphere"
    if 25 <= lev_idx <= 30:
        return "lower troposphere"
    if 31 <= lev_idx <= 36:
        return "boundary layer / near-surface"
    return "single-level / static"


PRETTY_BASE = {
    "geopotential": "Geopotential",
    "specific_humidity": "Specific humidity",
    "temperature": "Temperature",
    "u_component_of_wind": "U wind",
    "v_component_of_wind": "V wind",
    "vertical_velocity": "Vertical velocity",
    "2m_temperature": "2m temperature",
    "10m_u_component_of_wind": "10m U wind",
    "10m_v_component_of_wind": "10m V wind",
    "mean_sea_level_pressure": "Mean sea-level pressure",
    "total_precipitation_6hr": "Total precipitation 6h",
    "toa_incident_solar_radiation": "TOA solar radiation",
    "geopotential_at_surface": "Surface geopotential",
    "land_sea_mask": "Land-sea mask",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "latitude_sin": "Latitude sin",
    "latitude_cos": "Latitude cos",
    "longitude_sin": "Longitude sin",
    "longitude_cos": "Longitude cos",
    "local_time_sin": "Local time sin",
    "local_time_cos": "Local time cos",
    "year_progress_sin": "Year progress sin",
    "year_progress_cos": "Year progress cos",
}


VAR_COLORS = {
    "geopotential": "#8c6bb1",
    "specific_humidity": "#2b8cbe",
    "temperature": "#d95f0e",
    "u_component_of_wind": "#238b45",
    "v_component_of_wind": "#41ab5d",
    "vertical_velocity": "#006d2c",
    "static": "#636363",
    "surface": "#756bb1",
}


def parse_variable(name):
    m = re.match(r"^(.*)_lev(\d+)$", name)
    if m:
        base = m.group(1)
        lev_idx = int(m.group(2))
        hpa = HPA_BY_LEV.get(lev_idx)
        pretty = PRETTY_BASE.get(base, base.replace("_", " ").title())
        return {
            "raw": name,
            "base": base,
            "label": f"{pretty}, lev{lev_idx:02d}",
            "hpa": hpa,
            "band": atmospheric_band(lev_idx),
            "color_key": base,
        }

    pretty = PRETTY_BASE.get(name, name.replace("_", " ").title())
    color_key = "static" if name in {"land_sea_mask", "latitude", "longitude"} else "surface"
    return {
        "raw": name,
        "base": name,
        "label": pretty,
        "hpa": None,
        "band": "single-level / static context",
        "color_key": color_key,
    }


def get_top_rows(pc_data, top_k):
    rows = pc_data.get("top_variables")
    if rows is None:
        rows = pc_data.get("ranked_variables")
    if rows is None:
        raise KeyError("Expected PC data to contain 'top_variables' or 'ranked_variables'")
    return rows[:top_k]


def row_score(row):
    for key in ["mean_abs_r", "abs_r", "r", "score"]:
        if key in row:
            return float(row[key])
    return np.nan


def draw_pressure_ruler(ax, parsed_rows):
    ax.set_yscale("log")
    ax.set_ylim(1100, 0.8)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([1, 10, 50, 100, 200, 500, 850, 1000])
    ax.set_yticklabels(["1", "10", "50", "100", "200", "500", "850", "1000"], fontsize=9)
    ax.set_ylabel("Pressure [hPa]", fontsize=10, labelpad=8)
    ax.set_title("Atmospheric level", fontsize=11, pad=10)

    ax.grid(axis="y", alpha=0.25, linewidth=0.8)

    band_labels = [
        (5, "upper strat."),
        (45, "lower strat."),
        (125, "tropopause"),
        (250, "upper trop."),
        (525, "mid trop."),
        (775, "lower trop."),
        (950, "boundary"),
    ]

    for y, label in band_labels:
        ax.text(
            0.98,
            y,
            label,
            ha="right",
            va="center",
            fontsize=8,
            color="dimgray",
        )

    for rank, parsed in enumerate(parsed_rows, start=1):
        if parsed["hpa"] is None:
            continue

        color = VAR_COLORS.get(parsed["color_key"], "#444444")
        ax.scatter(
            0.18,
            parsed["hpa"],
            s=120,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
        )
        ax.text(
            0.28,
            parsed["hpa"],
            f"#{rank}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    for spine in ["top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)


def plot_pc_panel(pc_num, map_path, pc_data, out_path, top_k=3, dpi=300):
    img = mpimg.imread(map_path)
    rows = get_top_rows(pc_data, top_k)
    parsed_rows = [parse_variable(row["variable"]) for row in rows]

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[3.7, 1.55, 0.95],
        wspace=0.16,
    )

    ax_map = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    ax_ruler = fig.add_subplot(gs[0, 2])

    ax_map.imshow(img)
    ax_map.axis("off")
    ax_map.set_title(f"PC{pc_num}: year-mean activation map", fontsize=16, pad=14)

    ax_text.axis("off")
    ax_text.set_title("Top ERA5 matches", fontsize=14, loc="left", pad=12)

    y = 0.94
    for rank, (row, parsed) in enumerate(zip(rows, parsed_rows), start=1):
        score = row_score(row)
        color = VAR_COLORS.get(parsed["color_key"], "#444444")

        ax_text.text(
            0.00,
            y,
            f"#{rank}",
            fontsize=13,
            fontweight="bold",
            color=color,
            transform=ax_text.transAxes,
        )
        ax_text.text(
            0.12,
            y,
            parsed["label"],
            fontsize=12,
            fontweight="bold",
            transform=ax_text.transAxes,
        )
        ax_text.text(
            0.12,
            y - 0.07,
            f"mean |r| = {score:.3f}",
            fontsize=11,
            transform=ax_text.transAxes,
        )

        if parsed["hpa"] is not None:
            level_text = f"{parsed['hpa']} hPa; {parsed['band']}"
        else:
            level_text = parsed["band"]

        ax_text.text(
            0.12,
            y - 0.14,
            level_text,
            fontsize=10,
            color="dimgray",
            wrap=True,
            transform=ax_text.transAxes,
        )

        y -= 0.27

    draw_pressure_ruler(ax_ruler, parsed_rows)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create annotated PC map panels with top ERA5 correlation matches."
    )
    parser.add_argument(
        "--correlation-json",
        default='/home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/mapping_experiments/top_512_pcs/correlation_pc_era5_mesh_m6_yearly_top_variables.json',
        type=Path,
        help="JSON containing PC top_variables/ranked_variables.",
    )
    parser.add_argument(
        "--map-template",
        default="/home/student/s/sascholle/share/graphcast_analysis/plots/2019_2020_pca_projected_on_2021/pc{pc}_mean_activation_map_year.png",
        type=str,
        help="Template for PC map PNGs, e.g. /path/pc{pc}_mean_activation_map_year.png",
    )
    parser.add_argument("--out-dir", type=Path, default="/home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/mapping_experiments/top_512_pcs/annotated_pc_maps")
    parser.add_argument("--pcs", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=300)

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.correlation_json, "r") as f:
        data = json.load(f)

    for pc_num in args.pcs:
        pc_key = f"PC_{pc_num}"
        if pc_key not in data:
            print(f"WARNING: missing {pc_key}; skipping")
            continue

        map_path = Path(args.map_template.format(pc=pc_num))
        if not map_path.exists():
            print(f"WARNING: missing map {map_path}; skipping PC{pc_num}")
            continue

        out_path = args.out_dir / f"pc{pc_num:03d}_map_with_era5_matches.png"
        plot_pc_panel(
            pc_num=pc_num,
            map_path=map_path,
            pc_data=data[pc_key],
            out_path=out_path,
            top_k=args.top_k,
            dpi=args.dpi,
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()


'''
  --pcs 1 2 3 4 5 6 7 \
  --top-k 3

'''