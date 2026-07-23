from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_JSON = Path(
    "/home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/"
    "pc_era5_mesh_m6_yearly_top_variables.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PC-to-ERA5 JSON rankings into a tidy table."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to the JSON file with PC to ERA5 variable rankings.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/student/s/sascholle/share/graphcast_analysis/plots/sabines_experiments/mapping_experiments"),
        help="Directory for the output table.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top variables to include per PC.",
    )
    return parser.parse_args()


def load_pc_mapping(json_path: Path, top_k: int) -> pd.DataFrame:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for pc_name, payload in data.items():
        top_variables = payload.get("top_variables", [])[:top_k]
        for rank, item in enumerate(top_variables, start=1):
            variable = item["variable"]
            rows.append(
                {
                    "pc": pc_name,
                    "rank": rank,
                    "variable": variable,
                    "variable_family": variable.rsplit("_lev", 1)[0] if "_lev" in variable else variable,
                    "level": variable.rsplit("_lev", 1)[1] if "_lev" in variable else "",
                    "mean_abs_r": item.get("mean_abs_r"),
                    "wins": item.get("wins"),
                    "count": item.get("count"),
                    "win_rate": (item.get("wins", 0) / item.get("count", 1)) if item.get("count") else None,
                    "total_timesteps": payload.get("total_timesteps"),
                    "mesh_level": payload.get("mesh_level"),
                    "n_selected_nodes": payload.get("n_selected_nodes"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No rows found in {json_path}")

    pc_sort = df["pc"].str.extract(r"PC_(\d+)").astype(int)
    df = df.assign(pc_order=pc_sort[0]).sort_values(["pc_order", "rank"]).drop(columns=["pc_order"])
    return df


def to_markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "pc",
        "rank",
        "variable",
        "variable_family",
        "level",
        "mean_abs_r",
        "wins",
        "count",
        "win_rate",
        "total_timesteps",
        "mesh_level",
        "n_selected_nodes",
    ]
    view = df[cols].copy()
    view["mean_abs_r"] = view["mean_abs_r"].map(lambda x: f"{x:.4f}")
    view["win_rate"] = view["win_rate"].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    return view.to_markdown(index=False)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_pc_mapping(args.json, args.top_k)

    md_path = args.out_dir / f"{args.json.stem}_tidy_table.md"

    md_path.write_text(to_markdown_table(df), encoding="utf-8")

    print(f"Saved Markdown: {md_path}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
