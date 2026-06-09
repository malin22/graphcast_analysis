import os
import numpy as np
import xarray as xr
import pandas as pd

from graphcast import icosahedral_mesh
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def vertices_to_latlon(vertices):
    lat = np.degrees(np.arcsin(vertices[:, 2]))
    lon = np.degrees(np.arctan2(vertices[:, 1], vertices[:, 0]))
    lon = lon % 360
    return lat, lon


def get_mesh_latlon(splits=6):
    meshes = icosahedral_mesh.get_hierarchy_of_triangular_meshes_for_sphere(
        splits=splits
    )
    vertices = meshes[splits].vertices
    return vertices_to_latlon(vertices)


def corr(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[mask], b[mask])[0, 1]


def main():
    PC_SCORES_PATH = "plots/2021_pca_projected_on_2021/mean_pc_scores_year.npy"
    ERA5_FILE = "/share/prj-4d/graphcast_shared/data/era5_daily_nc/era5_2021-01-01.nc"
    OUT_DIR = "plots/2021_pc_analysis_geographic_features"
    os.makedirs(OUT_DIR, exist_ok=True)

    # -----------------------------
    # Load PC scores
    # -----------------------------
    pc_scores = np.load(PC_SCORES_PATH)  # [nodes, pcs]
    print("\nLoaded PC scores")
    print("PC scores shape:", pc_scores.shape)

    # -----------------------------
    # Load ERA5 static variables
    # -----------------------------
    ds = xr.open_dataset(ERA5_FILE)

    lsm = ds["land_sea_mask"]
    surface_geopotential = ds["geopotential_at_surface"]

    print("\nERA5 variables")
    print("Land-sea mask shape:", lsm.shape)
    print("Surface geopotential shape:", surface_geopotential.shape)

    # -----------------------------
    # Mesh node coordinates
    # -----------------------------
    lat, lon = get_mesh_latlon(splits=6)

    if len(lat) != pc_scores.shape[0]:
        raise ValueError(
            f"Node mismatch: mesh has {len(lat)} nodes, "
            f"PC scores have {pc_scores.shape[0]} nodes."
        )

    print("\nMesh coordinates")
    print("lat shape:", lat.shape)
    print("lon shape:", lon.shape)
    print("lat range:", lat.min(), lat.max())
    print("lon range:", lon.min(), lon.max())

    # -----------------------------
    # Sample ERA5 variables at mesh nodes
    # -----------------------------
    node_lat = xr.DataArray(lat, dims="node")
    node_lon = xr.DataArray(lon, dims="node")

    lsm_mesh = lsm.interp(
        lat=node_lat,
        lon=node_lon,
        method="nearest",
    ).values

    geopotential_mesh = surface_geopotential.interp(
        lat=node_lat,
        lon=node_lon,
        method="nearest",
    ).values

    # Convert geopotential to approximate elevation in meters
    elevation_mesh = geopotential_mesh / 9.80665

    land_binary = (lsm_mesh > 0.5).astype(int)

    print("\nSampled physical variables")
    print("elevation min/max:", np.nanmin(elevation_mesh), np.nanmax(elevation_mesh))

    # -----------------------------
    # Build geographic predictor matrix
    # -----------------------------
    X = np.stack(
        [
            lsm_mesh,
            land_binary,
            lat,
            np.sin(np.deg2rad(lon)),
            np.cos(np.deg2rad(lon)),
            elevation_mesh,
        ],
        axis=1,
    )

    predictor_names = [
        "fractional_lsm",
        "binary_land",
        "latitude",
        "sin_longitude",
        "cos_longitude",
        "elevation",
    ]

    valid = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(pc_scores), axis=1)

    X = X[valid]
    pc_scores_valid = pc_scores[valid]

    X_std = StandardScaler().fit_transform(X)

    print("\nValid nodes:", X.shape[0])

    # -----------------------------
    # Correlations and regressions
    # -----------------------------
    results = []

    for pc_idx in range(pc_scores_valid.shape[1]):
        pc = pc_scores_valid[:, pc_idx]

        model = LinearRegression()
        model.fit(X_std, pc)
        r2_geo = model.score(X_std, pc)

        row = {
            "PC": pc_idx + 1,
            "corr_fractional_lsm": corr(pc, X[:, 0]),
            "corr_binary_land": corr(pc, X[:, 1]),
            "corr_latitude": corr(pc, X[:, 2]),
            "corr_sin_longitude": corr(pc, X[:, 3]),
            "corr_cos_longitude": corr(pc, X[:, 4]),
            "corr_elevation": corr(pc, X[:, 5]),
            "R2_all_geography": r2_geo,
        }

        for name, coef in zip(predictor_names, model.coef_):
            row[f"beta_{name}"] = coef

        # strongest absolute single-variable correlation
        corr_cols = [
            "corr_fractional_lsm",
            "corr_binary_land",
            "corr_latitude",
            "corr_sin_longitude",
            "corr_cos_longitude",
            "corr_elevation",
        ]

        strongest = max(corr_cols, key=lambda c: abs(row[c]))
        row["strongest_single_predictor"] = strongest.replace("corr_", "")
        row["strongest_abs_corr"] = abs(row[strongest])

        results.append(row)

    df = pd.DataFrame(results)

    # -----------------------------
    # Print summary
    # -----------------------------
    # -----------------------------
    # Save nicer result tables
    # -----------------------------
    summary_cols = [
        "PC",
        "R2_all_geography",
        "strongest_single_predictor",
        "strongest_abs_corr",
        "corr_fractional_lsm",
        "corr_latitude",
        "corr_elevation",
    ]

    df_summary = df[summary_cols].copy()
    df_summary = df_summary.sort_values("R2_all_geography", ascending=False)

    csv_path = os.path.join(OUT_DIR, "pc_geography_full_results.csv")
    summary_csv_path = os.path.join(OUT_DIR, "pc_geography_summary.csv")


    df.to_csv(csv_path, index=False)
    df_summary.to_csv(summary_csv_path, index=False)

    df_summary_rounded = df_summary.copy()

    for col in [
        "R2_all_geography",
        "strongest_abs_corr",
        "corr_fractional_lsm",
        "corr_latitude",
        "corr_elevation",
    ]:
        df_summary_rounded[col] = df_summary_rounded[col].round(3)


    print(f"\nSaved full table to:\n{csv_path}")
    print(f"Saved summary table to:\n{summary_csv_path}")

    # -----------------------------
    # Plot 1: R² explained by geography
    # -----------------------------
    plt.figure(figsize=(8, 4))
    plt.bar(df["PC"].astype(str), df["R2_all_geography"])
    plt.axhline(0.5, linestyle="--", linewidth=1, label="geographic threshold = 0.5")
    plt.axhline(0.3, linestyle=":", linewidth=1, label="non-geographic threshold = 0.3")
    plt.xlabel("Principal component")
    plt.ylabel("R² explained by geography")
    plt.title("How much static geography explains each PC")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    r2_plot_path = os.path.join(OUT_DIR, "pc_geography_r2_barplot.png")
    plt.savefig(r2_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

        # -----------------------------
    # Plot 2: absolute correlation heatmap
    # -----------------------------
    corr_cols = [
        "corr_fractional_lsm",
        "corr_binary_land",
        "corr_latitude",
        "corr_sin_longitude",
        "corr_cos_longitude",
        "corr_elevation",
    ]

    corr_matrix = np.abs(df[corr_cols].to_numpy())

    plt.figure(figsize=(10, 5))
    im = plt.imshow(corr_matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, label="Absolute correlation")

    plt.yticks(
        ticks=np.arange(len(df)),
        labels=[f"PC{pc}" for pc in df["PC"]],
    )
    plt.xticks(
        ticks=np.arange(len(corr_cols)),
        labels=[
            "fractional LSM",
            "binary land",
            "latitude",
            "sin(lon)",
            "cos(lon)",
            "elevation",
        ],
        rotation=45,
        ha="right",
    )

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            plt.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.title("Absolute correlation between PCs and geographic variables")
    plt.tight_layout()

    corr_plot_path = os.path.join(OUT_DIR, "pc_geography_absolute_correlation_heatmap.png")
    plt.savefig(corr_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Plot 3: regression coefficient heatmap
    # -----------------------------
    beta_cols = [f"beta_{name}" for name in predictor_names]
    beta_matrix = df[beta_cols].to_numpy()

    max_abs_beta = np.nanmax(np.abs(beta_matrix))

    plt.figure(figsize=(10, 5))
    im = plt.imshow(
        beta_matrix,
        aspect="auto",
        vmin=-max_abs_beta,
        vmax=max_abs_beta,
        cmap="coolwarm",
    )
    plt.colorbar(im, label="Standardized regression coefficient")

    plt.yticks(
        ticks=np.arange(len(df)),
        labels=[f"PC{pc}" for pc in df["PC"]],
    )
    plt.xticks(
        ticks=np.arange(len(beta_cols)),
        labels=[
            "fractional LSM",
            "binary land",
            "latitude",
            "sin(lon)",
            "cos(lon)",
            "elevation",
        ],
        rotation=45,
        ha="right",
    )

    for i in range(beta_matrix.shape[0]):
        for j in range(beta_matrix.shape[1]):
            plt.text(
                j,
                i,
                f"{beta_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.title("Standardized geographic regression coefficients")
    plt.tight_layout()

    beta_plot_path = os.path.join(OUT_DIR, "pc_geography_regression_coefficients.png")
    plt.savefig(beta_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved R² bar plot to:\n{r2_plot_path}")
    print(f"Saved correlation heatmap to:\n{corr_plot_path}")
    print(f"Saved regression coefficient heatmap to:\n{beta_plot_path}")

    # -----------------------------
    # Identify likely geographic / non-geographic PCs
    # -----------------------------
    geo_threshold = 0.5
    non_geo_threshold = 0.3

    geo_pcs = df[df["R2_all_geography"] >= geo_threshold]["PC"].tolist()
    non_geo_pcs = df[df["R2_all_geography"] < non_geo_threshold]["PC"].tolist()

    print("\nLikely geography-dominated PCs:")
    print(geo_pcs)

    print("\nCandidate non-geographic / dynamic PCs:")
    print(non_geo_pcs)


if __name__ == "__main__":
    main()