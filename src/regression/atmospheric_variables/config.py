import os

PC_SCORES_PATHS = [
    (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2019_from_2019_2020_pca_per_timestep.npy"
    ),
    (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2020_from_2019_2020_pca_per_timestep.npy"
    ),
    (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2021_from_2019_2020_pca_per_timestep.npy"
    ),
]

TIMESTEP_FILES_TXTS = [
    (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2019_from_2019_2020_pca_per_timestep_files.txt"
    ),
    (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2020_from_2019_2020_pca_per_timestep_files.txt"
    ),
    (
        "/share/prj-4d/graphcast_shared/data/"
        "pc_scores_per_timestep/"
        "pc_scores_2021_from_2019_2020_pca_per_timestep_files.txt"
    ),
]

ERA5_MESH_BASE_DIR = (
    "/share/prj-4d/graphcast_shared/data/era5_daily_mesh"
)

NODE_HIERARCHY_LEVEL = 6

REGRESSION_TYPE = "ridge" # "linear" or "ridge"
SCORE_VALUES = "PCA"

TRAIN_YEARS = [2019, 2020]
TEST_YEARS = [2021]

OUT_DIR = (
    "results/regression/extreme_weather_events/"
    f"{SCORE_VALUES}/{REGRESSION_TYPE}/"
    f"l{NODE_HIERARCHY_LEVEL}_nodes"
)

os.makedirs(OUT_DIR, exist_ok=True)

PC_COUNTS = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300, 400, 512]

TARGETS = [
    {"name": "2t", "var": "2m_temperature", "level": None},
]


PRESSURE_LEVELS = [
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70,
    100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700,
    750, 775, 800, 825, 850, 875, 900, 925,
    950, 975, 1000,
]

LEVEL_TO_LEV = {
    level: f"lev{i:02d}"
    for i, level in enumerate(PRESSURE_LEVELS)
}