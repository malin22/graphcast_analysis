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

XXX_PC_COUNTS = [5, 10, 25, 50, 100, 200, 400, 512]

REGRESSION_TYPE = "linear"
SCORE_VALUES = "PCA"

TRAIN_YEARS = [2019, 2020]
TEST_YEARS = [2021]

OUT_DIR = (
    "results/malins_regression/"
    f"{SCORE_VALUES}/{REGRESSION_TYPE}/"
    f"l{NODE_HIERARCHY_LEVEL}_nodes"
)

os.makedirs(OUT_DIR, exist_ok=True)

PC_COUNTS = [5, 512]

TARGETS = [
    {"name": "2t", "var": "2m_temperature", "level": None},
]

XXX_TARGETS = [
    {"name": "2t", "var": "2m_temperature", "level": None},
    {"name": "10u", "var": "10m_u_component_of_wind", "level": None},
    {"name": "10v", "var": "10m_v_component_of_wind", "level": None},
    {"name": "msl", "var": "mean_sea_level_pressure", "level": None},
    {"name": "tp", "var": "total_precipitation_6hr", "level": None},

    {"name": "t50", "var": "temperature", "level": 50},
    {"name": "t250", "var": "temperature", "level": 250},
    {"name": "t500", "var": "temperature", "level": 500},
    {"name": "t600", "var": "temperature", "level": 600},
    {"name": "t700", "var": "temperature", "level": 700},
    {"name": "t850", "var": "temperature", "level": 850},
    {"name": "t1000", "var": "temperature", "level": 1000},

    {"name": "u50", "var": "u_component_of_wind", "level": 50},
    {"name": "u250", "var": "u_component_of_wind", "level": 250},
    {"name": "u500", "var": "u_component_of_wind", "level": 500},
    {"name": "u600", "var": "u_component_of_wind", "level": 600},
    {"name": "u700", "var": "u_component_of_wind", "level": 700},
    {"name": "u850", "var": "u_component_of_wind", "level": 850},
    {"name": "u1000", "var": "u_component_of_wind", "level": 1000},

    {"name": "v50", "var": "v_component_of_wind", "level": 50},
    {"name": "v250", "var": "v_component_of_wind", "level": 250},
    {"name": "v500", "var": "v_component_of_wind", "level": 500},
    {"name": "v600", "var": "v_component_of_wind", "level": 600},
    {"name": "v700", "var": "v_component_of_wind", "level": 700},
    {"name": "v850", "var": "v_component_of_wind", "level": 850},
    {"name": "v1000", "var": "v_component_of_wind", "level": 1000},

    {"name": "z50", "var": "geopotential", "level": 50},
    {"name": "z250", "var": "geopotential", "level": 250},
    {"name": "z500", "var": "geopotential", "level": 500},
    {"name": "z600", "var": "geopotential", "level": 600},
    {"name": "z700", "var": "geopotential", "level": 700},
    {"name": "z850", "var": "geopotential", "level": 850},
    {"name": "z1000", "var": "geopotential", "level": 1000},

    {"name": "q50", "var": "specific_humidity", "level": 50},
    {"name": "q250", "var": "specific_humidity", "level": 250},
    {"name": "q500", "var": "specific_humidity", "level": 500},
    {"name": "q600", "var": "specific_humidity", "level": 600},
    {"name": "q700", "var": "specific_humidity", "level": 700},
    {"name": "q850", "var": "specific_humidity", "level": 850},
    {"name": "q1000", "var": "specific_humidity", "level": 1000},

    {"name": "w50", "var": "vertical_velocity", "level": 50},
    {"name": "w250", "var": "vertical_velocity", "level": 250},
    {"name": "w500", "var": "vertical_velocity", "level": 500},
    {"name": "w600", "var": "vertical_velocity", "level": 600},
    {"name": "w700", "var": "vertical_velocity", "level": 700},
    {"name": "w850", "var": "vertical_velocity", "level": 850},
    {"name": "w1000", "var": "vertical_velocity", "level": 1000},
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