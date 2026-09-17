# GraphCast Latent Analysis

This repository contains the code used to analyze GraphCast latent activations and to study their structure with PCA and related diagnostics.

The project is organized as a research codebase rather than a packaged library. Most of the logic is in `src/`, and the `jobs/` folder contains shell scripts for running the analysis in a batch environment.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── jobs/
│   ├── correlation_analysis.sh
│   ├── layer_activation_analysis.sh
│   ├── node_analysis.sh
│   ├── pca_script.sh
│   ├── plot_correlation_and_regression.sh
│   ├── put_era5_on_node_mesh.sh
│   ├── regression.sh
│   ├── save_data.sh
│   ├── saving_gc_activations.sh
│   └── temporal_analysis.sh
├── plots/
│   └── generated figures and summary plots
├── src/
│   ├── helper/
│   │   ├── check_activation_times.py
│   │   ├── find_nans_missing_dates.py
│   │   └── node_analysis.py
│   ├── preprocessing/
│   │   ├── activation_preprocessing.py
│   │   ├── climatenet_preprocessing.py
│   │   ├── mesh_context.py
│   │   └── save_pc_scores_transform.py
│   ├── regression/
│   │   ├── atmospheric_variables/
│   │   └── extreme_weather_events/
│   ├── set_up/
│   │   ├── data_setup.py
│   │   ├── graphcast_setup.py
│   │   ├── pca_script.py
│   │   └── save_pc_scores_per_timestep.py
│   ├── perturbation/
│   ├── sabines_layer_experiments/
│   ├── sabines_mapping_experiments/
│   ├── sabines_temporal_pattern_experiments/
│   ├── depricated/
│   └── am_I_depricated_?/
└── ...
```

## Main folders

### `src/set_up/`
Contains the main setup scripts:
- `data_setup.py`: prepares ERA5 data for downstream analyses.
- `graphcast_setup.py`: configures GraphCast and saves hidden activations.
- `pca_script.py`: applies PCA to the activation data.
- `save_pc_scores_per_timestep.py`: saves projected PCA scores over time.

### `src/preprocessing/`
Contains scripts for preparing and transforming activation data and mesh-related information.

### `src/helper/`
Contains utility scripts for checking activation files, data quality, and node-level analysis.

### `src/regression/`
Contains scripts for regression-based analyses, including variable-specific and extreme-event workflows.

### `jobs/`
Contains shell scripts that run specific tasks such as PCA, correlation analysis, regression, and plotting.

### `plots/`
Stores output figures and visual diagnostics.

## Typical workflow

1. Prepare ERA5 files
2. Run GraphCast and save activations
3. Run PCA on those activations
4. Project the activations onto principal components
5. Plot results and run correlations/regressions

Example commands:

```bash
python src/set_up/data_setup.py
python src/set_up/graphcast_setup.py
python src/set_up/pca_script.py
```

Or use the batch scripts in `jobs/`:

```bash
bash jobs/pca_script.sh
bash jobs/correlation_analysis.sh
bash jobs/regression.sh
```

## Notes

- This repo is built around a research workflow and not a polished library.
- Some folders still contain older or experimental code.
- Several scripts use project-specific paths (for example under `/share/prj-4d/...`), so local paths may need to be adjusted.

## Dependencies

The repository depends on the Python packages listed in `requirements.txt`, including scientific Python libraries and the GraphCast-specific package.

If you want a very short version, this repository can be summarized as:

- data setup,
- GraphCast activation extraction,
- PCA and latent analysis,
- plotting and downstream statistical analysis.
