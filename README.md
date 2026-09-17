# Towards a Mechanistic Understanding of GraphCast Through Latent Analysis

This repository contains analysis code for studying the internal latent representations of GraphCast, a machine-learning weather forecasting model. The project focuses on the hidden activations produced by the model's processor layers and asks a simple but important scientific question: do these latent states encode physically interpretable atmospheric structures in a linearly accessible way?

To address this, the codebase extracts GraphCast activations, applies principal component analysis (PCA) to the latent space, and compares the resulting principal directions with ERA5 meteorological fields. The goal is to investigate whether the model's internal geometry reflects known atmospheric patterns such as waves, circulation structures, and other physically meaningful modes.

The repository is organized around a reproducible analysis pipeline: data preparation, GraphCast setup, latent extraction, PCA, projection and correlation analysis, and visualization.

## Why this project matters

Data-driven forecasting models such as GraphCast achieve strong predictive skill, but their internal representations are often difficult to interpret. A central challenge in modern ML-based weather prediction is understanding whether neural network latent states correspond to physically meaningful structures rather than opaque numerical features.

This project adopts a mechanistic interpretation perspective. By analyzing the latent activity of the model at the graph-node level and projecting it onto principal axes, we can test whether a low-dimensional representation captures coherent atmospheric variability. A direct comparison with ERA5 variables allows us to relate latent directions to observable dynamical and thermodynamic phenomena.

## Core idea

GraphCast predictions are produced from latent graph-based processor states. These states evolve over time and across spatial locations. The repository examines whether the hidden state can be decomposed into a small set of dominant latent directions representing organized atmospheric structure.

The analysis proceeds as follows:

1. Load ERA5 fields for relevant meteorological variables.
2. Run or load a GraphCast model configuration.
3. Extract activations from selected processor layers and graph nodes.
4. Assemble activation snapshots into a matrix of node-level features.
5. Apply PCA or incremental PCA to those activations.
6. Project latent states onto principal directions.
7. Correlate principal-component scores with ERA5 fields and other diagnostics.
8. Visualize spatial activation maps and summary statistics.

This workflow turns a high-dimensional latent representation into interpretable, physically grounded patterns.

## Repository layout

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
│   └── output figures and diagnostic visualizations
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
│   ├── sabines_layer_experiments/
│   ├── sabines_mapping_experiments/
│   ├── sabines_temporal_pattern_experiments/
│   ├── set_up/
│   │   ├── data_setup.py
│   │   ├── graphcast_setup.py
│   │   ├── pca_script.py
│   │   └── save_pc_scores_per_timestep.py
│   ├── perturbation/
│   ├── depricated/
│   └── am_I_depricated_?/
└── ...
```

## Main components

### 1. Data preparation

The scripts in `src/set_up/data_setup.py` prepare ERA5 data for analysis. The workflow loads a Zarr-based ERA5 dataset, keeps a subset of relevant variables, and writes one daily NetCDF file per day to a local analysis directory.

This setup is needed because the GraphCast pipeline uses temporally and spatially aligned weather variables, and the project-specific scripts rely on a local directory structure for the downstream activation and PCA workflow.

### 2. GraphCast setup

The file `src/set_up/graphcast_setup.py` sets up the GraphCast model and extracts latent activations from selected processor layers. It loads the model checkpoint from the public GraphCast bucket, configures normalization statistics, and defines the activation manager used to save hidden states.

The relevant activation manager configuration includes:

- `save_steps=[8]`
- `save_node_sets=["mesh_nodes"]`
- `mode="post_res"`

This means the code specifically captures the post-residual processor-layer activations on the mesh node set, which is the representation used for latent-space analysis in the project.

### 3. PCA of latent states

`src/set_up/pca_script.py` implements the core dimensionality reduction step. It:

- loads activation files stored as `.npy` arrays,
- validates their shape and missing-value status,
- fits an `IncrementalPCA` model,
- saves the principal components and mean activation vectors,
- and generates maps of selected principal-component scores.

The PCA acts on latent activations aggregated over graph nodes, making it possible to identify dominant axes of variation in the model's hidden space.

### 4. Projection and physical interpretation

Once a PCA basis is learned, the model can project new activation snapshots onto the principal directions. The resulting component scores can be mapped back to the sphere and compared with ERA5 fields. This is how the project investigates whether the latent space contains interpretable physical modes.

The repository also includes scripts for:

- regression analysis,
- correlation analysis,
- temporal analysis,
- node-level diagnostics,
- plotting summaries and spatial maps.

## Scientific workflow

The overall analysis is designed to answer the following questions:

- Which latent directions dominate GraphCast's hidden state?
- Are those directions associated with recognizable meteorological structure?
- How strongly do principal-component scores correlate with ERA5 variables?
- Do the dominant modes correspond to coherent, physically meaningful patterns in space and time?

These questions align with a mechanistic interpretability framework: rather than asking only whether the model predicts well, we ask what structure is internally represented and whether that structure resembles atmospheric dynamics.

## Installation

This project is based on Python and uses a set of scientific libraries for numerical computing, data processing, and GraphCast integration.

Clone the repository and install dependencies:

```bash
git clone https://github.com/malin22/graphcast_analysis.git
cd graphcast_analysis
pip install -r requirements.txt
```

The dependency list includes:

- NumPy
- Xarray
- NetCDF4
- Zarr
- JAX and related packages
- GraphCast from the project-specific fork
- Utility libraries for analysis and plotting

## Environment assumptions

The repository is configured around a shared scientific computing workflow and contains hard-coded paths such as:

```python
/share/prj-4d/graphcast_shared/data
```

These paths are specific to the project environment and may need to be adapted for local or cloud execution. In particular, the scripts assume:

- a local ERA5 directory,
- a shared storage area for saved activations,
- and an output directory for PCA matrices and plots.

If you are running the code outside the original HPC setup, update the paths in the relevant scripts before execution.

## Example usage

### 1. Prepare ERA5 data

```bash
python src/set_up/data_setup.py
```

This script reads the source ERA5 dataset and writes daily NetCDF files that are compatible with the GraphCast input pipeline.

### 2. Set up GraphCast and save activations

```bash
python src/set_up/graphcast_setup.py
```

This script loads the model checkpoint, constructs the GraphCast predictor, and saves selected latent activations from the processor layers.

### 3. Run PCA

```bash
python src/set_up/pca_script.py
```

This script performs incremental PCA on the saved activations, writes the component matrices and mean vectors to disk, and produces plots of cumulative explained variance and principal-component maps.

### 4. Run downstream analyses

The `jobs/` directory contains ready-to-use shell scripts for more specific analyses such as:

```bash
bash jobs/pca_script.sh
bash jobs/correlation_analysis.sh
bash jobs/regression.sh
bash jobs/temporal_analysis.sh
bash jobs/plot_correlation_and_regression.sh
```

These scripts are useful for running the project in batch mode on a cluster or server environment.

## Outputs

The project generates a set of analysis outputs, including:

- saved latent activation arrays,
- PCA component matrices,
- PCA mean vectors,
- cumulative explained variance plots,
- spatial principal-component maps,
- correlation and regression diagnostics,
- and summary figures in `plots/`.

These outputs are designed to support a mechanistic readout of GraphCast's latent dynamics and to connect them to known atmospheric structures.

## Notes on interpretation

This project is best understood as an exploratory scientific analysis rather than a production-ready software package. The repository includes both active analysis scripts and older or partially deprecated code. Some folders such as `src/depricated/` and `src/am_I_depricated_?/` indicate that the project is still evolving and that certain parts of the codebase are experimental or transitional.

This is typical for research code in atmospheric ML, where the analysis pipeline is modified as hypotheses are tested and new diagnostics are added.

## Contributions and reproducibility

This repository is meant to be reproducible within the project environment. For best results:

- keep the same dataset versions,
- save outputs in a structured folder hierarchy,
- document data paths and model checkpoints,
- and verify that activation files and PCA outputs are aligned in time and layer index.

## Summary

This repository combines machine learning, atmospheric science, and latent-space analysis to study whether GraphCast encodes physically meaningful patterns in its hidden state. By applying PCA to graph-node activations and comparing them with ERA5 variables, the project aims to move beyond black-box accuracy metrics and toward a mechanistic understanding of how the model represents weather dynamics.

The code is deliberately focused on scientific interpretation, and the project is best viewed as a research pipeline for analyzing the geometry of a learned atmospheric model.

---

If you want, I can also generate:

- a shorter project-level README for GitHub front page use,
- a more paper-style methods section,
- or a README tailored for a publication repository with abstract, data availability, and reproducibility sections.
