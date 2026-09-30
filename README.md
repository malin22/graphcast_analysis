## Towards a Mechanistic Understanding of GraphCast Through Latent Analysis ##

In this repo we provide the code for investigating the internal latent activations of GraphCast using principal component analysis (PCA) applied to the processor-layer embeddings across multi-year global forecasts. By projecting latent activations onto principal component directions, we can construct an unsupervised basis capturing the dominant directions of variation in the latent representation. We use this basis to test how global atmospheric information is organised within the representation and how much of it is retained in progressively lower-dimensional subspaces. We then extend this analysis to extreme-weather phenomena, testing whether such events can be identified from compact subsets of the latent representation of GraphCast. Finally, we move beyond association and decodability by intervening directly on selected latent directions and evaluating whether these interventions produce systematic changes in forecasts of atmospheric rivers (ARs) and tropical cyclones (TCs).


### Setup and Installation

Install the required dependencies from `requirements.txt`, for instance, with conda:

```bash
conda create -n graphcast-analysis python=3.12
conda activate graphcast-analysis
pip install -r requirements.txt
```

The perturbation experiments require access to and intervention on internal processor representations. For this purpose, the GraphCast implementation was extended with additional hooks for latent activation manipulation.

The modifications are based on the GraphCast fork of [MacMillan et al.'s GraphCast interpretability work](https://github.com/theodoremacmillan/graphcast-interpretability), with further changes added for the perturbation experiments in this repository.

The `requirements.txt` contains the modified GraphCast fork.


### Repository Structure
```text
graphcast_analysis/
├── README.md
├── requirements.txt
│
├── jobs/                              # HPC job scripts
│   ├── saving_gc_activations.sh       # Save GraphCast latent activations
│   └── ...
│
├── src/
│   ├── set_up/                        # GraphCast and environment setup
│   ├── preprocessing/                 # ERA5 and activation preprocessing
│   ├── layers/                        # Layer activation and PCA analyses
│   ├── correlation/                   # PC/ERA5 correlation and plotting
│   ├── regression/                    # Atmospheric and extreme-weather regression
│   ├── temporal_pattern_experiments/  # Temporal structure analyses
│   ├── perturbation/                  # Latent-direction interventions
│   ├── helper/                        # Shared helper utilities
│   └── deprecated/                    # Older or unused scripts
│
├── plots/                             # Generated figures
│   ├── regression/
│   └── ...
│
├── results/                           # Generated numerical results (not pushed!)
│   └── ...
│
```

### Data Directory
The data/ directory is not included in the repository. It is shown here to document the expected local data layout used by the analysis scripts.

It contains GraphCast latent activations, ERA5 and ocean data, PCA outputs, event labels, and derived datasets used by the analysis scripts.

```text
data/                                      # Local/shared analysis data
├── ClimateNetLarge/                       # Extreme-weather event labels
│   ├── AR_labels_cleaned/                 # Atmospheric river labels
│   ├── blocking_labels_cleaned/           # Atmospheric blocking labels
│   └── TC_labels_cleaned/                 # Tropical cyclone labels
│
├── era5_daily_mesh/                        # ERA5 mapped onto GraphCast mesh nodes
├── era5_daily_nc/                          # Original or processed ERA5 NetCDF data
├── glorys/                                 # GLORYS ocean data
│
├── graphcast_activation_2019/              # Activations from layer-8 for 2019
├── graphcast_activation_2020/              # Activations from layer-8 for 2020
├── graphcast_activation_2021/              # Activations from layer-8 for 2021
├── graphcast_activations_all_layers_2019/  # Activations from all model layers
│
├── m5_node_activations_for_tensor_decomposition/
│                                            # M5 node activations for tensor analysis
├── pc_scores_per_timestep/                  # pre-computed PC projections for each timestep
├── pca_components/                          # PCA component matrices and means
├── tensor_decomposition_subsets/            # Data subsets for tensor decomposition
│
└── deprecated_data/                         # Older or superseded datasets
    └── ...

```
