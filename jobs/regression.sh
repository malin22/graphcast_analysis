#!/bin/bash

#SBATCH --job-name=weather_regression
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --error=logs/error.o%j
#SBATCH --output=logs/output.o%j

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

cd "$SLURM_SUBMIT_DIR"

echo "Working directory: $(pwd)"

source /home/student/m/mbraatz/miniconda/etc/profile.d/conda.sh
conda activate graphcast

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Surface variables
python -m malins_pca_experiments.run_surface_regression

# Pressure-level variables
python -m malins_pca_experiments.run_multilevel_regression \
    --variable temperature

python -m malins_pca_experiments.run_multilevel_regression \
    --variable u_component_of_wind

python -m malins_pca_experiments.run_multilevel_regression \
    --variable v_component_of_wind

python -m malins_pca_experiments.run_multilevel_regression \
    --variable geopotential

python -m malins_pca_experiments.run_multilevel_regression \
    --variable specific_humidity

python -m malins_pca_experiments.run_multilevel_regression \
    --variable vertical_velocity

echo "Finished at: $(date)"