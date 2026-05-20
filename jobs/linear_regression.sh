#!/bin/bash
#SBATCH --job-name=pc_era5_linear_regression
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/regression_%j.out
#SBATCH --error=logs/regression_%j.err

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

source /home/student/s/sascholle/miniconda3/etc/profile.d/conda.sh
conda activate graphcast312

# Run your script
srun python -u src/sabines_correlation_experiments/pc_era5_linear_regression_analysis.py

echo "Finished at: $(date)"