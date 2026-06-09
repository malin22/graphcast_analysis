#!/bin/bash
#SBATCH --job-name=saving_weather_data
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --error=logs/error.o%j
#SBATCH --output=logs/output.o%j

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

source /home/student/m/mbraatz/miniconda/etc/profile.d/conda.sh
conda activate graphcast

# Run your script
srun python -u src/malins_pca_experiments/regression.py

echo "Finished at: $(date)"