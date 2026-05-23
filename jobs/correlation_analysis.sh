#!/bin/bash
#SBATCH --job-name=course_era5
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/course_era5_%j.out
#SBATCH --error=logs/course_era5_%j.err

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

source /home/student/s/sascholle/miniconda3/etc/profile.d/conda.sh
conda activate graphcast312

# Run your script
#srun python -u src/sabines_correlation_experiments/pc_era5_correlation_analysis.py
srun python -u src/sabines_correlation_experiments/put_era5_on_courser_grid.py

echo "Finished at: $(date)"