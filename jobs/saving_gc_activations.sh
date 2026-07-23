#!/bin/bash
#SBATCH --job-name=saving_weather_data
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --output=logs/saving_weather_data%j.out
#SBATCH --error=logs/saving_weather_data%j.err

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

# source /home/student/m/mbraatz/miniconda/etc/profile.d/conda.sh
# conda activate graphcast

source /home/student/s/sascholle/miniconda3/etc/profile.d/conda.sh
conda activate graphcast312

# Run your script
srun python -u src/graphcast_setup.py

echo "Finished at: $(date)"