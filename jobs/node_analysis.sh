#!/bin/bash
#SBATCH --job-name=graphcast_analysis
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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
srun python -u src/node_analysis.py

echo "Finished at: $(date)"