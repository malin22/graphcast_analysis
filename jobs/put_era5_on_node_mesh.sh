#!/bin/bash
#SBATCH --job-name=era5_to_mesh
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/era5_to_mesh_m5%j.out
#SBATCH --error=logs/era5_to_mesh_m5%j.err

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

source /home/student/s/sascholle/miniconda3/etc/profile.d/conda.sh
conda activate graphcast312

# Run your script
srun python -u src/sabines_correlation_experiments/put_era5_on_node_mesh.py


echo "Finished at: $(date)"