#!/bin/bash
#SBATCH --job-name=layers
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/layers%j.out
#SBATCH --error=logs/layers%j.err

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

source /home/student/s/sascholle/miniconda3/etc/profile.d/conda.sh
conda activate graphcast312

# Run your script
#srun python -u src/sabines_layer_experiments/save_all_layers.py
srun python -u src/sabines_layer_experiments/layer_animation.py
echo "Finished at: $(date)"