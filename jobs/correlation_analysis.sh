#!/bin/bash
#SBATCH --job-name=regression_mesh
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/regression_mesh_m5%j.out
#SBATCH --error=logs/regression_mesh_m5%j.err

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

source /home/student/s/sascholle/miniconda3/etc/profile.d/conda.sh
conda activate graphcast312

# Run your script
#srun python -u src/sabines_correlation_experiments/pc_era5_correlation_analysis_single_variable.py --mesh-level 6
srun python -u src/sabines_correlation_experiments/pc_era5_regression_analysis_all_variables.py --mesh-level 5 --model-type 'ElasticNet' --alpha-grid 0.0001 0.0003 0.001 0.003 0.01
#srun python -u src/sabines_correlation_experiments/put_era5_on_courser_grid.py
#srun python -u src/sabines_correlation_experiments/put_era5_on_node_mesh.py

# tar -czf /share/prj-4d/graphcast_shared/data/5_node_activations_for_tensor_decomposition.tar.gz \
#   -C /share/prj-4d/graphcast_shared/data m5_node_activations_for_tensor_decomposition


echo "Finished at: $(date)"