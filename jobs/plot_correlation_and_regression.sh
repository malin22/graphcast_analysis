#!/bin/bash
#SBATCH --job-name=histogram_plotting
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/histo_plot%j.out
#SBATCH --error=logs/histo_plot%j.err

set -euo pipefail

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

source /home/student/s/sascholle/miniconda3/etc/profile.d/conda.sh
conda activate graphcast312

srun python -u /home/student/s/sascholle/share/graphcast_analysis/src/plotting_script_for_correlation_and_regression.py \
  --correlation-json plots/sabines_experiments/mapping_experiments/test_withlatlontime/pc_era5_mesh_m6_screening_cache.json \
  --out-dir plots/sabines_experiments/mapping_experiments/histograms \
  --pcs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --per-pc \
  --per-pc-layout grid \
  --aggregation mean \
  --regression-json plots/sabines_experiments/mapping_experiments/test_withlatlontime/pc_era5_mesh_m6_allvars_linear_results.json \


echo "Finished at: $(date)"

# Use args: 
# pcs: if you want to select a specific set 
# per-pc: if you want the plot to show overall correlation between all pcs or independently for each pc
# per-pc-layout can take grid, separate and both
# aggregation can take mean, max and sum