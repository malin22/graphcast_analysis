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
  --out-dir plots/sabines_experiments/mapping_experiments/histograms/top_512_pcs_histo/histo_with_cutoff \
  --per-pc \
  --per-pc-layout grid \
  --aggregation mean \
  --regression-json plots/sabines_experiments/mapping_experiments/top_512_pcs/regression_pc_era5_mesh_m6_allvars_linear_results.json \
  --regression-cutoff 0.1 \
  #--regression-no-normalize \
  #--correlation-json plots/sabines_experiments/mapping_experiments/top_512_pcs/correlation_pc_era5_mesh_m6_screening_cache.json \
  #--correlation-cutoff 0.3 \
  #--pcs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  #--regression-json plots/sabines_experiments/mapping_experiments/top_512_pcs/regression_pc_era5_mesh_m6_allvars_elasticnet_results.json \

echo "Finished at: $(date)"

# Use args: 
# pcs: if you want to select a specific set 
# per-pc: if you want the plot to show overall correlation between all pcs or independently for each pc
# per-pc-layout can take grid, separate and both
# aggregation can take mean, max and sum