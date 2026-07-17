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
  --correlation-json plots/sabines_experiments/mapping_experiments/correlation_regression_json_results_depreciated/pc_era5_mesh_m5_screening_cache.json \
  --out-dir plots/sabines_experiments/mapping_experiments/histograms \
  --pcs 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \
  --per-pc \
  --per-pc-layout grid \
  --aggregation mean #use if wanting an aggregation of all PCs in the json files
  
  #--regression-json plots/sabines_experiments/all_variable_regression_results.json \

echo "Finished at: $(date)"

