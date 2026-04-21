#!/bin/bash
#SBATCH --job-name=pca
#SBATCH --output=logs/pca_%j.out
#SBATCH --error=logs/pca_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00

mkdir -p logs

echo "Running on node: $(hostname)"
echo "Start time: $(date)"

cd "$SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"

source ~/.bashrc
conda activate graphcast312
srun python /home/student/s/sascholle/share/graphcast_analysis/src/pca_script.py

echo "Finished at: $(date)"