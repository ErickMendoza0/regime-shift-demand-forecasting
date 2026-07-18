#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --partition=<cpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Set <cpu-partition> to a CPU partition on your cluster.
# Post-processing: switching ensemble, statistical tests and figures.
# Run once all model arrays (01, 02 and 04) have finished.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate energyq1
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
python -m src.ensemble
python -m src.stats_tests
python -m src.make_figures
