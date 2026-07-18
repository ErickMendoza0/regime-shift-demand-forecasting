#!/bin/bash
#SBATCH --job-name=cpu_models
#SBATCH --partition=<cpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Set <cpu-partition> to a CPU partition on your cluster.
# Statistical and ML models (no GPU): one (model, fold) combination per task.
# build_jobs prints the array range, e.g. sbatch --array=0-N slurm/01_cpu_array.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate energyq1
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
WORK=${ENERGY_WORK:-$SLURM_SUBMIT_DIR/work}
python -m src.run_experiment --jobs_csv "$WORK/jobs_cpu.csv" --task_id "$SLURM_ARRAY_TASK_ID"
