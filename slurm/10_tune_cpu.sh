#!/bin/bash
#SBATCH --job-name=tune_cpu
#SBATCH --partition=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
# One Optuna study per array task (Prophet and LightGBM).
#   sbatch --array=0-$(( $(wc -l < work/jobs/tune_cpu.csv) - 2 ))%8 slurm/10_tune_cpu.sh
set -e
source slurm/env.sh
python -m src.tune --jobs work/jobs/tune_cpu.csv --task "$SLURM_ARRAY_TASK_ID"
