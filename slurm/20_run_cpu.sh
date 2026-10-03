#!/bin/bash
#SBATCH --job-name=run_cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Statistical, Prophet and LightGBM forecasts, split into N parts:
#   sbatch --array=0-63%24 slurm/20_run_cpu.sh
set -e
source slurm/env.sh
python -m src.run --jobs work/jobs/run_cpu.csv --part "$SLURM_ARRAY_TASK_ID" \
  --parts "$SLURM_ARRAY_TASK_COUNT"
