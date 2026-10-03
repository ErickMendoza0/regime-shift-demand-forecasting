#!/bin/bash
#SBATCH --job-name=run_fm
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Foundation models, zero-shot, one model per array task:
#   sbatch --array=0-2 slurm/22_run_fm.sh
# Finished forecasts are skipped, so a task that hits the time limit can simply
# be submitted again. Moirai has its own environment (see requirements-moirai.txt).
set -e
MODELS=(chronos timesfm moirai)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
if [ "$MODEL" = "moirai" ]; then REGIME_ENV=regime2_moirai; else REGIME_ENV=regime2_fm; fi
source slurm/env.sh
python -m src.run --jobs work/jobs/run_fm.csv --models "$MODEL"
