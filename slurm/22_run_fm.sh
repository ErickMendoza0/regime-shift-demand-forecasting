#!/bin/bash
#SBATCH --job-name=run_fm
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Foundation models, zero-shot. Chronos-2 and TimesFM share one environment;
# Moirai needs its own because uni2ts pins an older PyTorch.
#   sbatch --array=0-1 slurm/22_run_fm.sh
set -e
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  REGIME_ENV=regime2_fm; MODELS="chronos timesfm"
else
  REGIME_ENV=regime2_moirai; MODELS="moirai"
fi
source slurm/env.sh
python -m src.run --jobs work/jobs/run_fm.csv --models $MODELS
