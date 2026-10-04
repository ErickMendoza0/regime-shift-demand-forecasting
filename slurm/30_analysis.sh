#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --partition=medium
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Combinations, scores, tests and figures, once every forecast is in.
set -e
source slurm/env.sh
for d in ecuador brazil europe; do
  python -m src.combine --dataset "$d"
  python -m src.evaluate --dataset "$d"
done
python -m src.pooled
python -m src.figures
python -m src.summary
python -m src.tables
