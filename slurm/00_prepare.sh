#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Change-point analysis of every target. Needs data/raw filled by
# `python -m src.fetch` on the login node.
set -e
source slurm/env.sh
for d in ecuador brazil europe; do
  python -m src.changepoints --dataset "$d"
done
