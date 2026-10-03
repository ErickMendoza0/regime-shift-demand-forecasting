#!/bin/bash
#SBATCH --job-name=tune_gpu
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
# The neural studies are small, so each array task takes one GPU and runs
# WORKERS studies on it at once. With 4 tasks: sbatch --array=0-3 slurm/11_tune_gpu.sh
set -e
source slurm/env.sh
WORKERS=${WORKERS:-4}
PARTS=$(( ${SLURM_ARRAY_TASK_COUNT:-1} * WORKERS ))
for w in $(seq 0 $(( WORKERS - 1 ))); do
  part=$(( SLURM_ARRAY_TASK_ID * WORKERS + w ))
  SLURM_CPUS_PER_TASK=4 python -m src.tune --jobs work/jobs/tune_gpu.csv \
    --part "$part" --parts "$PARTS" > "logs/tune_gpu_${SLURM_ARRAY_JOB_ID}_part${part}.out" 2>&1 &
done
wait
