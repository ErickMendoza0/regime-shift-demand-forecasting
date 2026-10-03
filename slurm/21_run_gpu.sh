#!/bin/bash
#SBATCH --job-name=run_gpu
#SBATCH --partition=medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Recurrent and neuralforecast models, WORKERS processes per GPU:
#   sbatch --array=0-3 slurm/21_run_gpu.sh
set -e
source slurm/env.sh
WORKERS=${WORKERS:-4}
PARTS=$(( ${SLURM_ARRAY_TASK_COUNT:-1} * WORKERS ))
for w in $(seq 0 $(( WORKERS - 1 ))); do
  part=$(( SLURM_ARRAY_TASK_ID * WORKERS + w ))
  SLURM_CPUS_PER_TASK=2 python -m src.run --jobs work/jobs/run_gpu.csv \
    --part "$part" --parts "$PARTS" > "logs/run_gpu_${SLURM_ARRAY_JOB_ID}_part${part}.out" 2>&1 &
done
wait
