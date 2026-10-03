#!/bin/bash
#SBATCH --job-name=retune
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Redo the studies and forecasts of the models given in MODELS, discarding what
# exists. Used when a search space had to be fixed after the main campaign:
#   sbatch --export=ALL,MODELS="nbeats" slurm/12_retune.sh
set -e
source slurm/env.sh
for m in $MODELS; do
  rm -f work/tuning/*/"$m"/*.json work/tuning/*/"$m"/*.trials.csv
  rm -rf work/preds/*/"$m"
done
WORKERS=4
for w in $(seq 0 $(( WORKERS - 1 ))); do
  SLURM_CPUS_PER_TASK=2 python -m src.tune --jobs work/jobs/tune_gpu.csv --models $MODELS \
    --part "$w" --parts "$WORKERS" > "logs/retune_${SLURM_JOB_ID}_part${w}.out" 2>&1 &
done
wait
