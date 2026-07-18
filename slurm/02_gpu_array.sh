#!/bin/bash
#SBATCH --job-name=gpu_models
#SBATCH --partition=<gpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# Set <gpu-partition> to a GPU partition on your cluster.
# Keras recurrent models (GRU/LSTM/BiLSTM): one GPU per task. Use %2 (or the
# concurrency your allocation allows) to cap simultaneous GPU tasks, e.g.
#   sbatch --array=0-N%2 slurm/02_gpu_array.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate energyq1
export PYTHONUNBUFFERED=1
echo "assigned GPU: $CUDA_VISIBLE_DEVICES"
cd "$SLURM_SUBMIT_DIR"
WORK=${ENERGY_WORK:-$SLURM_SUBMIT_DIR/work}
python -m src.run_experiment --jobs_csv "$WORK/jobs_gpu.csv" --task_id "$SLURM_ARRAY_TASK_ID"
