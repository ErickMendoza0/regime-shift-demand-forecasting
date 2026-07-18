#!/bin/bash
#SBATCH --job-name=prepare
#SBATCH --partition=<cpu-partition>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Set <cpu-partition> to a CPU partition on your cluster.
# Preprocess the raw CSV into the sub-series and national parquet files and
# build the job lists. Requires data/raw/Datos_Energeticos_Ecuador_2014_2024.csv
# and data/raw/oni.csv (run `python -m src.fetch_oni` beforehand from a machine
# with internet access).
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate energyq1
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
python -m src.prepare_data
python -m src.build_jobs
