# Sourced by every job script. Override REGIME_ENV to use another environment.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${REGIME_ENV:-regime2}"
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
# Several jobs can land on the same GPU, so let PyTorch return memory it no
# longer needs instead of holding on to fragmented blocks.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Compute nodes have no internet: models must already be in the local cache.
export HF_HOME="${HF_HOME_LOCAL:-$HOME/hf_cache}" HF_HUB_OFFLINE=1
export TMPDIR=/tmp
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
# Queue limits on our cluster: medium allows the whole group 112 CPUs and 4 GPUs,
# short has no group limit but a 4-hour wall clock, so the CPU stages are split
# into parts small enough for short.
