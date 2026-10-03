#!/bin/bash
# Create the three conda environments. Run on a machine with internet access
# (on our cluster, the login node). Re-running skips environments that exist.
#
#   bash setup_env.sh          # loose requirements
#   LOCK=1 bash setup_env.sh   # exact versions from envs/*.lock.txt
set -e
source ~/miniconda3/etc/profile.d/conda.sh
env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

make_env() {
  local name=$1 reqs=$2
  if env_exists "$name"; then
    echo "$name already exists, skipping"
    return
  fi
  conda create -y -n "$name" python=3.11
  conda activate "$name"
  python -m pip install --upgrade pip
  if [ -n "$LOCK" ]; then
    pip install -r "envs/$name.lock.txt"
  else
    pip install -r "$reqs"
  fi
  conda deactivate
}

make_env regime2 requirements.txt
make_env regime2_fm requirements-fm.txt
make_env regime2_moirai requirements-moirai.txt
echo "Next: python -m src.fetch --weights (in regime2_fm), then bash slurm/submit.sh"
