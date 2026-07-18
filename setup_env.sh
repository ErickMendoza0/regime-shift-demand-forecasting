#!/bin/bash
# Create the two conda environments used by the pipeline.
#
#   energyq1        TensorFlow/Keras + statistics + ML (everything except SOTA)
#   energyq1_torch  PyTorch + neuralforecast (N-BEATS / PatchTST)
#
# The two environments are kept separate because TensorFlow and PyTorch ship
# their own copies of the CUDA userspace libraries, and keeping them apart
# avoids version conflicts between the two stacks.
#
#   bash setup_env.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh

# PyTorch wheel index for CUDA 12.x drivers.
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

# Re-running only creates the environments that are missing.
env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

echo "[1/2] energyq1 (TensorFlow/Keras + stats + ML), Python 3.10"
if env_exists energyq1; then
  echo "energyq1 already exists; skipping (remove with 'conda env remove -n energyq1' to rebuild)"
else
  conda create -y -n energyq1 python=3.10
  conda activate energyq1
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  python - <<'EOF'
import tensorflow as tf, pmdarima, lightgbm, prophet
print("tf", tf.__version__, "| pmdarima", pmdarima.__version__,
      "| lgbm", lightgbm.__version__)
EOF
  conda deactivate
fi

echo "[2/2] energyq1_torch (PyTorch + neuralforecast), Python 3.10"
if env_exists energyq1_torch; then
  echo "energyq1_torch already exists; skipping"
else
  conda create -y -n energyq1_torch python=3.10
  conda activate energyq1_torch
  python -m pip install --upgrade pip
  # Install torch from the CUDA 12.x wheel index first, then neuralforecast on top.
  pip install torch --index-url "${TORCH_INDEX}"
  pip install -r requirements_sota.txt
  python - <<'EOF'
import torch, neuralforecast
print("torch", torch.__version__, "| neuralforecast", neuralforecast.__version__,
      "| cuda:", torch.cuda.is_available())
EOF
  conda deactivate
fi

echo "Done."
echo "Next: upload the raw CSV to data/raw/ and run (env energyq1): python -m src.fetch_oni"
echo "CPU and Keras GPU models use 'energyq1'; the SOTA models use 'energyq1_torch'."
