"""Diagnostic for the neuralforecast (SOTA) stack. Run it on a GPU node to
check that torch, Lightning and neuralforecast import correctly and that a
minimal N-BEATS fit/predict cycle works before launching the full jobs:

    srun --pty --partition=<gpu-partition> --gres=gpu:1 --cpus-per-task=4 \\
         --mem=16G --time=00:15:00 bash -i
    conda activate energyq1_torch
    python -m src.diag_sota
"""
import sys


def main() -> None:
    print("Python:", sys.version.split()[0])
    for mod in ("torch", "pytorch_lightning", "neuralforecast"):
        try:
            m = __import__(mod)
            print(f"  {mod} {getattr(m, '__version__', '?')}")
        except Exception as e:
            print(f"  could not import {mod}: {type(e).__name__}: {e}")
    try:
        import torch
        print("cuda available:", torch.cuda.is_available(),
              "| device:", torch.cuda.get_device_name(0)
              if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print("torch cuda check failed:", e)

    # Minimal end-to-end fit on synthetic data.
    try:
        import numpy as np
        import pandas as pd
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS
        from src.models_sota import _trainer_kwargs
        ds = pd.date_range("2014-01-01", periods=120, freq="MS")
        df = pd.concat([
            pd.DataFrame({"unique_id": f"s{i}", "ds": ds,
                          "y": 100 + 10 * np.sin(np.arange(120)) + i})
            for i in range(3)])
        nf = NeuralForecast(models=[NBEATS(h=12, input_size=24, max_steps=20,
                                           **_trainer_kwargs())], freq="MS")
        nf.fit(df)
        out = nf.predict()
        print("minimal NBEATS fit/predict ok; predict cols:", list(out.columns))
    except Exception as e:
        import traceback
        print("minimal fit failed:", type(e).__name__, e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
