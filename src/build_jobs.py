"""Enumerate every (model, fold[, exog_mode]) combination into three job lists
and print the sbatch --array ranges to use.

  work/jobs_cpu.csv   statistical + ML models (no GPU)
  work/jobs_gpu.csv   recurrent Keras models (GPU)
  work/jobs_sota.csv  neuralforecast models (separate PyTorch env)
"""
import pandas as pd

import config as C

EXOG_MODELS = {"sarimax_dis", "lgbm_exog", "lstm_exog"}


def main() -> None:
    rows = []
    for model, (_, gpu) in C.MODELS.items():
        for fold in C.FOLD_YEARS:
            if model in EXOG_MODELS:
                for mode in C.EXOG_MODES:
                    rows.append({"model": model, "fold": fold,
                                 "exog_mode": mode, "gpu": gpu})
            else:
                rows.append({"model": model, "fold": fold,
                             "exog_mode": "persist", "gpu": gpu})
    jobs = pd.DataFrame(rows)
    is_sota = jobs.model.isin(["nbeats", "patchtst"])
    cpu = jobs[~jobs.gpu].drop(columns="gpu").reset_index(drop=True)
    # The torch-based SOTA models run in their own environment, so they get a
    # separate job list rather than sharing the Keras GPU list.
    gpu = jobs[jobs.gpu & ~is_sota].drop(columns="gpu").reset_index(drop=True)
    sota = jobs[is_sota].drop(columns="gpu").reset_index(drop=True)
    C.WORK.mkdir(parents=True, exist_ok=True)
    cpu.to_csv(C.WORK / "jobs_cpu.csv", index=False)
    gpu.to_csv(C.WORK / "jobs_gpu.csv", index=False)
    sota.to_csv(C.WORK / "jobs_sota.csv", index=False)
    print(f"CPU:  {len(cpu)} jobs -> sbatch --array=0-{len(cpu)-1} slurm/01_cpu_array.sh")
    print(f"GPU:  {len(gpu)} jobs -> sbatch --array=0-{len(gpu)-1}%2 slurm/02_gpu_array.sh")
    print(f"SOTA: {len(sota)} jobs -> sbatch --array=0-{len(sota)-1}%2 slurm/04_sota_array.sh")


if __name__ == "__main__":
    main()
