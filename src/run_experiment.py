"""Experiment dispatcher for a single (model, fold) job.

Direct:      python -m src.run_experiment --model lstm --fold 2023
Array mode:  python -m src.run_experiment --jobs_csv work/jobs_cpu.csv --task_id $SLURM_ARRAY_TASK_ID
"""
import argparse
import importlib

import pandas as pd

import config as C


def run_one(model: str, fold: int, exog_mode: str) -> None:
    base = model.replace("_persist", "").replace("_observed", "")
    module_name, _ = C.MODELS[base]
    mod = importlib.import_module(f"src.{module_name}")
    print(f"running {model} | fold {fold} | exog_mode={exog_mode}", flush=True)
    try:
        mod.run(base, fold, exog_mode=exog_mode)
    except Exception:
        import traceback
        print(f"failed: {model} fold {fold} ({exog_mode})", flush=True)
        traceback.print_exc()
        raise   # non-zero exit so SLURM marks the task as failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--exog_mode", default="persist", choices=C.EXOG_MODES)
    ap.add_argument("--jobs_csv")
    ap.add_argument("--task_id", type=int)
    a = ap.parse_args()

    if a.jobs_csv is not None:
        jobs = pd.read_csv(a.jobs_csv)
        row = jobs.iloc[a.task_id]
        run_one(row["model"], int(row["fold"]), row.get("exog_mode", "persist"))
    else:
        run_one(a.model, a.fold, a.exog_mode)


if __name__ == "__main__":
    main()
