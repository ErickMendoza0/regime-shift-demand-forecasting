"""Hyper-parameter tuning with Optuna, the same budget for every tuned model.

A study belongs to a (dataset, model, tuning origin). It scores each trial by
mean MASE on two validation origins twelve and six months before the tuning
origin, counting only target months before it. The winning parameters are used
for every forecast origin up to the next tuning origin.

    python -m src.tune --dataset ecuador --model lstm --at 2021-01
    python -m src.tune --jobs work/jobs/tune_gpu.csv --task 3
    python -m src.tune --jobs work/jobs/tune_gpu.csv --part 2 --parts 16
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

import config as C
from src import data, forecast
from src import oni as O
from src.models import Context
from src.spaces import DEFAULTS, SPACES


def tuning_origin(dataset: str, origin) -> pd.Timestamp:
    """Tuning origin whose parameters apply to a forecast origin."""
    origin = pd.Timestamp(origin)
    cfg = C.DATASETS[dataset]
    if cfg["retune"] == "yearly":
        return pd.Timestamp(year=origin.year, month=1, day=1)
    starts = [pd.Timestamp(s) for s, _ in cfg["origins"] if pd.Timestamp(s) <= origin]
    return max(starts)


def path(dataset, model, at) -> Path:
    return C.TUNING / dataset / model / f"{pd.Timestamp(at):%Y-%m}.json"


def params(dataset, model, origin) -> dict:
    if model not in SPACES:
        return {}
    f = path(dataset, model, tuning_origin(dataset, origin))
    if not f.exists():
        raise FileNotFoundError(f"no tuned parameters at {f}; run src.tune first")
    return json.loads(f.read_text())["params"]


def study(dataset, model, at, trials=C.TUNING_TRIALS) -> dict:
    # Imported here so the runner can read tuned parameters in environments
    # without Optuna (the foundation-model ones).
    import optuna

    at = pd.Timestamp(at)
    panel, static = data.load(dataset)
    truth = data.actuals(panel, static)
    oni = O.load() if C.DATASETS[dataset]["exogenous"] else None
    tier = "x1" if model.endswith("_oni") else None
    val_origins = [at - pd.DateOffset(months=12), at - pd.DateOffset(months=6)]

    def objective(trial):
        p = SPACES[model](trial)
        scores = []
        for vo in val_origins:
            ctx = Context(dataset, vo, seed=0, params=p, tier=tier, oni=oni)
            fc = forecast.targets(dataset, model, ctx, panel, static)
            scores.append(forecast.score(fc, truth, before=at))
        return sum(scores) / len(scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    st.enqueue_trial(DEFAULTS[model])
    t0 = time.time()
    st.optimize(objective, n_trials=trials, catch=(RuntimeError, ValueError))
    best = {"params": st.best_params, "value": st.best_value,
            "default_value": st.trials[0].value, "trials": len(st.trials),
            "seconds": round(time.time() - t0, 1)}
    out = path(dataset, model, at)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2))
    st.trials_dataframe().to_csv(out.with_suffix(".trials.csv"), index=False)
    print(f"{dataset} {model} {at:%Y-%m}: {best['value']:.4f} "
          f"(default {best['default_value']:.4f}) in {best['seconds']} s")
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--model")
    ap.add_argument("--at")
    ap.add_argument("--jobs")
    ap.add_argument("--task", type=int)
    ap.add_argument("--part", type=int)
    ap.add_argument("--parts", type=int)
    ap.add_argument("--trials", type=int, default=C.TUNING_TRIALS)
    a = ap.parse_args()
    if a.jobs:
        jobs = pd.read_csv(a.jobs)
        if a.task is not None:
            jobs = jobs.iloc[[a.task]]
        else:
            jobs = jobs[jobs.index % a.parts == a.part]
        todo = list(jobs[["dataset", "model", "at"]].itertuples(index=False))
    else:
        todo = [(a.dataset, a.model, a.at)]
    for dataset, model, at in todo:
        if path(dataset, model, at).exists():
            print(f"{dataset} {model} {at} already tuned, skipping")
            continue
        study(dataset, model, at, a.trials)


if __name__ == "__main__":
    main()
