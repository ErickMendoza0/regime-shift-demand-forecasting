"""Write the job lists that the SLURM arrays work through.

    python -m src.jobs

work/jobs/tune_{cpu,gpu}.csv   one row per (dataset, model, tuning origin)
work/jobs/run_{cpu,gpu,fm}.csv one row per (dataset, model, tier, origin, seed)
"""
import pandas as pd

import config as C
from src import data
from src.spaces import SPACES


def origins(dataset: str) -> list[pd.Timestamp]:
    panel, static = data.load(dataset)
    truth = data.actuals(panel, static)
    last_common = truth.groupby("target")["date"].max().min()
    out = []
    for start, end in C.DATASETS[dataset]["origins"]:
        end = pd.Timestamp(end) if end else last_common
        out += list(pd.date_range(start, end, freq="MS"))
    return out


def tuning_origins(dataset: str) -> list[pd.Timestamp]:
    cfg = C.DATASETS[dataset]
    if cfg["retune"] == "block":
        return [pd.Timestamp(s) for s, _ in cfg["origins"]]
    years = sorted({o.year for o in origins(dataset)})
    return [pd.Timestamp(year=y, month=1, day=1) for y in years]


def _variants(dataset, model, spec):
    tiers = spec.get("tier")
    if tiers and not C.DATASETS[dataset]["exogenous"]:
        return []
    return tiers or [None]


def main() -> None:
    C.ensure_dirs()
    tune, run = {"cpu": [], "gpu": []}, {"cpu": [], "gpu": [], "fm": []}
    for dataset, cfg in C.DATASETS.items():
        ors = origins(dataset)
        for model, spec in C.MODELS.items():
            tiers = _variants(dataset, model, spec)
            if not tiers:
                continue
            if model in SPACES:
                for at in tuning_origins(dataset):
                    tune[spec["kind"]].append({"dataset": dataset, "model": model,
                                               "at": f"{at:%Y-%m}"})
            seeds = [0] if model in C.DETERMINISTIC else cfg["seeds"]
            for tier in tiers:
                for o in ors:
                    for seed in seeds:
                        run[spec["kind"]].append({"dataset": dataset, "model": model,
                                                  "tier": tier or "", "origin": f"{o:%Y-%m}",
                                                  "seed": seed})
    for kind, rows in tune.items():
        pd.DataFrame(rows).to_csv(C.JOBS / f"tune_{kind}.csv", index=False)
        print(f"tune_{kind}: {len(rows)} studies")
    for kind, rows in run.items():
        pd.DataFrame(rows).to_csv(C.JOBS / f"run_{kind}.csv", index=False)
        print(f"run_{kind}: {len(rows)} forecasts")


if __name__ == "__main__":
    main()
