"""Run forecast jobs from a job list, skipping any whose output already exists.

    python -m src.run --dataset ecuador --model snaive --origin 2024-01
    python -m src.run --jobs work/jobs/run_gpu.csv --part 3 --parts 16

With --part/--parts a process takes every row whose index is congruent to
`part` modulo `parts`, so several processes can share one job list.
"""
import argparse
import time

import pandas as pd

import config as C
from src import data, forecast, tune
from src import oni as O
from src.models import Context

_cache = {}


def _inputs(dataset):
    if dataset not in _cache:
        panel, static = data.load(dataset)
        oni = O.load() if C.DATASETS[dataset]["exogenous"] else None
        _cache[dataset] = (panel, static, oni)
    return _cache[dataset]


def output(dataset, model, tier, origin, seed):
    tag = f"{model}_{tier}" if tier else model
    return C.PREDS / dataset / tag / f"{pd.Timestamp(origin):%Y-%m}_s{seed}.parquet"


def one(dataset, model, tier, origin, seed) -> None:
    out = output(dataset, model, tier, origin, seed)
    if out.exists():
        return
    panel, static, oni = _inputs(dataset)
    params = tune.params(dataset, model, origin)
    t0 = time.time()
    for attempt in range(3):
        ctx = Context(dataset, pd.Timestamp(origin), seed=int(seed), params=params,
                      tier=tier or None, oni=oni)
        try:
            fc = forecast.targets(dataset, model, ctx, panel, static)
            break
        except Exception as e:
            # GPU memory is not isolated between jobs on our cluster; when another
            # job fills the card, retry with a smaller batch instead of failing.
            if "out of memory" not in str(e).lower() or "batch_size" not in params:
                raise
            import torch
            torch.cuda.empty_cache()
            params = {**params, "batch_size": max(8, params["batch_size"] // 2)}
            print(f"  out of memory, retrying with batch_size={params['batch_size']}")
    else:
        raise RuntimeError(f"{model} kept running out of memory at {origin}")
    fc["model"] = f"{model}_{tier}" if tier else model
    fc["seed"] = int(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out, index=False)
    print(f"{dataset} {fc['model'].iloc[0]} {origin} s{seed}: {time.time() - t0:.1f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--models", nargs="*", help="only run these models from the job list")
    ap.add_argument("--dataset")
    ap.add_argument("--model")
    ap.add_argument("--tier", default="")
    ap.add_argument("--origin")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.jobs:
        one(a.dataset, a.model, a.tier, a.origin, a.seed)
        return
    jobs = pd.read_csv(a.jobs, keep_default_na=False)
    if a.models:
        jobs = jobs[jobs["model"].isin(a.models)].reset_index(drop=True)
    mine = jobs[jobs.index % a.parts == a.part]
    print(f"{len(mine)} of {len(jobs)} jobs in {a.jobs} (part {a.part}/{a.parts})", flush=True)
    failed = 0
    for row in mine.itertuples():
        try:
            one(row.dataset, row.model, row.tier, row.origin, row.seed)
        except Exception as e:
            failed += 1
            print(f"FAILED {row.dataset} {row.model} {row.tier} {row.origin} s{row.seed}: "
                  f"{type(e).__name__}: {e}", flush=True)
    if failed:
        raise SystemExit(f"{failed} jobs failed")


if __name__ == "__main__":
    main()
