# Regime-shift demand forecasting

Code for a benchmark of monthly electricity forecasters under real regime
shifts. The question is simple: when a series goes through a genuine break, do
more elaborate models hold up better than simple ones, and how much of their
accuracy in calm periods survives?

The main case is Ecuador's billed electricity consumption (January 2014 to July
2026), which went through the COVID-19 lockdown in 2020 and a drought with
scheduled load shedding in late 2024. Brazil (the 2001 rationing and COVID-19) and the 27 EU
countries (COVID-19 and the 2022 energy crisis) serve as external checks.

The version submitted in August 2026, with four annual folds, is tagged
[`v1.0`](https://github.com/ErickMendoza0/regime-shift-demand-forecasting/tree/v1.0).
Version `v2.0` replaces it with monthly forecast origins, preprocessing cut at
each origin, uniform tuning and dependence-aware tests. The analysis plan was
committed before the campaign ran: [docs/preregistration.md](docs/preregistration.md).

## What is compared

| Family | Models |
|---|---|
| Simple | naive, seasonal naive, seasonal naive with drift, 3-year seasonal mean, drift |
| Statistical | ETS, Theta, M4 Comb, ARIMA (aggregate and per series), SARIMA, SARIMAX with ONI |
| Decomposition | Prophet |
| Gradient boosting | LightGBM, LightGBM with ONI |
| Recurrent | GRU, LSTM, LSTM with ONI, BiLSTM (PyTorch) |
| Deep | N-BEATS, N-HiTS, PatchTST, DLinear, TiDE (neuralforecast) |
| Foundation, zero-shot | Chronos-2, TimesFM 3.0, Moirai 2.0 |
| Combinations | regime-adaptive switch, change-point switch, fixed-share weights, median, 3-model mean |

The ONI enters in three information tiers: the last published value persisted
(`x1`), an ARIMA forecast of ONI made at the origin (`x2`), and the realised
values (`x3`). Only `x1` and `x2` are usable in practice; `x3` is reported as a
hindsight bound and never ranked with the others.

## Protocol in short

- Monthly origins with an expanding window and a 12-month horizon.
- Series admissibility, gap filling, scaling and ONI are computed with the data
  available at each origin; `tests/test_leakage.py` checks that no forecast
  changes when anything after the origin is altered.
- Every learned model gets 50 Optuna trials scored on validation windows before
  the tuning origin. Seeded models run with several seeds and are represented by
  the median forecast.
- Regimes come from a change-point analysis of year-on-year growth
  (`src/changepoints.py`), not from calendar years.
- Months in which a distributor did not report to the Ecuadorian regulator are
  treated as missing rather than as drops in consumption (`src/data.py`).
- Errors are scaled per target and origin (MASE). Comparisons use the model
  confidence set, Diebold-Mariano tests with a HAC variance and Holm correction,
  and Giacomini-White and Giacomini-Rossi tests of regime-dependent performance.

## Layout

```text
config.py            paths, datasets, origins, model registry
src/
  fetch.py           download ONI, Brazil, EU data and model weights
  data.py            panels and the origin cut (history)
  oni.py             ONI as known at each origin, information tiers
  models/            one module per family
  spaces.py          tuning search spaces (python -m src.spaces prints them)
  tune.py            Optuna studies
  jobs.py, run.py    job lists and the forecast runner
  changepoints.py    break dates and regime labels
  combine.py         combinations and switching rules
  evaluate.py        scores, degradation, tests
  inference.py       DM, MCS, Giacomini-White, Giacomini-Rossi, Friedman
  pooled.py          comparison across all shift episodes
  figures.py         paper figures (plot_style.py sets the look)
  tables.py          LaTeX tables of the paper
  summary.py         LaTeX macros with every number quoted in the paper
slurm/               job scripts and submit.sh
tests/               leakage, data and style tests
docs/                analysis plan
envs/                exact package versions
```

## Running

```bash
bash setup_env.sh                       # three conda environments
conda activate regime2_fm
python -m src.fetch --weights           # needs internet; Ecuador's file is manual
conda activate regime2
python -m pytest
bash slurm/submit.sh                    # the whole campaign on SLURM
```

Each step also runs on a single machine, for example
`python -m src.run --dataset ecuador --model ets --origin 2024-01` or
`python -m src.evaluate --dataset ecuador`. Outputs go to `work/` (or
`$REGIME_WORK`): forecasts in `preds/`, tuned parameters in `tuning/`, tables in
`tables/` and figures in `figures/`. `python -m src.summary` writes
`tables/results.tex`, which defines `
es{key}` for every number the paper
quotes, and `tables/summary.md` with the checks of the analysis plan;
`python -m src.tables` writes the LaTeX tables to `tables/latex/`.

## Data

See [data/README.md](data/README.md). Ecuador's billing extract comes from the
ARCONEL portal and is not redistributed here; the other inputs are downloaded
by `src/fetch.py`.
