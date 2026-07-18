# Regime-shift demand forecasting

A monthly electricity-demand forecasting benchmark for Ecuador, evaluated with
a rolling-origin protocol over 2021-2024. The 2024 fold captures a sharp demand
shift, which lets us ask the headline question: does a more sophisticated model
actually buy robustness when the regime changes, or do simple seasonal
baselines hold up just as well under stress?

Eighteen forecasters are compared on the same folds: seasonal-naive,
ARIMA/SARIMA/SARIMAX, Prophet, LightGBM (with and without ONI), GRU, LSTM (with
and without ONI), BiLSTM, N-BEATS and PatchTST. The comparison is backed by a
Diebold-Mariano test (with the HLN small-sample correction) and a
Friedman/Nemenyi procedure, an ENSO (ONI) exogenous ablation, and a
regime-adaptive switching (RAS) ensemble.

## Layout

```
config.py              paths, evaluation protocol, model registry
setup_env.sh           creates the two conda environments
requirements*.txt      dependencies for each environment
src/
  common.py            data loading, metrics, prediction I/O, seeding
  prepare_data.py      raw CSV -> sub-series and national parquet files
  fetch_oni.py         download the NOAA ONI series
  models_stat.py       seasonal-naive, ARIMA/SARIMA/SARIMAX
  models_ml.py         Prophet, LightGBM (+ONI)
  models_dl.py         GRU, LSTM (+ONI), BiLSTM (Keras)
  models_sota.py       N-BEATS, PatchTST (neuralforecast)
  ensemble.py          regime-adaptive switching ensemble
  stats_tests.py       metric tables + DM / Friedman / Nemenyi
  make_figures.py      paper figures
  build_jobs.py        enumerate jobs into work/jobs_*.csv
  run_experiment.py    run a single (model, fold) job
  diag_sota.py         sanity check for the neuralforecast stack
slurm/                 SLURM submission scripts (array jobs)
data/                  raw inputs (not versioned); see data/README.md
```

## Environments

Two conda environments are used, one for the TensorFlow/Keras + statistics + ML
stack and one for the PyTorch-based SOTA models, kept separate so their CUDA
libraries do not conflict:

```bash
bash setup_env.sh
```

## Running

The steps below assume a SLURM cluster; set the partition placeholders in
`slurm/*.sh` first. Each step can also be run directly with the corresponding
`python -m src.*` command on a single machine.

```bash
# 1. Fetch the ONI series (needs internet) and place the raw CSV in data/raw/
conda activate energyq1 && python -m src.fetch_oni

# 2. Preprocess the data and build the job lists
sbatch slurm/00_prepare.sh          # build_jobs prints the --array ranges

# 3. CPU models (statistical + ML)
sbatch --array=0-N slurm/01_cpu_array.sh

# 4. GPU models (GRU/LSTM/BiLSTM, Keras)
sbatch --array=0-N%2 slurm/02_gpu_array.sh

# 5. SOTA models (N-BEATS/PatchTST, separate PyTorch env)
sbatch --array=0-7%2 slurm/04_sota_array.sh

# 6. Analysis: switching ensemble, statistical tests, figures
sbatch slurm/03_analysis.sh
```

Outputs land under `work/results/`: monthly predictions per (model, fold) in
`preds/`, the metric and test tables in `tables/`, and the figures in
`figures/`.

## Reproducibility

The global seed (42) is set in `config.py` and applied in `common.set_seeds`.
The raw data are not versioned; see `data/README.md` for how to obtain them.
