"""Paths, evaluation protocol and model registry.

Everything written by the pipeline goes under WORK, which can be moved with the
REGIME_WORK environment variable so one checkout serves a laptop and the cluster.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
WORK = Path(os.environ.get("REGIME_WORK", ROOT / "work"))

PANELS = WORK / "panels"
JOBS = WORK / "jobs"
TUNING = WORK / "tuning"
PREDS = WORK / "preds"
TABLES = WORK / "tables"
FIGURES = WORK / "figures"

ECUADOR_CSV = RAW / "Datos_Energeticos_Ecuador_2014_2024.csv"
ONI_TXT = RAW / "oni.ascii.txt"
BRAZIL_CSV = RAW / "brazil_ipeadata.csv"
EUROPE_CSV = RAW / "europe_nrg_cb_em.csv"

HORIZON = 12
SEASON = 12

# Forecast origins are the first month of each 12-month forecast, so an origin
# of 2024-01 sees data up to 2023-12. Brazil is evaluated in two blocks, one
# around the 2001 rationing and one around COVID-19.
DATASETS = {
    "ecuador": {
        "train_start": "2014-01",
        "origins": [("2019-01", None)],          # None: last month with data
        "seeds": [1, 2, 3, 4, 5],
        "retune": "yearly",
        "exogenous": True,
    },
    "brazil": {
        "train_start": "1990-01",
        "origins": [("1999-01", "2003-12"), ("2018-01", "2022-12")],
        "seeds": [1, 2, 3],
        "retune": "block",
        "exogenous": False,
    },
    "europe": {
        "train_start": "2008-01",
        "origins": [("2018-01", None)],
        "seeds": [1, 2, 3],
        "retune": "block",
        "exogenous": False,
    },
}

# Ecuador sub-series admissibility, applied at every origin with the data seen
# so far: at least MIN_MONTHS of history and no more than MAX_BAD_YEARS calendar
# years with more than MAX_MISSING_PER_YEAR missing months.
MIN_MONTHS = 36
MAX_BAD_YEARS = 2
MAX_MISSING_PER_YEAR = 2

# ONI is a centred three-month mean that CPC publishes about a week after the
# last month closes, so at an origin the newest usable value is centred two
# months back.
ONI_LAG = 2

TUNING_TRIALS = 50

# Model registry. `kind` decides which job list and environment run the model:
# "cpu" and "gpu" use the regime2 environment, "fm" the foundation-model one.
# `tier` lists the ONI information tiers a model is run with (see src/oni.py).
MODELS = {
    "naive":          {"family": "simple", "kind": "cpu"},
    "snaive":         {"family": "simple", "kind": "cpu"},
    "snaive_drift":   {"family": "simple", "kind": "cpu"},
    "swa3":           {"family": "simple", "kind": "cpu"},
    "drift":          {"family": "simple", "kind": "cpu"},
    "ets":            {"family": "statistical", "kind": "cpu"},
    "theta":          {"family": "statistical", "kind": "cpu"},
    "comb":           {"family": "statistical", "kind": "cpu"},
    "arima_agg":      {"family": "statistical", "kind": "cpu"},
    "arima":          {"family": "statistical", "kind": "cpu"},
    "sarima":         {"family": "statistical", "kind": "cpu"},
    "sarimax":        {"family": "statistical", "kind": "cpu", "tier": ["x1", "x2", "x3"]},
    "prophet":        {"family": "decomposition", "kind": "cpu", "tuned": True},
    "lgbm":           {"family": "boosting", "kind": "cpu", "tuned": True},
    "lgbm_oni":       {"family": "boosting", "kind": "cpu", "tuned": True, "tier": ["x1", "x2", "x3"]},
    "gru":            {"family": "recurrent", "kind": "gpu", "tuned": True},
    "lstm":           {"family": "recurrent", "kind": "gpu", "tuned": True},
    "lstm_oni":       {"family": "recurrent", "kind": "gpu", "tuned": True, "tier": ["x1", "x2", "x3"]},
    "bilstm":         {"family": "recurrent", "kind": "gpu", "tuned": True},
    "nbeats":         {"family": "deep", "kind": "gpu", "tuned": True},
    "nhits":          {"family": "deep", "kind": "gpu", "tuned": True},
    "patchtst":       {"family": "deep", "kind": "gpu", "tuned": True},
    "dlinear":        {"family": "deep", "kind": "gpu", "tuned": True},
    "tide":           {"family": "deep", "kind": "gpu", "tuned": True},
    "chronos":        {"family": "foundation", "kind": "fm"},
    "timesfm":        {"family": "foundation", "kind": "fm"},
    "moirai":         {"family": "foundation", "kind": "fm"},
}

# Models whose output does not depend on the random seed.
DETERMINISTIC = {"naive", "snaive", "snaive_drift", "swa3", "drift", "ets", "theta",
                 "comb", "arima_agg", "arima", "sarima", "sarimax", "prophet",
                 "chronos", "timesfm", "moirai"}


def ensure_dirs() -> None:
    for p in (PANELS, JOBS, TUNING, PREDS, TABLES, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
