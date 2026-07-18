"""Central configuration: paths, evaluation protocol, model registry.

Paths default to a `work/` directory next to this file but can be redirected
with the ENERGY_WORK environment variable so the same code runs on a laptop
and on a cluster.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("ENERGY_WORK", REPO_ROOT / "work"))

DATA_RAW = REPO_ROOT / "data" / "raw"
DATA_PROCESSED = WORK / "processed"
RESULTS = WORK / "results"
PREDS = RESULTS / "preds"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
LOGS = REPO_ROOT / "logs"

RAW_CSV = DATA_RAW / "Datos_Energeticos_Ecuador_2014_2024.csv"
ONI_CSV = DATA_RAW / "oni.csv"

# Rolling-origin (expanding window) evaluation. For each fold the models are
# trained on every month strictly before January of `test_year` and produce a
# recursive 12-month forecast for that year. 2021-2023 form the "stable"
# regime; 2024 is the demand-shift stress test.
FOLD_YEARS = [2021, 2022, 2023, 2024]
STABLE_YEARS = [2021, 2022, 2023]   # pooled stable regime (36 test months)
CRISIS_YEARS = [2024]               # pooled crisis regime (12 test months)
HORIZON = 12

# Sub-series admissibility: drop a series if more than MAX_BAD_YEARS of its
# years have more than MAX_MISSING_PER_YEAR missing months, or if it is shorter
# than MIN_MONTHS.
MIN_MONTHS = 120
MAX_BAD_YEARS = 2
MAX_MISSING_PER_YEAR = 2

SEED = 42

# Model registry: name -> (module, needs_gpu)
MODELS = {
    "seasonal_naive": ("models_stat", False),
    "arima_agg":      ("models_stat", False),   # ARIMA(1,1,1) on the national series
    "arima_dis":      ("models_stat", False),   # per-subseries auto_arima
    "sarima_dis":     ("models_stat", False),   # per-subseries seasonal auto_arima
    "sarimax_dis":    ("models_stat", False),   # SARIMA + ONI exogenous
    "prophet_dis":    ("models_ml", False),
    "lgbm":           ("models_ml", False),     # global LightGBM with lag features
    "lgbm_exog":      ("models_ml", False),     # + ONI features
    "gru":            ("models_dl", True),
    "lstm":           ("models_dl", True),
    "lstm_exog":      ("models_dl", True),       # + ONI channel
    "bilstm":         ("models_dl", True),
    "nbeats":         ("models_sota", True),
    "patchtst":       ("models_sota", True),
}

# How the exogenous ONI series is handled over a 12-month horizon.
# "persist" freezes the last observed ONI value (deployable scenario);
# "observed" feeds the realised ONI values (a hindcast upper bound).
EXOG_MODES = ["persist", "observed"]

# Regime-adaptive switching (RAS) ensemble, computed post-hoc from the stored
# predictions. Under normal conditions the default forecaster is used; when its
# recent accuracy degrades enough the fallback takes over the rest of the year.
SWITCH_DEFAULT = "lstm"
SWITCH_FALLBACK = "sarima_dis"
SWITCH_WINDOW = 2             # consecutive months of degradation before switching
SWITCH_FACTOR = 1.5           # trailing-MAPE degradation ratio that triggers a switch

for _p in (DATA_PROCESSED, PREDS, TABLES, FIGURES, LOGS):
    _p.mkdir(parents=True, exist_ok=True)
