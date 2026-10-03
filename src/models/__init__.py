"""Forecasting models. Every model exposes

    forecast(hist, static, ctx) -> DataFrame[series_id, date, y_hat]

where `hist` is the panel returned by data.history (nothing after the origin)
and `ctx` carries the origin, horizon, seed, tuned parameters and ONI tier.
"""
from dataclasses import dataclass, field
import importlib
import os

import pandas as pd

import config as C


@dataclass
class Context:
    dataset: str
    origin: pd.Timestamp
    horizon: int = C.HORIZON
    seed: int = 0
    params: dict = field(default_factory=dict)
    tier: str | None = None
    oni: pd.Series | None = None

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.origin, periods=self.horizon, freq="MS")

    @property
    def n_jobs(self) -> int:
        return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


# model name -> module that implements it
MODULES = {
    "naive": "simple", "snaive": "simple", "snaive_drift": "simple",
    "swa3": "simple", "drift": "simple",
    "ets": "statistical", "theta": "statistical", "comb": "statistical",
    "arima_agg": "statistical", "arima": "statistical", "sarima": "statistical",
    "sarimax": "statistical",
    "prophet": "prophet_model",
    "lgbm": "boosting", "lgbm_oni": "boosting",
    "gru": "recurrent", "lstm": "recurrent", "lstm_oni": "recurrent", "bilstm": "recurrent",
    "nbeats": "neural", "nhits": "neural", "patchtst": "neural", "dlinear": "neural",
    "tide": "neural",
    "chronos": "foundation", "timesfm": "foundation", "moirai": "foundation",
}


def forecaster(name: str):
    mod = importlib.import_module(f"src.models.{MODULES[name]}")
    return lambda hist, static, ctx: mod.forecast(name, hist, static, ctx)


def frame(series_id, dates, values) -> pd.DataFrame:
    return pd.DataFrame({"series_id": series_id, "date": dates, "y_hat": values})


def nixtla(hist: pd.DataFrame) -> pd.DataFrame:
    """Panel in the unique_id / ds / y layout used by the Nixtla libraries."""
    return (hist.rename(columns={"series_id": "unique_id", "date": "ds"})
                [["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])
                .reset_index(drop=True))
