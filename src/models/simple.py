"""Simple reference forecasts, one series at a time."""
import numpy as np
import pandas as pd

import config as C
from src.models import frame

M = C.SEASON


def _naive(y, h):
    return np.repeat(y[-1], h)


def _snaive(y, h):
    return np.array([y[-M + (i % M)] for i in range(h)])


def _snaive_drift(y, h):
    # Seasonal naive shifted by the average year-on-year change seen so far.
    step = np.mean(y[M:] - y[:-M])
    return np.array([y[-M + (i % M)] + step * (1 + i // M) for i in range(h)])


def _swa3(y, h):
    # Mean of the same calendar month over the last three years.
    out = []
    for i in range(h):
        k = i % M
        out.append(np.mean([y[-M * j + k] for j in (1, 2, 3) if M * j - k <= len(y)]))
    return np.array(out)


def _drift(y, h):
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + slope * np.arange(1, h + 1)


RULES = {"naive": _naive, "snaive": _snaive, "snaive_drift": _snaive_drift,
         "swa3": _swa3, "drift": _drift}


def forecast(name, hist, static, ctx) -> pd.DataFrame:
    rule = RULES[name]
    out = []
    for sid, g in hist.groupby("series_id"):
        y = g.sort_values("date")["y"].to_numpy(float)
        out.append(frame(sid, ctx.dates, rule(y, ctx.horizon)))
    return pd.concat(out, ignore_index=True)
