"""Prophet, one model per series, sharing one set of tuned priors."""
import logging

import pandas as pd

from src.models import frame

DEFAULTS = {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10.0,
            "seasonality_mode": "additive", "changepoint_range": 0.8}


def _one(sid, g, dates, params):
    from prophet import Prophet
    logging.getLogger("cmdstanpy").disabled = True
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, **params)
    m.fit(g.rename(columns={"date": "ds"})[["ds", "y"]])
    yhat = m.predict(pd.DataFrame({"ds": dates}))["yhat"].to_numpy(float)
    return frame(sid, dates, yhat)


def forecast(name, hist, static, ctx) -> pd.DataFrame:
    from joblib import Parallel, delayed
    params = {**DEFAULTS, **ctx.params}
    parts = Parallel(n_jobs=ctx.n_jobs)(
        delayed(_one)(sid, g, ctx.dates, params) for sid, g in hist.groupby("series_id"))
    return pd.concat(parts, ignore_index=True)
