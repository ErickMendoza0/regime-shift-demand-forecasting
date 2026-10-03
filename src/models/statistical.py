"""Statistical models: ETS, Theta, the M4 Comb benchmark and (S)ARIMA(X).

ETS, Theta and the ARIMA family use statsforecast and pick their own order or
components by information criterion at every origin, which is how they are
tuned. arima_agg keeps the original ARIMA(1,1,1) on the aggregate.
"""
import warnings

import numpy as np
import pandas as pd

import config as C
from src import oni as O
from src.models import frame, nixtla

warnings.filterwarnings("ignore")
M = C.SEASON


def _statsforecast(name, hist, ctx):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

    model = {
        "ets": lambda: AutoETS(season_length=M),
        "theta": lambda: AutoTheta(season_length=M),
        "arima": lambda: AutoARIMA(season_length=1),
        "sarima": lambda: AutoARIMA(season_length=M),
        "sarimax": lambda: AutoARIMA(season_length=M),
    }[name]()
    df = nixtla(hist)
    X_future = None
    if name == "sarimax":
        df["oni"] = O.feature(ctx.oni, df["ds"])
        future = O.horizon(ctx.oni, ctx.origin, ctx.dates, ctx.tier)
        X_future = pd.DataFrame([
            {"unique_id": sid, "ds": d, "oni": v}
            for sid in df["unique_id"].unique() for d, v in zip(ctx.dates, future)])
    sf = StatsForecast(models=[model], freq="MS", n_jobs=ctx.n_jobs)
    fc = sf.forecast(df=df, h=ctx.horizon, X_df=X_future)
    col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
    return frame(fc["unique_id"].to_numpy(), pd.DatetimeIndex(fc["ds"]), fc[col].to_numpy(float))


def _seasonal(y) -> bool:
    """M4 seasonality test: lag-12 autocorrelation beyond a 90% band."""
    from statsmodels.tsa.stattools import acf
    if len(y) < 3 * M:
        return False
    r = acf(y, nlags=M, fft=False)
    limit = 1.645 * np.sqrt((1 + 2 * np.sum(r[1:M] ** 2)) / len(y))
    return abs(r[M]) > limit


def _comb(y, h):
    """Average of SES, Holt and damped Holt on seasonally adjusted data, as in
    the M4 competition's Comb benchmark."""
    from statsmodels.tsa.holtwinters import Holt, SimpleExpSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose

    in_sample = np.ones(len(y))
    if np.all(y > 0) and _seasonal(y):
        in_sample = seasonal_decompose(y, model="multiplicative", period=M).seasonal
    index = in_sample[-M:]       # index[i % M] is the factor for forecast step i
    adjusted = y / in_sample
    fits = [SimpleExpSmoothing(adjusted, initialization_method="estimated").fit(),
            Holt(adjusted, initialization_method="estimated").fit(),
            Holt(adjusted, damped_trend=True, initialization_method="estimated").fit()]
    mean = np.mean([f.forecast(h) for f in fits], axis=0)
    return mean * np.array([index[i % M] for i in range(h)])


def _arima_agg(hist, static, ctx):
    from statsmodels.tsa.arima.model import ARIMA
    df = hist.merge(static[["series_id", "target"]], on="series_id")
    out = []
    for target, g in df.groupby("target"):
        y = g.groupby("date")["y"].sum().asfreq("MS")
        fc = ARIMA(y, order=(1, 1, 1)).fit().forecast(ctx.horizon)
        out.append(pd.DataFrame({"target": target, "date": ctx.dates,
                                 "y_hat": np.asarray(fc, float)}))
    return pd.concat(out, ignore_index=True)


def forecast(name, hist, static, ctx) -> pd.DataFrame:
    if name == "arima_agg":
        return _arima_agg(hist, static, ctx)
    if name == "comb":
        out = []
        for sid, g in hist.groupby("series_id"):
            y = g.sort_values("date")["y"].to_numpy(float)
            out.append(frame(sid, ctx.dates, _comb(y, ctx.horizon)))
        return pd.concat(out, ignore_index=True)
    return _statsforecast(name, hist, ctx)
