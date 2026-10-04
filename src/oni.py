"""Oceanic Niño Index as it was known at each forecast origin.

The feature attached to month t is the ONI centred two months earlier (t - 2),
the newest value that is published by the time month t - 1 has closed. Over a
forecast horizon some of those values lie in the future of the origin, and the
information tier decides what replaces them:

  x1  persist the last published value (deployable)
  x2  an ARIMA forecast of ONI fitted at the origin (deployable)
  x3  the realised values (hindsight; an upper bound, never ranked with x1/x2)
"""
import io

import numpy as np
import pandas as pd

import config as C

SEASONS = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
           "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}

TIERS = ("x1", "x2", "x3")


def load() -> pd.Series:
    """ONI indexed by the centre month of each three-month season."""
    df = pd.read_csv(io.StringIO(C.ONI_TXT.read_text()), sep=r"\s+")
    dates = pd.to_datetime(df["YR"].astype(str) + "-" + df["SEAS"].map(SEASONS).astype(str))
    return pd.Series(df["ANOM"].values, index=dates, name="oni").sort_index().asfreq("MS")


def feature(oni: pd.Series, dates) -> np.ndarray:
    """Realised feature values (ONI at t - ONI_LAG) for in-sample months."""
    idx = pd.DatetimeIndex(dates) - pd.DateOffset(months=C.ONI_LAG)
    return oni.reindex(idx).to_numpy(float)


def known_at(oni: pd.Series, origin) -> pd.Series:
    """ONI values already published at the origin."""
    last = pd.Timestamp(origin) - pd.DateOffset(months=C.ONI_LAG)
    return oni.loc[:last].dropna()


def horizon(oni: pd.Series, origin, dates, tier: str) -> np.ndarray:
    """Feature values over the forecast dates under an information tier."""
    if tier not in TIERS:
        raise ValueError(tier)
    need = pd.DatetimeIndex(dates) - pd.DateOffset(months=C.ONI_LAG)
    if tier == "x3":
        # Months past the end of the ONI record have no realised value yet; they
        # also lie past the last month of consumption data and are never scored,
        # so the last realised value is carried forward to keep the run finite.
        return oni.reindex(need).ffill().fillna(oni.dropna().iloc[-1]).to_numpy(float)
    known = known_at(oni, origin)
    out = known.reindex(need)
    missing = out.isna()
    if not missing.any():
        return out.to_numpy(float)
    if tier == "x1":
        out[missing] = known.iloc[-1]
    else:
        steps = int(((need[missing].max().year - known.index[-1].year) * 12
                     + need[missing].max().month - known.index[-1].month))
        path = _arima_path(known, steps)
        out[missing] = path.reindex(need[missing]).to_numpy(float)
    return out.to_numpy(float)


def _arima_path(known: pd.Series, steps: int) -> pd.Series:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    df = pd.DataFrame({"unique_id": "oni", "ds": known.index, "y": known.values})
    fc = StatsForecast(models=[AutoARIMA()], freq="MS").forecast(df=df, h=steps)
    return pd.Series(fc["AutoARIMA"].to_numpy(float), index=pd.DatetimeIndex(fc["ds"]))
