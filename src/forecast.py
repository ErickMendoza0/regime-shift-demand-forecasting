"""Produce one model's forecast of every target from one origin."""
import numpy as np
import pandas as pd

import config as C
from src import data
from src.models import Context, forecaster


def targets(dataset, model, ctx: Context, panel, static) -> pd.DataFrame:
    """Forecast summed to each target, with the series the model could not use
    at this origin covered by a seasonal-naive fallback."""
    hist, dropped = data.history(panel, ctx.origin, dataset)
    out = forecaster(model)(hist, static, ctx)
    if not np.isfinite(out["y_hat"]).all():
        raise ValueError(f"{model} produced non-finite forecasts at {ctx.origin:%Y-%m}")
    if not out["date"].isin(ctx.dates).all():
        raise ValueError(f"{model} forecast dates outside the horizon at {ctx.origin:%Y-%m}")
    spare = data.fallback(panel, dropped, ctx.origin, ctx.horizon)
    spare = spare.merge(static[["series_id", "target"]], on="series_id")
    if "target" not in out:
        out = out.merge(static[["series_id", "target"]], on="series_id")
    both = pd.concat([out[["target", "date", "y_hat"]], spare[["target", "date", "y_hat"]]])
    fc = both.groupby(["target", "date"], as_index=False)["y_hat"].sum()
    fc["origin"] = ctx.origin
    fc["horizon"] = ((fc["date"].dt.year - ctx.origin.year) * 12
                     + fc["date"].dt.month - ctx.origin.month + 1)
    return fc


def mase_scale(truth: pd.DataFrame, origin) -> pd.Series:
    """In-sample seasonal-naive MAE of each target before the origin."""
    past = truth[truth["date"] < pd.Timestamp(origin)]
    out = {}
    for t, g in past.groupby("target"):
        y = g.sort_values("date")["y"].to_numpy(float)
        out[t] = np.mean(np.abs(y[C.SEASON:] - y[:-C.SEASON]))
    return pd.Series(out)


def score(fc: pd.DataFrame, truth: pd.DataFrame, before=None) -> float:
    """Mean MASE over targets; `before` limits scoring to earlier target dates
    (used in tuning, where nothing at or after the tuning origin is visible)."""
    origin = fc["origin"].iloc[0]
    df = fc.merge(truth, on=["target", "date"])
    if before is not None:
        df = df[df["date"] < pd.Timestamp(before)]
    scale = mase_scale(truth, origin)
    err = (df["y"] - df["y_hat"]).abs().groupby(df["target"]).mean()
    return float((err / scale.reindex(err.index)).mean())
