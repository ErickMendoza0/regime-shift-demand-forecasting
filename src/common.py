"""Shared helpers: reproducibility, data loading, error metrics, prediction I/O."""
import os
import random

import numpy as np
import pandas as pd

import config as C


def set_seeds(seed: int = C.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_series() -> pd.DataFrame:
    """Long panel of admissible sub-series with columns
    [series_id, date, energy_gwh, consumer_group, group_code, latitude, longitude]."""
    return pd.read_parquet(C.DATA_PROCESSED / "subseries.parquet")


def load_national() -> pd.Series:
    df = pd.read_parquet(C.DATA_PROCESSED / "national.parquet")
    return df.set_index("date")["energy_gwh"].asfreq("MS")


def load_oni() -> pd.Series:
    oni = pd.read_csv(C.ONI_CSV, parse_dates=["date"])
    return oni.set_index("date")["oni"].asfreq("MS")


def split_fold(index: pd.DatetimeIndex, test_year: int):
    train = index[index < pd.Timestamp(f"{test_year}-01-01")]
    test = index[(index >= pd.Timestamp(f"{test_year}-01-01"))
                 & (index < pd.Timestamp(f"{test_year + 1}-01-01"))]
    return train, test


def exog_for_horizon(oni: pd.Series, train_end: pd.Timestamp,
                     test_index: pd.DatetimeIndex, mode: str) -> pd.Series:
    """ONI values fed to the models over the forecast horizon."""
    if mode == "observed":
        return oni.reindex(test_index)
    last = oni.loc[:train_end].dropna().iloc[-1]
    return pd.Series(last, index=test_index)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def mape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs((y - yhat) / y)) * 100.0)


def mase(y, yhat, y_train, m: int = 12):
    """Seasonal MASE (Hyndman & Koehler, 2006) with period m."""
    y, yhat, y_train = map(lambda a: np.asarray(a, float), (y, yhat, y_train))
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(np.mean(np.abs(y - yhat)) / scale)


ALL_METRICS = {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def save_predictions(model: str, fold: int, y_true: pd.Series, y_pred: pd.Series,
                     extra: dict | None = None) -> None:
    """Write the monthly national-level predictions for one (model, fold)."""
    out = pd.DataFrame({
        "date": y_true.index,
        "y_true": y_true.values,
        "y_pred": np.asarray(y_pred, float),
    })
    out["model"] = model
    out["fold"] = fold
    for k, v in (extra or {}).items():
        out[k] = v
    path = C.PREDS / f"{model}_{fold}.csv"
    out.to_csv(path, index=False)
    print(f"saved {path} ({len(out)} rows)")


def load_all_predictions() -> pd.DataFrame:
    frames = [pd.read_csv(p, parse_dates=["date"]) for p in sorted(C.PREDS.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"no predictions found under {C.PREDS}")
    return pd.concat(frames, ignore_index=True)
