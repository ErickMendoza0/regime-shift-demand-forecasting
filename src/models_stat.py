"""Statistical benchmarks: seasonal-naive, ARIMA (aggregate and disaggregated),
SARIMA and SARIMAX (+ONI).

The auto_arima search space is stepwise with p, q up to 3 and differencing
order chosen by unit-root test; the seasonal variants use m=12 with P, Q up to
2 and seasonal differencing chosen automatically.
"""
import warnings

import numpy as np
import pandas as pd

import config as C
from src import common

warnings.filterwarnings("ignore")


def _auto_arima(y, seasonal: bool, X=None):
    import pmdarima as pm
    return pm.auto_arima(
        y, X=X,
        start_p=0, start_q=0, max_p=3, max_q=3, d=None,
        seasonal=seasonal, m=12 if seasonal else 1,
        start_P=0, start_Q=0, max_P=2, max_Q=2, D=None,
        stepwise=True, suppress_warnings=True, error_action="ignore")


def _iter_series(panel: pd.DataFrame):
    for sid, g in panel.groupby("series_id"):
        yield sid, g.set_index("date")["energy_gwh"].asfreq("MS")


def run(model: str, fold: int, exog_mode: str = "persist") -> None:
    national = common.load_national()
    tr_idx, te_idx = common.split_fold(national.index, fold)
    y_test = national.loc[te_idx]

    if model == "seasonal_naive":
        y_pred = national.shift(12).loc[te_idx]

    elif model == "arima_agg":
        from statsmodels.tsa.arima.model import ARIMA
        fit = ARIMA(national.loc[tr_idx], order=(1, 1, 1)).fit()
        y_pred = pd.Series(fit.forecast(C.HORIZON).values, index=te_idx)

    elif model in ("arima_dis", "sarima_dis", "sarimax_dis"):
        panel = common.load_series()
        seasonal = model != "arima_dis"
        use_x = model == "sarimax_dis"
        oni = common.load_oni() if use_x else None
        total = pd.Series(0.0, index=te_idx)
        for sid, s in _iter_series(panel):
            s_tr = s.loc[s.index < te_idx[0]]
            X_tr = X_te = None
            if use_x:
                X_tr = oni.reindex(s_tr.index).ffill().bfill().values.reshape(-1, 1)
                X_te = common.exog_for_horizon(
                    oni, s_tr.index[-1], te_idx, exog_mode).values.reshape(-1, 1)
            try:
                m = _auto_arima(s_tr, seasonal=seasonal, X=X_tr)
                f = m.predict(n_periods=C.HORIZON, X=X_te)
            except Exception as e:
                # Rare non-invertible series: fall back to a seasonal-naive forecast.
                print(f"  warning: {sid} raised {type(e).__name__}, using seasonal naive")
                f = s_tr.iloc[-12:].values
            total += pd.Series(np.asarray(f, float), index=te_idx)
        y_pred = total
    else:
        raise ValueError(model)

    extra = {"exog_mode": exog_mode} if model == "sarimax_dis" else None
    name = model if model != "sarimax_dis" else f"{model}_{exog_mode}"
    common.save_predictions(name, fold, y_test, y_pred, extra)
