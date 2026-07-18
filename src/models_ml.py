"""Machine-learning benchmarks: Prophet (one model per sub-series) and a global
LightGBM with lag features, optionally extended with ONI exogenous features.
"""
import numpy as np
import pandas as pd

import config as C
from src import common

N_LAGS = 12


def _prophet(panel, te_idx):
    from prophet import Prophet
    total = pd.Series(0.0, index=te_idx)
    for sid, g in panel.groupby("series_id"):
        s = g.set_index("date")["energy_gwh"].asfreq("MS")
        tr = s.loc[s.index < te_idx[0]].reset_index()
        tr.columns = ["ds", "y"]
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False)
        m.fit(tr)
        future = pd.DataFrame({"ds": te_idx})
        total += pd.Series(m.predict(future)["yhat"].values, index=te_idx)
    return total


def _lgbm_features(df: pd.DataFrame, oni: pd.Series | None):
    """Lag, cyclical and static features for each row of the sub-series panel."""
    df = df.sort_values(["series_id", "date"]).copy()
    for l in range(1, N_LAGS + 1):
        df[f"lag_{l}"] = df.groupby("series_id")["energy_gwh"].shift(l)
    month = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    if oni is not None:
        df["oni"] = df["date"].map(oni)
    return df.dropna(subset=[f"lag_{N_LAGS}"])


def _lgbm(panel, te_idx, exog_mode: str | None):
    import lightgbm as lgb
    oni = common.load_oni() if exog_mode else None
    feats = _lgbm_features(panel, oni)
    fcols = ([f"lag_{l}" for l in range(1, N_LAGS + 1)]
             + ["month_sin", "month_cos", "latitude", "longitude", "group_code"]
             + (["oni"] if exog_mode else []))
    train = feats[feats["date"] < te_idx[0]]
    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, random_state=C.SEED, verbose=-1)
    model.fit(train[fcols], train["energy_gwh"],
              categorical_feature=["group_code"])

    # Recursive 12-step forecast per sub-series.
    total = pd.Series(0.0, index=te_idx)
    hist = {sid: g.set_index("date")["energy_gwh"].asfreq("MS")
                 .loc[lambda s: s.index < te_idx[0]]
            for sid, g in panel.groupby("series_id")}
    static = (panel.groupby("series_id")[["latitude", "longitude", "group_code"]]
                   .first().to_dict("index"))
    oni_h = (common.exog_for_horizon(oni, te_idx[0] - pd.offsets.MonthBegin(1),
                                     te_idx, exog_mode) if exog_mode else None)
    for t in te_idx:
        rows, sids = [], []
        for sid, s in hist.items():
            lags = s.iloc[-N_LAGS:].values[::-1]
            row = dict({f"lag_{l+1}": lags[l] for l in range(N_LAGS)},
                       month_sin=np.sin(2 * np.pi * t.month / 12),
                       month_cos=np.cos(2 * np.pi * t.month / 12),
                       **static[sid])
            if exog_mode:
                row["oni"] = oni_h.loc[t]
            rows.append(row)
            sids.append(sid)
        X = pd.DataFrame(rows)[fcols]
        yhat = model.predict(X)
        for sid, v in zip(sids, yhat):
            hist[sid] = pd.concat([hist[sid], pd.Series([v], index=[t])])
        total.loc[t] = float(yhat.sum())
    return total


def run(model: str, fold: int, exog_mode: str = "persist") -> None:
    common.set_seeds()
    national = common.load_national()
    _, te_idx = common.split_fold(national.index, fold)
    y_test = national.loc[te_idx]
    panel = common.load_series()

    if model == "prophet_dis":
        y_pred = _prophet(panel, te_idx)
        name = model
    elif model == "lgbm":
        y_pred = _lgbm(panel, te_idx, exog_mode=None)
        name = model
    elif model == "lgbm_exog":
        y_pred = _lgbm(panel, te_idx, exog_mode=exog_mode)
        name = f"{model}_{exog_mode}"
    else:
        raise ValueError(model)

    common.save_predictions(name, fold, y_test, y_pred)
