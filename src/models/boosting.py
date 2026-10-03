"""Global LightGBM on lagged values, forecasting recursively.

Each series is divided by the mean of its last 12 training months so one model
can learn shapes shared by series of very different size.
"""
import numpy as np
import pandas as pd

import config as C
from src import oni as O
from src.models import frame

DEFAULTS = {"n_lags": 12, "num_leaves": 31, "learning_rate": 0.05, "n_estimators": 600,
            "min_child_samples": 20, "subsample": 0.9, "colsample_bytree": 0.9,
            "reg_lambda": 1e-3}


def _static_cols(static):
    if "group_code" in static:
        return ["group_code", "latitude", "longitude"]
    return ["series_code"]


def forecast(name, hist, static, ctx) -> pd.DataFrame:
    import lightgbm as lgb

    p = {**DEFAULTS, **ctx.params}
    n_lags = int(p.pop("n_lags"))
    use_oni = name == "lgbm_oni"
    static = static.assign(series_code=np.arange(len(static)))
    scols = _static_cols(static)
    info = static.set_index("series_id")[scols]

    series = {sid: g.sort_values("date").set_index("date")["y"]
              for sid, g in hist.groupby("series_id")}
    scale = {sid: max(float(s.iloc[-12:].mean()), 1e-6) for sid, s in series.items()}

    def features(sid, values, dates):
        rows = []
        for i, d in enumerate(dates):
            lags = values[i:i + n_lags][::-1]
            rows.append([*lags, np.sin(2 * np.pi * d.month / 12),
                         np.cos(2 * np.pi * d.month / 12), *info.loc[sid]])
        return rows

    X, y = [], []
    for sid, s in series.items():
        v = s.to_numpy(float) / scale[sid]
        if len(v) <= n_lags:
            continue
        X += features(sid, v, s.index[n_lags:])
        y += list(v[n_lags:])
    cols = [f"lag_{k}" for k in range(1, n_lags + 1)] + ["month_sin", "month_cos"] + scols
    X = pd.DataFrame(X, columns=cols)
    if use_oni:
        X["oni"] = np.concatenate([O.feature(ctx.oni, s.index[n_lags:])
                                   for s in series.values() if len(s) > n_lags])
    model = lgb.LGBMRegressor(**p, subsample_freq=1, random_state=ctx.seed, verbose=-1,
                              n_jobs=ctx.n_jobs)
    model.fit(X, np.asarray(y), categorical_feature=[c for c in scols if c.endswith("code")])

    oni_path = O.horizon(ctx.oni, ctx.origin, ctx.dates, ctx.tier) if use_oni else None
    buffers = {sid: list(s.to_numpy(float)[-n_lags:] / scale[sid]) for sid, s in series.items()}
    preds = {sid: [] for sid in series}
    for step, d in enumerate(ctx.dates):
        rows = [features(sid, np.array(buf[-n_lags:]), [d])[0] for sid, buf in buffers.items()]
        Xs = pd.DataFrame(rows, columns=cols)
        if use_oni:
            Xs["oni"] = oni_path[step]
        yhat = model.predict(Xs)
        for sid, v in zip(buffers, yhat):
            buffers[sid].append(v)
            preds[sid].append(v * scale[sid])
    return pd.concat([frame(sid, ctx.dates, preds[sid]) for sid in series], ignore_index=True)
