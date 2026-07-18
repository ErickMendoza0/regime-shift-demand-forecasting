"""Deep recurrent models (Keras/TensorFlow).

  LSTM   : window 12x5  -> LSTM(128, seq) -> Dropout(0.2) -> LSTM(64)
           -> Dense(32, ReLU) -> Dense(1);      up to 50 epochs, patience 15
  BiLSTM : window 18x5  -> Bi(LSTM(128, seq)) -> Dropout(0.2) -> Bi(LSTM(64))
           -> Dense(64)+LeakyReLU -> Dense(32)+LeakyReLU -> Dense(1);
           up to 100 epochs, patience 20
  GRU    : the LSTM topology with GRU cells (window 12x5)
  lstm_exog : the LSTM topology with an extra ONI channel (window 12x6)

The consumer-group id is fused into the sequence through an Embedding of
dimension 4. All models use Adam(1e-3), MSE loss, batch size 64 and a 10%
validation split with best-weights restore. Per-timestep features are
[scaled GWh, sin, cos, latitude, longitude(, oni)].
"""
import numpy as np
import pandas as pd

import config as C
from src import common

EMB_DIM = 4


def _build(kind: str, window: int, n_feat: int, n_groups: int):
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (LSTM, GRU, Bidirectional, Concatenate,
                                         Dense, Dropout, Embedding, Flatten,
                                         Input, LeakyReLU, RepeatVector)
    seq_in = Input(shape=(window, n_feat), name="seq_input")
    grp_in = Input(shape=(1,), name="group_input")
    emb = Flatten()(Embedding(n_groups, EMB_DIM)(grp_in))
    merged = Concatenate()([seq_in, RepeatVector(window)(emb)])

    if kind == "bilstm":
        x = Bidirectional(LSTM(128, return_sequences=True))(merged)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(64))(x)
        x = LeakyReLU()(Dense(64)(x))
        x = LeakyReLU()(Dense(32)(x))
        out = Dense(1)(x)
    else:
        cell = GRU if kind == "gru" else LSTM
        x = cell(128, return_sequences=True)(merged)
        x = Dropout(0.2)(x)
        x = cell(64)(x)
        x = Dense(32, activation="relu")(x)
        out = Dense(1, activation="linear")(x)

    m = Model([seq_in, grp_in], out)
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m


def _sequences(panel, window, te_start, oni=None):
    """Pooled training sequences from every sub-series (strictly pre-fold)."""
    Xs, gs, ys = [], [], []
    scalers, hists = {}, {}
    for sid, g in panel.groupby("series_id"):
        s = g.set_index("date").asfreq("MS")
        s = s[s.index < te_start]
        e = s["energy_gwh"].astype(float)
        lo, hi = float(e.min()), float(e.max())
        scalers[sid] = (lo, hi if hi > lo else lo + 1.0)
        feats = pd.DataFrame({
            "e": (e - scalers[sid][0]) / (scalers[sid][1] - scalers[sid][0]),
            "sin": np.sin(2 * np.pi * s.index.month / 12),
            "cos": np.cos(2 * np.pi * s.index.month / 12),
            "lat": s["latitude"], "lon": s["longitude"]}, index=s.index)
        if oni is not None:
            feats["oni"] = oni.reindex(s.index).ffill().bfill()
        code = int(g["group_code"].iloc[0])
        arr = feats.values
        for i in range(window, len(arr)):
            Xs.append(arr[i - window:i])
            gs.append(code)
            ys.append(feats["e"].iloc[i])
        hists[sid] = (feats, code)
    return (np.asarray(Xs, "float32"), np.asarray(gs), np.asarray(ys, "float32"),
            scalers, hists)


def run(model: str, fold: int, exog_mode: str = "persist") -> None:
    from tensorflow.keras.callbacks import EarlyStopping
    common.set_seeds()

    kind = "lstm" if model == "lstm_exog" else model
    window = 18 if kind == "bilstm" else 12
    epochs, patience = (100, 20) if kind == "bilstm" else (50, 15)
    use_oni = model == "lstm_exog"

    national = common.load_national()
    _, te_idx = common.split_fold(national.index, fold)
    y_test = national.loc[te_idx]
    panel = common.load_series()
    oni = common.load_oni() if use_oni else None

    X, g, y, scalers, hists = _sequences(panel, window, te_idx[0], oni)
    net = _build(kind, window, X.shape[2], int(panel["group_code"].max()) + 1)
    net.fit({"seq_input": X, "group_input": g}, y,
            validation_split=0.1, epochs=epochs, batch_size=64, verbose=2,
            callbacks=[EarlyStopping(monitor="val_loss", patience=patience,
                                     restore_best_weights=True)])

    # Recursive 12-month forecast per sub-series, summed to the national curve.
    oni_h = (common.exog_for_horizon(oni, te_idx[0] - pd.offsets.MonthBegin(1),
                                     te_idx, exog_mode) if use_oni else None)
    total = pd.Series(0.0, index=te_idx)
    for sid, (feats, code) in hists.items():
        lo, hi = scalers[sid]
        buf = feats.copy()
        for t in te_idx:
            row = {"e": np.nan,
                   "sin": np.sin(2 * np.pi * t.month / 12),
                   "cos": np.cos(2 * np.pi * t.month / 12),
                   "lat": buf["lat"].iloc[-1], "lon": buf["lon"].iloc[-1]}
            if use_oni:
                row["oni"] = oni_h.loc[t]
            win = buf.iloc[-window:].values[np.newaxis].astype("float32")
            yhat = float(net.predict(
                {"seq_input": win, "group_input": np.array([code])},
                verbose=0)[0, 0])
            row["e"] = yhat
            buf = pd.concat([buf, pd.DataFrame([row], index=[t])])
            total.loc[t] += yhat * (hi - lo) + lo

    name = model if not use_oni else f"{model}_{exog_mode}"
    common.save_predictions(name, fold, y_test, total)
