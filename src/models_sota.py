"""Neural forecasters from Nixtla's neuralforecast: N-BEATS and PatchTST.

Both are trained as global models on the sub-series panel and produce a direct
12-month forecast per fold, summed to the national curve.
"""
import numpy as np
import pandas as pd

import config as C
from src import common


def _import_nf():
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS, PatchTST
    return NeuralForecast, NBEATS, PatchTST


def _trainer_kwargs() -> dict:
    """Single-accelerator, silent Lightning trainer; GPU when visible, else CPU."""
    try:
        import torch
        accel = "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        accel = "cpu"
    return dict(accelerator=accel, devices=1, enable_progress_bar=False,
                enable_model_summary=False, logger=False)


def _predict_to_national(nf, te_idx) -> pd.Series:
    fc = nf.predict()
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    model_col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
    fc["ds"] = pd.to_datetime(fc["ds"])
    total = fc.groupby("ds")[model_col].sum().reindex(te_idx)
    return pd.Series(np.asarray(total.values, float), index=te_idx)


def run(model: str, fold: int, exog_mode: str = "persist") -> None:
    NeuralForecast, NBEATS, PatchTST = _import_nf()
    common.set_seeds()

    national = common.load_national()
    _, te_idx = common.split_fold(national.index, fold)
    y_test = national.loc[te_idx]

    panel = common.load_series()
    df = (panel.rename(columns={"series_id": "unique_id", "date": "ds",
                                "energy_gwh": "y"})[["unique_id", "ds", "y"]]
              .copy())
    df["ds"] = pd.to_datetime(df["ds"])
    train = df[df["ds"] < te_idx[0]]

    common_kw = dict(h=C.HORIZON, input_size=2 * C.HORIZON, max_steps=1000,
                     scaler_type="robust", random_seed=C.SEED,
                     **_trainer_kwargs())
    if model == "nbeats":
        nets = [NBEATS(**common_kw)]
    elif model == "patchtst":
        # patch_len and stride must fit input_size; keep conservative defaults.
        nets = [PatchTST(patch_len=8, stride=8, **common_kw)]
    else:
        raise ValueError(model)

    nf = NeuralForecast(models=nets, freq="MS")
    nf.fit(train)
    y_pred = _predict_to_national(nf, te_idx)
    common.save_predictions(model, fold, y_test, y_pred)
