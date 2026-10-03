"""Pretrained time-series foundation models, used zero-shot.

Chronos-2 (Amazon), TimesFM 3.0 (Google) and Moirai 2.0 (Salesforce) forecast
each series from its own history without any training on our data. The point
forecast is the median of the predictive distribution.

Weights must already be in the Hugging Face cache (compute nodes have no
internet): run `python -m src.fetch --weights` on a login node first. Moirai
lives in a separate environment because uni2ts pins an older PyTorch.
"""
import numpy as np
import pandas as pd

from src.models import frame

REPOS = {"chronos": "amazon/chronos-2",
         "timesfm": "google/timesfm-3.0-pytorch",
         "moirai": "Salesforce/moirai-2.0-R-small"}
MAX_CONTEXT = 2048
_loaded = {}


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _chronos(contexts, h):
    import torch
    from chronos import BaseChronosPipeline
    if "chronos" not in _loaded:
        _loaded["chronos"] = BaseChronosPipeline.from_pretrained(REPOS["chronos"], device_map=_device())
    q, _ = _loaded["chronos"].predict_quantiles(
        [torch.tensor(c, dtype=torch.float32) for c in contexts],
        prediction_length=h, quantile_levels=[0.5])
    return [t[0, :, 0].cpu().numpy() for t in q]


def _timesfm(contexts, h):
    import timesfm
    if "timesfm" not in _loaded:
        _loaded["timesfm"] = timesfm.TimesFM3Forecaster.from_pretrained(REPOS["timesfm"], device=_device())
    model = _loaded["timesfm"]
    # Quantile columns are the deciles 0.1..0.9; column 4 is the median.
    return [model.predict(c, horizon=h, return_quantiles=True).quantiles[:, 4] for c in contexts]


def _moirai(contexts, h):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    if "moirai" not in _loaded:
        _loaded["moirai"] = Moirai2Module.from_pretrained(REPOS["moirai"])
    module = _loaded["moirai"]
    median = list(module.quantile_levels).index(0.5)
    out = []
    for c in contexts:
        model = Moirai2Forecast(module=module, prediction_length=h, context_length=len(c),
                                target_dim=1, feat_dynamic_real_dim=0,
                                past_feat_dynamic_real_dim=0).to(_device())
        q = model.predict(past_target=[c[:, None]])        # (1, quantiles, h)
        out.append(np.asarray(q)[0, median])
    return out


RUNNERS = {"chronos": _chronos, "timesfm": _timesfm, "moirai": _moirai}


def forecast(name, hist, static, ctx) -> pd.DataFrame:
    ids, contexts = [], []
    for sid, g in hist.groupby("series_id"):
        ids.append(sid)
        contexts.append(g.sort_values("date")["y"].to_numpy(np.float32)[-MAX_CONTEXT:])
    preds = RUNNERS[name](contexts, ctx.horizon)
    return pd.concat([frame(sid, ctx.dates, np.asarray(p, float)) for sid, p in zip(ids, preds)],
                     ignore_index=True)
