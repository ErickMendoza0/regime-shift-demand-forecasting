"""Deep forecasters from neuralforecast: N-BEATS, N-HiTS, PatchTST, DLinear, TiDE.

All are global models trained on the whole panel and forecast the 12 months
directly. The last 12 months of history are the early-stopping window.
"""
import logging

import pandas as pd

from src.models import frame, nixtla


def _blocks(spec: str) -> list[int]:
    return [int(v) for v in spec.split("-")]


def _build(name, p, h, seed):
    from neuralforecast.models import DLinear, NBEATS, NHITS, PatchTST, TiDE

    import torch
    common = dict(
        h=h, input_size=int(p["input_size"]), learning_rate=float(p["learning_rate"]),
        max_steps=int(p["max_steps"]), batch_size=int(p["batch_size"]),
        scaler_type=p["scaler_type"], random_seed=seed,
        early_stop_patience_steps=5, val_check_steps=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
        logger=False, enable_progress_bar=False, enable_model_summary=False,
        enable_checkpointing=False,
    )
    if name == "nbeats":
        n = int(p["n_blocks"])
        w = int(p["width"])
        return NBEATS(n_blocks=[n, n, n], mlp_units=[[w, w]] * 3,
                      n_basis=int(p["n_basis"]), n_harmonics=int(p["n_harmonics"]),
                      **common)
    if name == "nhits":
        w = int(p["width"])
        return NHITS(n_pool_kernel_size=_blocks(p["pooling"]),
                     n_freq_downsample=_blocks(p["downsample"]), mlp_units=[[w, w]] * 3,
                     dropout_prob_theta=float(p["dropout_prob_theta"]), **common)
    if name == "patchtst":
        patch = min(int(p["patch_len"]), int(p["input_size"]))
        stride = max(patch // 2, 1) if p["overlap"] else patch
        return PatchTST(patch_len=patch, stride=stride, hidden_size=int(p["hidden_size"]),
                        n_heads=int(p["n_heads"]), encoder_layers=int(p["encoder_layers"]),
                        dropout=float(p["dropout"]), **common)
    if name == "dlinear":
        return DLinear(moving_avg_window=int(p["moving_avg_window"]), **common)
    if name == "tide":
        return TiDE(hidden_size=int(p["hidden_size"]),
                    decoder_output_dim=int(p["decoder_output_dim"]),
                    temporal_decoder_dim=int(p["temporal_decoder_dim"]),
                    num_encoder_layers=int(p["num_encoder_layers"]),
                    num_decoder_layers=int(p["num_decoder_layers"]),
                    dropout=float(p["dropout"]), **common)
    raise ValueError(name)


def forecast(name, hist, static, ctx) -> pd.DataFrame:
    from neuralforecast import NeuralForecast
    from src.spaces import DEFAULTS

    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    p = {**DEFAULTS[name], **ctx.params}
    nf = NeuralForecast(models=[_build(name, p, ctx.horizon, ctx.seed)], freq="MS")
    nf.fit(nixtla(hist), val_size=ctx.horizon)
    fc = nf.predict()
    if "unique_id" not in fc.columns:
        fc = fc.reset_index()
    col = [c for c in fc.columns if c not in ("unique_id", "ds", "cutoff")][0]
    return frame(fc["unique_id"].to_numpy(), pd.DatetimeIndex(fc["ds"]), fc[col].to_numpy(float))
