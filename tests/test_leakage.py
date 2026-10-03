"""A forecast made at an origin must not move when data after the origin change.

The check runs every model on a small synthetic panel twice: once as is and
once with everything from the origin onwards scaled by 50, a gap punched into
one series and the future ONI values altered. Only the hindsight ONI tier (x3)
is allowed to change, and it must.
"""
import numpy as np
import pandas as pd
import pytest

from src import forecast
from src.models import Context, recurrent

ORIGIN = pd.Timestamp("2019-01-01")


def toy():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2014-01-01", "2020-12-01", freq="MS")
    rows, static = [], []
    for i in range(4):
        level = 50 + 20 * i
        y = (level * (1 + 0.003 * np.arange(len(dates)))
             * (1 + 0.08 * np.sin(2 * np.pi * dates.month / 12 + i))
             + rng.normal(0, 1, len(dates)))
        rows.append(pd.DataFrame({"series_id": f"s{i}", "date": dates, "y": y}))
        static.append({"series_id": f"s{i}", "target": "T", "group_code": i % 2,
                       "latitude": -1.0 - i, "longitude": -78.0 - i})
    panel = pd.concat(rows, ignore_index=True)
    oni = pd.Series(np.sin(np.arange(400) / 7.0),
                    index=pd.date_range("1990-01-01", periods=400, freq="MS"))
    return panel, pd.DataFrame(static), oni


def poisoned(panel, oni):
    p = panel.copy()
    future = p["date"] >= ORIGIN
    p.loc[future, "y"] *= 50
    p.loc[future & (p["series_id"] == "s1") & (p["date"] < "2019-06-01"), "y"] = np.nan
    o = oni.copy()
    o[o.index > ORIGIN - pd.DateOffset(months=2)] += 5
    return p, o


FAST = {
    "prophet": {},
    "lgbm": {"n_estimators": 50},
    "lgbm_oni": {"n_estimators": 50},
    "lstm": {}, "lstm_oni": {}, "gru": {}, "bilstm": {},
    "nbeats": {"max_steps": 10}, "nhits": {"max_steps": 10}, "patchtst": {"max_steps": 10},
    "dlinear": {"max_steps": 10}, "tide": {"max_steps": 10},
}
MODELS = [("naive", None), ("snaive", None), ("snaive_drift", None), ("swa3", None),
          ("drift", None), ("ets", None), ("theta", None), ("comb", None),
          ("arima_agg", None), ("arima", None), ("sarima", None),
          ("sarimax", "x1"), ("sarimax", "x2"), ("prophet", None), ("lgbm", None),
          ("lgbm_oni", "x1"), ("lgbm_oni", "x2"), ("lstm", None), ("lstm_oni", "x1"),
          ("gru", None), ("bilstm", None), ("nbeats", None), ("nhits", None),
          ("patchtst", None), ("dlinear", None), ("tide", None)]


def run(model, tier, panel, static, oni):
    ctx = Context("ecuador", ORIGIN, seed=1, params=FAST.get(model, {}), tier=tier, oni=oni)
    return forecast.targets("ecuador", model, ctx, panel, static)["y_hat"].to_numpy()


@pytest.fixture(autouse=True)
def short_training(monkeypatch):
    monkeypatch.setattr(recurrent, "MAX_EPOCHS", 3)


@pytest.mark.parametrize("model,tier", MODELS)
def test_future_does_not_leak(model, tier):
    panel, static, oni = toy()
    bad_panel, bad_oni = poisoned(panel, oni)
    a = run(model, tier, panel, static, oni)
    b = run(model, tier, bad_panel, static, bad_oni)
    np.testing.assert_allclose(a, b, rtol=1e-6)


@pytest.mark.parametrize("model", ["sarimax", "lgbm_oni"])
def test_hindsight_tier_sees_the_future(model):
    panel, static, oni = toy()
    _, bad_oni = poisoned(panel, oni)
    a = run(model, "x3", panel, static, oni)
    b = run(model, "x3", panel, static, bad_oni)
    assert not np.allclose(a, b)
