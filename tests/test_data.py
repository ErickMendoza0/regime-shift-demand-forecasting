import numpy as np
import pandas as pd

from src import data
from src import oni as O


def test_history_stops_before_origin():
    dates = pd.date_range("2014-01-01", "2020-12-01", freq="MS")
    panel = pd.DataFrame({"series_id": "a", "date": dates, "y": np.arange(len(dates), dtype=float)})
    hist, dropped = data.history(panel, "2019-01-01", "ecuador")
    assert hist["date"].max() == pd.Timestamp("2018-12-01")
    assert dropped == []


def test_admissibility_ignores_gaps_after_origin():
    dates = pd.date_range("2014-01-01", "2020-12-01", freq="MS")
    y = np.arange(len(dates), dtype=float)
    y[(dates >= "2019-01-01")] = np.nan          # the series vanishes after the origin
    panel = pd.DataFrame({"series_id": "a", "date": dates, "y": y})
    _, dropped = data.history(panel, "2019-01-01", "ecuador")
    assert dropped == []


def test_short_series_is_left_out():
    dates = pd.date_range("2017-06-01", "2020-12-01", freq="MS")
    panel = pd.DataFrame({"series_id": "a", "date": dates, "y": 1.0})
    _, dropped = data.history(panel, "2019-01-01", "ecuador")
    assert dropped == ["a"]


def test_series_that_stopped_reporting_is_dropped_and_forecast_as_zero():
    dates = pd.date_range("2014-01-01", "2017-06-01", freq="MS")
    panel = pd.DataFrame({"series_id": "a", "date": dates, "y": 5.0})
    _, dropped = data.history(panel, "2019-01-01", "ecuador")
    assert dropped == ["a"]
    fb = data.fallback(panel, dropped, "2019-01-01")
    assert (fb["y_hat"] == 0).all() and len(fb) == 12


def test_oni_known_at_respects_publication_lag():
    oni = pd.Series(np.arange(24.0), index=pd.date_range("2018-01-01", periods=24, freq="MS"))
    known = O.known_at(oni, "2019-01-01")
    assert known.index.max() == pd.Timestamp("2018-11-01")


def test_oni_persist_tier_freezes_last_published_value():
    oni = pd.Series(np.arange(24.0), index=pd.date_range("2018-01-01", periods=24, freq="MS"))
    dates = pd.date_range("2019-01-01", periods=12, freq="MS")
    x1 = O.horizon(oni, "2019-01-01", dates, "x1")
    # Jan 2019 needs ONI centred in Nov 2018 (published); later months reuse it.
    assert np.all(x1 == oni["2018-11-01"])
    x3 = O.horizon(oni, "2019-01-01", dates, "x3")
    assert x3[-1] == oni["2019-10-01"]


def _billing(clients_by_month):
    rows = []
    for i, n in enumerate(clients_by_month):
        d = pd.Timestamp("2020-01-01") + pd.DateOffset(months=i)
        rows.append({"Empresa": "A", "date": d, "clients": n})
        rows.append({"Empresa": "B", "date": d, "clients": 100})
    return pd.DataFrame(rows)


def test_a_distributor_that_did_not_report_is_a_gap():
    gaps = data.reporting_gaps(_billing([100] * 8 + [0] + [100] * 3))
    assert list(gaps["company"]) == ["A"]
    assert gaps["date"].iloc[0] == pd.Timestamp("2020-09-01")
    assert not gaps["incomplete_month"].any()


def test_unknown_months_are_left_out_of_the_truth():
    dates = pd.date_range("2020-01-01", periods=6, freq="MS")
    panel = pd.DataFrame({"series_id": ["a"] * 6 + ["b"] * 6, "date": list(dates) * 2,
                          "y": [1, 1, np.nan, 1, 1, 1] + [2] * 6})
    static = pd.DataFrame({"series_id": ["a", "b"], "target": "T"})
    truth = data.actuals(panel, static)
    assert pd.Timestamp("2020-03-01") not in set(truth["date"])
    assert len(truth) == 5


def test_an_unfinished_last_month_is_flagged():
    billing = _billing([100] * 10 + [0])
    billing.loc[(billing["Empresa"] == "B") & (billing["date"] == billing["date"].max()), "clients"] = 0
    gaps = data.reporting_gaps(billing)
    assert gaps["incomplete_month"].all() and len(gaps) == 2
