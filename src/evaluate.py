"""Collect forecasts, score them by regime and run the comparison tests.

    python -m src.evaluate --dataset ecuador

Seeded models are represented by the median of their seeds' forecasts; the
spread across seeds is reported separately. Errors are scaled per target and
origin by the in-sample seasonal-naive MAE (MASE), so targets of different
size can be pooled.

Tables written to work/tables/<dataset>/:
  accuracy.csv       MASE, MAPE, RMSE, MAE and bias by model and regime
  by_horizon.csv     MASE by model, regime and horizon
  degradation.csv    stable-to-shift error ratio per model, with intervals
  on_the_line.csv    rank correlation of stable and shift accuracy
  seeds.csv          spread of MASE across seeds
  dm_<regime>.csv    all pairwise Diebold-Mariano tests, Holm and BY adjusted
  mcs_<regime>.csv   model confidence set
  gw.csv             Giacomini-White tests of regime-dependent performance
  fluctuation.csv    Giacomini-Rossi fluctuation paths for key pairs
  episodes.csv       mean MASE per (target, shift episode), the blocks for E9
  sensitivity.csv    Ecuador only: MASE and MCS membership under the four
                     drought windows fixed in docs/preregistration.md
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from src import data, forecast, inference

LAG = 11                     # target-month losses pool horizons 1..12
# With fewer target months than this, a HAC variance with lag 11 or a block
# bootstrap means nothing, so DM and MCS are not run within that regime.
MIN_TEST_MONTHS = 24
ORACLE = "_x3"               # hindsight ONI tier, kept out of every ranking

# Alternative definitions of the 2024 drought window (preregistered). The
# "detected" window is whatever the change-point analysis labels as shift.
DROUGHT_WINDOWS = {"all 2024": ("2024-01", "2024-12"),
                   "Sep-Dec 2024": ("2024-09", "2024-12"),
                   "Oct 2023-Dec 2024": ("2023-10", "2024-12")}


def collect(dataset: str) -> pd.DataFrame:
    files = sorted((C.PREDS / dataset).glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no forecasts under {C.PREDS / dataset}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.to_parquet(C.PREDS / f"{dataset}.parquet", index=False)
    return df


def errors(dataset: str, preds: pd.DataFrame) -> pd.DataFrame:
    panel, static = data.load(dataset)
    truth = data.actuals(panel, static)
    keys = ["model", "target", "origin", "date", "horizon"]
    med = preds.groupby(keys, as_index=False)["y_hat"].median()
    df = med.merge(truth, on=["target", "date"])
    scales = {o: forecast.mase_scale(truth, o) for o in df["origin"].unique()}
    df["scale"] = [scales[o][t] for o, t in zip(df["origin"], df["target"])]
    df["err"] = df["y"] - df["y_hat"]
    df["ae"] = df["err"].abs()
    df["se"] = df["err"] ** 2
    df["ape"] = 100 * df["ae"] / df["y"]
    df["sae"] = df["ae"] / df["scale"]
    reg = pd.read_csv(C.TABLES / f"regimes_{dataset}.csv", parse_dates=["date"])
    df = df.merge(reg[["target", "date", "regime"]], on=["target", "date"], how="left")
    df["regime"] = df["regime"].fillna("stable")
    return df


def accuracy(df: pd.DataFrame, by) -> pd.DataFrame:
    """Error measures by group. Every target month weighs the same, as in the
    paper: errors are first averaged over the forecasts of that month."""
    keys = list(by) + [k for k in ("target", "date") if k not in by]
    per_month = df.groupby(keys)[["sae", "ape", "se", "ae", "err"]].mean()
    g = per_month.groupby(level=list(by))
    return pd.DataFrame({
        "MASE": g["sae"].mean(), "MAPE": g["ape"].mean(), "RMSE": np.sqrt(g["se"].mean()),
        "MAE": g["ae"].mean(), "bias": g["err"].mean(), "months": g.size(),
        "n": df.groupby(by).size(),
    }).reset_index()


def month_losses(df: pd.DataFrame, regime=None) -> pd.DataFrame:
    """Scaled error per (target, target month), averaged over every forecast of
    that month, as a T x models frame."""
    sub = df if regime is None else df[df["regime"] == regime]
    L = sub.groupby(["target", "date", "model"])["sae"].mean().unstack("model")
    return L.dropna(axis=1, how="any")


def degradation(df: pd.DataFrame) -> pd.DataFrame:
    L = df.groupby(["target", "date", "regime", "model"])["sae"].mean().unstack("model")
    L = L.reset_index(level="regime")
    rows = []
    for m in L.columns.drop("regime"):
        sub = L[["regime", m]].dropna()

        def ratio(frame, m=m):
            s = frame.loc[frame["regime"] == "stable", m].mean()
            k = frame.loc[frame["regime"] == "shift", m].mean()
            return k / s

        st = sub.loc[sub["regime"] == "stable", m].mean()
        sh = sub.loc[sub["regime"] == "shift", m].mean()
        lo, hi = inference.stationary_bootstrap_ci(ratio, sub.reset_index(drop=True))
        rows.append({"model": m, "mase_stable": st, "mase_shift": sh, "ratio": sh / st,
                     "ratio_low": lo, "ratio_high": hi, "difference": sh - st})
    out = pd.DataFrame(rows)
    out["rank_stable"] = out["mase_stable"].rank()
    out["rank_shift"] = out["mase_shift"].rank()
    out["rank_change"] = out["rank_shift"] - out["rank_stable"]
    return out.sort_values("ratio")


def on_the_line(deg: pd.DataFrame) -> pd.DataFrame:
    rho, p = stats.spearmanr(deg["mase_stable"], deg["mase_shift"])
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        s = deg.sample(len(deg), replace=True, random_state=int(rng.integers(1e9)))
        boots.append(stats.spearmanr(s["mase_stable"], s["mase_shift"])[0])
    return pd.DataFrame([{"spearman": rho, "p": p, "low": np.nanpercentile(boots, 2.5),
                          "high": np.nanpercentile(boots, 97.5), "models": len(deg)}])


def episodes(df: pd.DataFrame) -> pd.DataFrame:
    """Mean MASE of every model over each contiguous run of shift months."""
    out = []
    for target, g in df[df["regime"] == "shift"].groupby("target"):
        months = np.sort(g["date"].unique())
        run_id = np.cumsum(np.r_[1, np.diff(months).astype("timedelta64[D]").astype(int) > 31])
        ep = dict(zip(months, run_id))
        g = g.assign(episode=g["date"].map(ep))
        m = (g.groupby(["episode", "model", "date"])["sae"].mean()
              .groupby(["episode", "model"]).mean().unstack("model"))
        first = g.groupby("episode")["date"].min()
        m.index = [f"{target}:{first[e]:%Y-%m}" for e in m.index]
        out.append(m)
    return pd.concat(out) if out else pd.DataFrame()


def sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    detected = df[(df["regime"] == "shift") & (df["date"] >= "2023-01-01")]["date"]
    windows = {**DROUGHT_WINDOWS, "detected": (f"{detected.min():%Y-%m}", f"{detected.max():%Y-%m}")}
    rows = []
    for name, (a, b) in windows.items():
        sub = df[(df["date"] >= a) & (df["date"] <= pd.Timestamp(b))]
        L = sub.groupby(["target", "date", "model"])["sae"].mean().unstack("model").dropna(axis=1)
        mcs = inference.mcs(L) if len(L) >= 3 else None
        for m, v in L.mean().items():
            rows.append({"window": name, "start": a, "end": b, "months": len(L), "model": m,
                         "MASE": v, "in_mcs": None if mcs is None else bool(mcs.loc[m, "in_mcs"])})
    out = pd.DataFrame(rows)
    out["rank"] = out.groupby("window")["MASE"].rank()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ds = ap.parse_args().dataset
    out = C.TABLES / ds
    out.mkdir(parents=True, exist_ok=True)

    preds = collect(ds)
    df = errors(ds, preds)
    df.to_parquet(out / "errors.parquet", index=False)

    accuracy(df, ["model", "regime"]).to_csv(out / "accuracy.csv", index=False)
    accuracy(df, ["model"]).assign(regime="all").to_csv(out / "accuracy_all.csv", index=False)
    accuracy(df, ["model", "regime", "horizon"]).to_csv(out / "by_horizon.csv", index=False)

    keys = ["model", "target", "origin", "date"]
    per_seed = preds.merge(df[keys + ["y", "scale"]], on=keys)
    per_seed["sae"] = (per_seed["y"] - per_seed["y_hat"]).abs() / per_seed["scale"]
    (per_seed.groupby(["model", "seed", "target", "date"])["sae"].mean()
             .groupby(["model", "seed"]).mean().groupby("model")
             .agg(["mean", "std", "min", "max", "count"]).to_csv(out / "seeds.csv"))

    deployable = df[~df["model"].str.endswith(ORACLE)]
    deg = degradation(deployable)
    deg.to_csv(out / "degradation.csv", index=False)
    on_the_line(deg).to_csv(out / "on_the_line.csv", index=False)

    for regime in ("stable", "shift", "rebound", None):
        L = month_losses(deployable, regime)
        if len(L) < MIN_TEST_MONTHS:
            print(f"{regime}: {len(L)} target months, too few for DM and MCS")
            continue
        name = regime or "all"
        inference.pairwise_dm(L, LAG).to_csv(out / f"dm_{name}.csv", index=False)
        inference.mcs(L).to_csv(out / f"mcs_{name}.csv")

    # Does relative performance depend on the regime? One test per model
    # against the seasonal-naive reference, and fluctuation paths over time.
    L = month_losses(deployable)
    shift = deployable.groupby(["target", "date"])["regime"].first().eq("shift").astype(float)
    gw, fl = [], []
    for m in L.columns.drop("snaive", errors="ignore"):
        d = L[m] - L["snaive"]
        stat, p = inference.giacomini_white(d, shift.reindex(d.index), LAG)
        gw.append({"model": m, "reference": "snaive", "gw": stat, "p": p})
        for target in d.index.get_level_values("target").unique():
            dt = d.xs(target, level="target")
            if len(dt) >= 36:
                f = inference.fluctuation(dt, window=12, lag=LAG)
                fl.append(f.assign(model=m, target=target).reset_index())
    gw = pd.DataFrame(gw)
    from statsmodels.stats.multitest import multipletests
    gw["p_holm"] = multipletests(gw["p"], method="holm")[1]
    gw.to_csv(out / "gw.csv", index=False)
    if fl:
        pd.concat(fl).to_csv(out / "fluctuation.csv", index=False)

    if ds == "ecuador":
        sensitivity(deployable).to_csv(out / "sensitivity.csv", index=False)

    ep = episodes(deployable)
    if not ep.empty:
        ep.to_csv(out / "episodes.csv")
    print(f"tables written to {out}")


if __name__ == "__main__":
    main()
