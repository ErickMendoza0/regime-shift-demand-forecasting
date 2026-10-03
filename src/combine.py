"""Forecast combinations and switching rules built from stored forecasts.

At an origin a rule may use only what is already observed: the one-step-ahead
errors of every model up to the month before the origin.

  ras          switch from a default to a fallback model when the default's
               recent one-step error exceeds `factor` times its earlier average
  bocpd        the same switch, triggered by change-point detection on the
               default's one-step errors
  fixed_share  exponentially weighted average with fixed share (Herbster and
               Warmuth, 1998) over every deployable model
  median_all   median of every deployable model
  comb3        equal-weight mean of seasonal naive, ETS and LightGBM

Default and fallback models and the free parameters of each rule are chosen on
the first SELECT origins of each block, scoring only target months inside that
window, and then frozen.

    python -m src.combine --dataset ecuador
"""
import argparse
import itertools

import numpy as np
import pandas as pd

import config as C
from src import changepoints, data, evaluate, forecast

SELECT = 24
RAS_FACTORS = [1.1, 1.25, 1.5, 2.0, 3.0]
RAS_WINDOWS = [1, 2, 3]
BOCPD_THRESHOLDS = [0.3, 0.5, 0.7]
FS_ETAS = [0.5, 1.0, 2.0, 5.0]
FS_SHARES = [0.0, 0.01, 0.05, 0.1]
SIMPLE = {"simple", "statistical"}
RULES = ("ras", "bocpd", "fixed_share", "median_all", "comb3")


def _family(model):
    return C.MODELS[model.rsplit("_x", 1)[0]]["family"]


def _blocks(dataset):
    out = []
    for start, end in C.DATASETS[dataset]["origins"]:
        out.append((pd.Timestamp(start), pd.Timestamp(end) if end else pd.Timestamp.max))
    return out


def _table(dataset):
    """Seed-median forecasts with truth and MASE scale, deployable models only."""
    preds = evaluate.collect(dataset)
    preds = preds[~preds["model"].isin(RULES) & ~preds["model"].str.endswith("_x3")]
    keys = ["model", "target", "origin", "date", "horizon"]
    df = preds.groupby(keys, as_index=False)["y_hat"].median()
    panel, static = data.load(dataset)
    truth = data.actuals(panel, static)
    df = df.merge(truth, on=["target", "date"], how="left")
    scales = {o: forecast.mase_scale(truth, o) for o in df["origin"].unique()}
    df["scale"] = [scales[o][t] for o, t in zip(df["origin"], df["target"])]
    df["sae"] = (df["y"] - df["y_hat"]).abs() / df["scale"]
    return df


def _one_step(df):
    """Scaled one-step error of every model, indexed by (target, month)."""
    e = df[df["horizon"] == 1].pivot_table(index=["target", "date"], columns="model", values="sae")
    return e


def _score(fc, df, start, stop):
    """Mean MASE of combined forecasts with origin and target inside [start, stop)."""
    sub = fc[(fc["origin"] >= start) & (fc["origin"] < stop) & (fc["date"] < stop)]
    sub = sub.merge(df[["target", "origin", "date", "y", "scale"]].drop_duplicates(),
                    on=["target", "origin", "date"])
    return float(((sub["y"] - sub["y_hat"]).abs() / sub["scale"]).mean())


def _pick(df, start, stop, families):
    sub = df[(df["origin"] >= start) & (df["origin"] < stop) & (df["date"] < stop)]
    sub = sub[sub["model"].map(_family).isin(families)]
    return sub.groupby("model")["sae"].mean().idxmin()


def _switch(df, e1, default, fallback, use_fallback):
    """Forecasts of `default`, replaced by `fallback` at origins flagged by
    use_fallback(target, origin, errors_so_far)."""
    F = df[df["model"].isin([default, fallback])]
    out = []
    for (target, origin), g in F.groupby(["target", "origin"]):
        past = e1.loc[target][default] if target in e1.index.get_level_values(0) else pd.Series()
        past = past[past.index < origin].dropna()
        chosen = fallback if use_fallback(past) else default
        out.append(g[g["model"] == chosen])
    return pd.concat(out)[["target", "origin", "date", "horizon", "y_hat"]]


def _ras(factor, window):
    def rule(past):
        if len(past) <= window:
            return False
        return past.iloc[-window:].mean() > factor * past.iloc[:-window].mean()
    return rule


def _bocpd(threshold):
    def rule(past):
        if len(past) < 6:
            return False
        run = changepoints.bocpd(past.to_numpy(float))
        return run[-1, :3].sum() > threshold
    return rule


def _fixed_share(df, e1, eta, share):
    models = sorted(df["model"].unique())
    out = []
    for target, gt in df.groupby("target"):
        losses = e1.loc[target].reindex(columns=models) if target in e1.index.get_level_values(0) else None
        for origin, g in gt.groupby("origin"):
            w = np.full(len(models), 1 / len(models))
            if losses is not None:
                for _, row in losses[losses.index < origin].iterrows():
                    l = row.to_numpy(float)
                    ok = np.isfinite(l)
                    v = w * np.where(ok, np.exp(-eta * np.where(ok, l, 0)), 1.0)
                    w = (1 - share) * v / v.sum() + share / len(models)
            piv = g.pivot_table(index=["date", "horizon"], columns="model", values="y_hat")
            piv = piv.reindex(columns=models)
            avail = piv.notna().to_numpy()
            ww = np.where(avail, w, 0)
            ww = ww / ww.sum(axis=1, keepdims=True)
            yhat = np.nansum(piv.to_numpy() * ww, axis=1)
            out.append(pd.DataFrame({"target": target, "origin": origin,
                                     "date": piv.index.get_level_values("date"),
                                     "horizon": piv.index.get_level_values("horizon"),
                                     "y_hat": yhat}))
    return pd.concat(out, ignore_index=True)


def _save(dataset, name, fc):
    fc = fc.assign(model=name, seed=0)
    path = C.PREDS / dataset / name / "combined.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(path, index=False)


def run(dataset: str) -> None:
    df = _table(dataset)
    e1 = _one_step(df)
    out_dir = C.TABLES / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    pieces = {r: [] for r in RULES}
    choices, grid_rows = [], []
    for start, end in _blocks(dataset):
        block = df[(df["origin"] >= start) & (df["origin"] <= end)]
        if block.empty:
            continue
        origins = sorted(block["origin"].unique())
        stop = origins[min(SELECT, len(origins) - 1)]
        default = _pick(block, start, stop, set(_family(m) for m in block["model"]) - SIMPLE)
        fallback = _pick(block, start, stop, SIMPLE)

        best = None
        for f, w in itertools.product(RAS_FACTORS, RAS_WINDOWS):
            fc = _switch(block, e1, default, fallback, _ras(f, w))
            sel, full = _score(fc, block, start, stop), _score(fc, block, stop, pd.Timestamp.max)
            grid_rows.append({"block": f"{start:%Y-%m}", "factor": f, "window": w,
                              "mase_selection": sel, "mase_after": full})
            if best is None or sel < best[0]:
                best = (sel, fc, f, w)
        pieces["ras"].append(best[1])
        ras_choice = {"factor": best[2], "window": best[3]}

        best = None
        for thr in BOCPD_THRESHOLDS:
            fc = _switch(block, e1, default, fallback, _bocpd(thr))
            sel = _score(fc, block, start, stop)
            if best is None or sel < best[0]:
                best = (sel, fc, thr)
        pieces["bocpd"].append(best[1])
        bocpd_choice = best[2]

        best = None
        for eta, share in itertools.product(FS_ETAS, FS_SHARES):
            fc = _fixed_share(block, e1, eta, share)
            sel = _score(fc, block, start, stop)
            if best is None or sel < best[0]:
                best = (sel, fc, eta, share)
        pieces["fixed_share"].append(best[1])

        keys = ["target", "origin", "date", "horizon"]
        pieces["median_all"].append(block.groupby(keys, as_index=False)["y_hat"].median())
        trio = block[block["model"].isin(["snaive", "ets", "lgbm"])]
        pieces["comb3"].append(trio.groupby(keys, as_index=False)["y_hat"].mean())

        choices.append({"block": f"{start:%Y-%m}", "selection_end": f"{stop:%Y-%m}",
                        "default": default, "fallback": fallback, **{f"ras_{k}": v for k, v in ras_choice.items()},
                        "bocpd_threshold": bocpd_choice, "fs_eta": best[2], "fs_share": best[3]})
    for name, parts in pieces.items():
        _save(dataset, name, pd.concat(parts, ignore_index=True))
    pd.DataFrame(choices).to_csv(out_dir / "combine_choices.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(out_dir / "ras_grid.csv", index=False)
    print(pd.DataFrame(choices).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    run(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
