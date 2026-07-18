"""Error-metric tables and statistical comparison of the models.

Outputs (work/results/tables/):
  metrics_by_fold.csv     RMSE/MAE/MAPE/MASE per model x fold
  metrics_by_regime.csv   pooled stable (2021-23) vs. crisis (2024) vs. all
  dm_tests.csv            pairwise Diebold-Mariano with HLN small-sample correction
  friedman.csv            Friedman chi-square and p-value
  nemenyi_matrix.csv      Nemenyi post-hoc p-value matrix
  mean_ranks.csv          mean rank per model

    python -m src.stats_tests
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from src import common


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1, power: int = 2):
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample
    correction. Returns (statistic, p_value); a positive statistic means the
    first model has the larger loss."""
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dbar = d.mean()
    # Autocovariance up to h-1 lags.
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        return np.nan, np.nan
    dm = dbar / np.sqrt(var_d)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    return float(hln), float(p)


def _pools(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "stable": df[df.fold.isin(C.STABLE_YEARS)],
        "crisis": df[df.fold.isin(C.CRISIS_YEARS)],
        "all": df,
    }


def main() -> None:
    df = common.load_all_predictions()
    national = common.load_national()

    # Per-fold metric table.
    rows = []
    for (m, f), g in df.groupby(["model", "fold"]):
        tr_idx, _ = common.split_fold(national.index, int(f))
        y_train = national.loc[tr_idx].values
        rows.append({"model": m, "fold": f,
                     **{k: fn(g.y_true, g.y_pred) for k, fn in common.ALL_METRICS.items()},
                     "MASE": common.mase(g.y_true, g.y_pred, y_train)})
    by_fold = pd.DataFrame(rows).sort_values(["model", "fold"])
    by_fold.to_csv(C.TABLES / "metrics_by_fold.csv", index=False)

    # Pooled regime metrics.
    rows = []
    for pool, sub in _pools(df).items():
        for m, g in sub.groupby("model"):
            rows.append({"pool": pool, "model": m, "n_months": len(g),
                         **{k: fn(g.y_true, g.y_pred) for k, fn in common.ALL_METRICS.items()}})
    pd.DataFrame(rows).to_csv(C.TABLES / "metrics_by_regime.csv", index=False)

    # Pairwise Diebold-Mariano within each pool.
    rows = []
    for pool, sub in _pools(df).items():
        piv = sub.pivot_table(index="date", columns="model",
                              values=["y_true", "y_pred"])
        models = piv["y_pred"].columns.tolist()
        for m1, m2 in itertools.combinations(models, 2):
            common_idx = piv["y_pred"][[m1, m2]].dropna().index
            e1 = (piv.loc[common_idx, ("y_true", m1)]
                  - piv.loc[common_idx, ("y_pred", m1)]).values
            e2 = (piv.loc[common_idx, ("y_true", m2)]
                  - piv.loc[common_idx, ("y_pred", m2)]).values
            stat, p = dm_test(e1, e2, h=1)
            rows.append({"pool": pool, "model_1": m1, "model_2": m2,
                         "n": len(common_idx), "DM_HLN": stat, "p_value": p})
    pd.DataFrame(rows).to_csv(C.TABLES / "dm_tests.csv", index=False)

    # Friedman test plus Nemenyi post-hoc over the monthly absolute errors.
    # Only models evaluated on every month enter the test (complete blocks).
    coverage = df.groupby("model")["date"].nunique()
    full = coverage[coverage == coverage.max()].index.tolist()
    piv = df[df.model.isin(full)].pivot_table(index="date", columns="model",
                                              values="y_pred")
    truth = df.pivot_table(index="date", values="y_true", aggfunc="first")
    abs_err = (piv.sub(truth["y_true"], axis=0)).abs().dropna()
    if abs_err.shape[1] >= 3 and len(abs_err) >= 10:
        fried_stat, fried_p = stats.friedmanchisquare(
            *[abs_err[c].values for c in abs_err.columns])
        out = [{"test": "friedman", "statistic": fried_stat, "p_value": fried_p,
                "n_blocks": len(abs_err), "k_models": abs_err.shape[1]}]
        try:
            import scikit_posthocs as sp
            nem = sp.posthoc_nemenyi_friedman(abs_err.values)
            nem.index = nem.columns = abs_err.columns
            nem.to_csv(C.TABLES / "nemenyi_matrix.csv")
        except ImportError:
            print("scikit-posthocs not installed; skipping Nemenyi post-hoc")
        ranks = abs_err.rank(axis=1).mean().sort_values()
        ranks.to_csv(C.TABLES / "mean_ranks.csv", header=["mean_rank"])
        pd.DataFrame(out).to_csv(C.TABLES / "friedman.csv", index=False)
    else:
        print(f"Friedman skipped: {abs_err.shape[1]} fully-covered models, "
              f"{len(abs_err)} blocks (need at least 3 and 10)")

    print(f"tables written to {C.TABLES}")


if __name__ == "__main__":
    main()
