"""Forecast-comparison tests that account for serial dependence.

Losses arrive as a T x K frame (target months by models). Consecutive target
months share forecast origins, so loss differentials are autocorrelated and
every variance below is a Newey-West (Bartlett) long-run variance.
"""
import itertools
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats


def long_run_variance(d: np.ndarray, lag: int) -> float:
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return float(v)


def dm_test(d: np.ndarray, lag: int):
    """Diebold-Mariano on a loss differential with a HAC variance and the
    Harvey-Leybourne-Newbold correction (horizon taken as lag + 1). A negative
    statistic means the first model has the smaller loss."""
    d = np.asarray(d, float)
    n, h = len(d), lag + 1
    v = long_run_variance(d, lag)
    if n < 3 or v <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(v / n)
    hln = dm * np.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 1) / n)
    return float(hln), float(2 * stats.t.sf(abs(hln), df=n - 1))


def pairwise_dm(L: pd.DataFrame, lag: int) -> pd.DataFrame:
    """All pairs, with Holm and Benjamini-Yekutieli adjusted p-values."""
    from statsmodels.stats.multitest import multipletests
    rows = []
    for a, b in itertools.combinations(L.columns, 2):
        d = (L[a] - L[b]).dropna().to_numpy()
        stat, p = dm_test(d, lag)
        rows.append({"model_1": a, "model_2": b, "n": len(d), "mean_diff": d.mean(),
                     "dm": stat, "p": p})
    out = pd.DataFrame(rows)
    ok = out["p"].notna()
    out.loc[ok, "p_holm"] = multipletests(out.loc[ok, "p"], method="holm")[1]
    out.loc[ok, "p_by"] = multipletests(out.loc[ok, "p"], method="fdr_by")[1]
    return out


def mcs(L: pd.DataFrame, size: float = 0.10, block: int = 6, reps: int = 5000, seed: int = 0):
    """Model confidence set (Hansen, Lunde and Nason, 2011) with a stationary
    bootstrap. Returns each model's MCS p-value and whether it is in the set."""
    from arch.bootstrap import MCS
    L = L.dropna()
    # Models with identical losses (a switching rule that never switched, for
    # instance) make the differences degenerate; test one copy and give the
    # others the same p-value.
    dup = L.round(12).T.duplicated()
    same_as = {c: next(k for k in L.columns[~dup] if np.allclose(L[c], L[k]))
               for c in L.columns[dup]}
    U = L.loc[:, ~dup]
    try:
        m = MCS(U, size=size, reps=reps, block_size=block, method="R",
                bootstrap="stationary", seed=seed)
        m.compute()
        method = "R"
    except (IndexError, ValueError, np.linalg.LinAlgError):
        m = MCS(U, size=size, reps=reps, block_size=block, method="max",
                bootstrap="stationary", seed=seed)
        m.compute()
        method = "max"
    p = m.pvalues["Pvalue"].reindex(L.columns)
    for c, k in same_as.items():
        p[c] = p[k]
    out = pd.DataFrame({"mcs_p": p, "in_mcs": p >= size, "statistic": method,
                        "same_as": pd.Series(same_as).reindex(L.columns)})
    return out.sort_values("mcs_p", ascending=False)


@lru_cache(maxsize=None)
def fluctuation_critical_value(mu: float, alpha: float = 0.05, reps: int = 20000,
                               grid: int = 1000, seed: int = 0) -> float:
    """Critical value of the Giacomini-Rossi (2010) fluctuation test,
    sup_t |B(t) - B(t - mu)| / sqrt(mu), simulated from Brownian paths."""
    rng = np.random.default_rng(seed)
    w = int(round(mu * grid))
    sups = np.empty(reps)
    for i in range(reps):
        b = np.concatenate([[0.0], np.cumsum(rng.standard_normal(grid)) / np.sqrt(grid)])
        sups[i] = np.max(np.abs(b[w:] - b[:-w])) / np.sqrt(mu)
    return float(np.quantile(sups, 1 - alpha))


def fluctuation(d: pd.Series, window: int, lag: int) -> pd.DataFrame:
    """Rolling standardised mean of a loss differential (Giacomini and Rossi,
    2010). Crossing the critical value means the relative performance of the
    two models was not constant over the sample."""
    d = d.dropna()
    n = len(d)
    sigma = np.sqrt(long_run_variance(d.to_numpy(), lag))
    stat = d.rolling(window).sum() / (sigma * np.sqrt(window))
    cv = fluctuation_critical_value(round(window / n, 3))
    return pd.DataFrame({"statistic": stat, "critical": cv}).dropna()


def giacomini_white(d: pd.Series, instrument: pd.Series, lag: int):
    """Conditional predictive ability test (Giacomini and White, 2006) with a
    constant and the given instrument (for example a shift-regime dummy)."""
    df = pd.concat({"d": d, "z": instrument}, axis=1).dropna()
    Z = np.column_stack([np.ones(len(df)), df["z"].to_numpy(float)])
    g = Z * df["d"].to_numpy()[:, None]
    gbar = g.mean(axis=0)
    gc = g - gbar
    n = len(g)
    S = gc.T @ gc / n
    for k in range(1, lag + 1):
        G = gc[k:].T @ gc[:-k] / n
        S += (1 - k / (lag + 1)) * (G + G.T)
    stat = float(n * gbar @ np.linalg.pinv(S) @ gbar)
    return stat, float(stats.chi2.sf(stat, df=Z.shape[1]))


def friedman_nemenyi(X: pd.DataFrame):
    """Friedman test and Nemenyi post-hoc over independent blocks (rows)."""
    import scikit_posthocs as sp
    X = X.dropna()
    chi2, p = stats.friedmanchisquare(*[X[c] for c in X.columns])
    ranks = X.rank(axis=1).mean().sort_values()
    nem = sp.posthoc_nemenyi_friedman(X.to_numpy())
    nem.index = nem.columns = X.columns
    k, n = X.shape[1], X.shape[0]
    q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return {"chi2": chi2, "p": p, "blocks": n, "cd": cd}, ranks, nem


def bayes_signed_rank(x: np.ndarray, y: np.ndarray, rope: float = 0.01, seed: int = 0):
    """Benavoli et al. (2017) Bayesian signed-rank test on relative differences
    across independent blocks: P(x better), P(practically equal), P(y better)."""
    import baycomp
    rel = (np.asarray(x) - np.asarray(y)) / np.asarray(y)
    p = baycomp.SignedRankTest.probs(rel, np.zeros_like(rel), rope=rope, random_state=seed)
    return {"p_first_better": p[0], "p_rope": p[1], "p_second_better": p[2]}


def stationary_bootstrap_ci(stat, data: pd.DataFrame, block: int = 6, reps: int = 2000,
                            seed: int = 0):
    """Percentile interval of `stat(frame)` under a stationary bootstrap of rows."""
    from arch.bootstrap import StationaryBootstrap
    bs = StationaryBootstrap(block, data, seed=seed)
    draws = [stat(d[0][0]) for d in bs.bootstrap(reps)]
    return np.nanpercentile(draws, 2.5), np.nanpercentile(draws, 97.5)
