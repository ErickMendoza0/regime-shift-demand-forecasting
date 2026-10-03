"""Date the regime shifts of every target from its own data.

The statistic is year-on-year growth, which removes the seasonal cycle and the
trend and leaves shocks as shifts in its mean. Three detectors run on it:

  segmentation  optimal partition in mean (Bai-Perron style dynamic programming),
                number of breaks chosen by BIC, with block-bootstrap intervals
  pelt          PELT with a BIC-type penalty, as a cross-check
  bocpd         Bayesian online change-point detection (Adams and MacKay, 2007),
                which also tells how many months a real-time detector needed

Regimes are read off the segmentation: a segment whose mean growth is at or
below SHIFT_THRESHOLD is a shift; a segment that starts within 13 months of the
start of a shift and whose mean is above the median of all segments is a
rebound (mostly the base effect of the year-on-year ratio twelve months after
the shock); every other month is stable. These labels are used only to
group forecast errors afterwards, never as model inputs.

    python -m src.changepoints --dataset ecuador
"""
import argparse

import numpy as np
import pandas as pd
from scipy import special

import config as C
from src import data

SHIFT_THRESHOLD = -2.0      # percent, mean year-on-year growth of a segment
MAX_BREAKS = 8
MIN_SEGMENT = 3
HAZARD = 1 / 60             # BOCPD prior: one regime change every five years
BOOT = 500
BLOCK = 6


def yoy(y: pd.Series) -> pd.Series:
    return (100 * (y / y.shift(C.SEASON) - 1)).dropna()


def segment(x: np.ndarray):
    """Breaks (as end indices) of the BIC-optimal partition in mean."""
    import ruptures as rpt
    n = len(x)
    algo = rpt.Dynp(model="l2", min_size=MIN_SEGMENT, jump=1).fit(x)
    best, best_bic = [n], n * np.log(np.var(x)) + np.log(n)
    for k in range(1, MAX_BREAKS + 1):
        try:
            bk = algo.predict(n_bkps=k)
        except Exception:
            break
        ssr = sum(np.sum((s - s.mean()) ** 2) for s in np.split(x, bk[:-1]))
        bic = n * np.log(ssr / n) + (2 * k + 1) * np.log(n)
        if bic < best_bic:
            best, best_bic = bk, bic
    return best


def pelt(x: np.ndarray):
    import ruptures as rpt
    sigma2 = (np.median(np.abs(np.diff(x))) / 0.6745) ** 2 / 2
    return rpt.Pelt(model="l2", min_size=MIN_SEGMENT).fit(x).predict(pen=3 * sigma2 * np.log(len(x)))


def bootstrap_intervals(x: np.ndarray, breaks, rng) -> list[tuple[int, int]]:
    """Percentile intervals for each break from a moving-block bootstrap of the
    residuals around the fitted segment means, re-segmented with the same k."""
    import ruptures as rpt
    k = len(breaks) - 1
    if k == 0:
        return []
    parts = np.split(np.arange(len(x)), breaks[:-1])
    fitted = np.concatenate([np.full(len(p), x[p].mean()) for p in parts])
    resid = x - fitted
    draws = []
    for _ in range(BOOT):
        starts = rng.integers(0, len(x) - BLOCK + 1, size=len(x) // BLOCK + 1)
        r = np.concatenate([resid[s:s + BLOCK] for s in starts])[:len(x)]
        bk = rpt.Dynp(model="l2", min_size=MIN_SEGMENT, jump=1).fit(fitted + r).predict(n_bkps=k)
        draws.append(bk[:-1])
    draws = np.array(draws)
    return [(int(np.percentile(draws[:, j], 2.5)), int(np.percentile(draws[:, j], 97.5)))
            for j in range(k)]


def bocpd(x: np.ndarray, hazard: float = HAZARD) -> np.ndarray:
    """Run-length posterior P(r_t = r | x_1..t) under a Gaussian model with
    unknown mean and variance (normal-gamma prior)."""
    n = len(x)
    mu0, kappa0, alpha0, beta0 = np.median(x), 1.0, 1.0, np.var(x)
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1.0
    mu, kappa = np.array([mu0]), np.array([kappa0])
    alpha, beta = np.array([alpha0]), np.array([beta0])
    for t, xt in enumerate(x):
        scale = np.sqrt(beta * (kappa + 1) / (alpha * kappa))
        df = 2 * alpha
        z = (xt - mu) / scale
        logpdf = (special.gammaln((df + 1) / 2) - special.gammaln(df / 2)
                  - 0.5 * np.log(df * np.pi) - np.log(scale)
                  - (df + 1) / 2 * np.log1p(z ** 2 / df))
        pred = np.exp(logpdf)
        growth = R[t, :t + 1] * pred * (1 - hazard)
        change = np.sum(R[t, :t + 1] * pred * hazard)
        R[t + 1, 1:t + 2] = growth
        R[t + 1, 0] = change
        R[t + 1] /= R[t + 1].sum()
        mu_new = (kappa * mu + xt) / (kappa + 1)
        beta_new = beta + kappa * (xt - mu) ** 2 / (2 * (kappa + 1))
        mu = np.concatenate([[mu0], mu_new])
        kappa = np.concatenate([[kappa0], kappa + 1])
        alpha = np.concatenate([[alpha0], alpha + 0.5])
        beta = np.concatenate([[beta0], beta_new])
    return R[1:]


def regimes(dates, x, breaks) -> pd.DataFrame:
    seg_id = np.zeros(len(x), dtype=int)
    for j, b in enumerate(breaks[:-1]):
        seg_id[b:] = j + 1
    means = pd.Series(x).groupby(seg_id).mean()
    starts = pd.Series(dates).groupby(seg_id).min()
    median = means.median()
    label, shift_starts = {}, []
    for j, m in means.items():
        after_shift = any(0 < (starts[j] - s).days <= 13 * 31 for s in shift_starts)
        if m <= SHIFT_THRESHOLD:
            label[j] = "shift"
            shift_starts.append(starts[j])
        elif after_shift and m > median:
            label[j] = "rebound"
        else:
            label[j] = "stable"
    return pd.DataFrame({"date": dates, "yoy": x, "segment": seg_id,
                         "segment_mean": means.reindex(seg_id).to_numpy(),
                         "regime": [label[j] for j in seg_id]})


def _one_target(target, g, seed):
    """Breaks, intervals and regime labels for one target."""
    from statsmodels.stats.diagnostic import breaks_cusumolsresid

    rng = np.random.default_rng(seed)
    rows = []
    y = g.set_index("date")["y"].asfreq("MS")
    growth = yoy(y)
    dates, x = growth.index, growth.to_numpy(float)
    bk = segment(x)
    ci = bootstrap_intervals(x, bk, rng)
    pk = pelt(x)
    run = bocpd(x)
    p_recent = run[:, :3].sum(axis=1)       # P(change within the last 3 months)
    cusum_stat, cusum_p, _ = breaks_cusumolsresid(x - x.mean())
    for j, b in enumerate(bk[:-1]):
        lo, hi = ci[j]
        first_alarm = next((dates[t] for t in range(b, len(x)) if p_recent[t] > 0.5), None)
        rows.append({"target": target, "method": "segmentation", "break": dates[b],
                     "ci_low": dates[lo], "ci_high": dates[min(hi, len(x) - 1)],
                     "bocpd_alarm": first_alarm})
    for b in pk[:-1]:
        rows.append({"target": target, "method": "pelt", "break": dates[b]})
    rows.append({"target": target, "method": "cusum", "statistic": cusum_stat,
                 "p_value": cusum_p})
    reg = regimes(dates, x, bk)
    reg["target"] = target
    reg["bocpd_p_recent"] = p_recent
    return rows, reg


def analyse(dataset: str) -> None:
    import os
    from joblib import Parallel, delayed

    panel, static = data.load(dataset)
    truth = data.actuals(panel, static)
    jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    out = Parallel(n_jobs=jobs)(delayed(_one_target)(t, g, i)
                                for i, (t, g) in enumerate(truth.groupby("target")))
    rows = [r for part, _ in out for r in part]
    labels = [reg for _, reg in out]
    C.TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(C.TABLES / f"changepoints_{dataset}.csv", index=False)
    pd.concat(labels).to_csv(C.TABLES / f"regimes_{dataset}.csv", index=False)
    print(pd.DataFrame(rows).query("method == 'segmentation'").to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    analyse(ap.parse_args().dataset)


if __name__ == "__main__":
    main()
