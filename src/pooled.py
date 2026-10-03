"""Compare models across every shift episode of the three datasets.

Each (target, episode) is one independent block: its mean MASE per model comes
from src.evaluate (episodes.csv). Only models run on all datasets enter, so the
ONI variants are left out.

Outputs in work/tables/pooled/:
  episodes.csv     MASE per episode and model, with episode depth and dataset
  friedman.csv     Friedman statistic, p-value and Nemenyi critical difference
  ranks.csv        mean rank of every model over the episodes
  nemenyi.csv      Nemenyi p-values
  bayes.csv        Bayesian signed-rank test of every model against the seasonal
                   naive and the 3-year seasonal mean (ROPE 1 %)
  depth.csv        exploratory: does the gap between learned and classical
                   models grow with the depth of the shift? (not preregistered)

    python -m src.pooled
"""
import numpy as np
import pandas as pd
from scipy import stats

import config as C
from src import inference
from src import plot_style as S

DATASETS = ("ecuador", "brazil", "europe")
REFERENCES = ("snaive", "swa3")


def load() -> pd.DataFrame:
    parts = []
    for d in DATASETS:
        path = C.TABLES / d / "episodes.csv"
        if not path.exists():
            continue
        e = pd.read_csv(path, index_col=0)
        reg = pd.read_csv(C.TABLES / f"regimes_{d}.csv", parse_dates=["date"])
        depth = {}
        for label in e.index:
            target, start = label.split(":")
            row = reg[(reg["target"] == target) & (reg["date"] == pd.Timestamp(start))]
            depth[label] = float(row["segment_mean"].iloc[0]) if len(row) else np.nan
        e.insert(0, "depth", pd.Series(depth))
        e.insert(0, "dataset", d)
        parts.append(e)
    df = pd.concat(parts)
    common = [c for c in df.columns if c not in ("dataset", "depth") and df[c].notna().all()]
    return df[["dataset", "depth"] + common]


def main() -> None:
    out = C.TABLES / "pooled"
    out.mkdir(parents=True, exist_ok=True)
    df = load()
    df.to_csv(out / "episodes.csv")
    X = df.drop(columns=["dataset", "depth"])

    summary, ranks, nem = inference.friedman_nemenyi(X)
    pd.DataFrame([summary]).to_csv(out / "friedman.csv", index=False)
    ranks.rename("mean_rank").to_csv(out / "ranks.csv")
    nem.to_csv(out / "nemenyi.csv")

    rows = []
    for ref in REFERENCES:
        for m in X.columns.drop(ref):
            r = inference.bayes_signed_rank(X[m].to_numpy(), X[ref].to_numpy())
            rows.append({"model": m, "reference": ref, "blocks": len(X), **r})
    pd.DataFrame(rows).to_csv(out / "bayes.csv", index=False)

    # Exploratory, not in the analysis plan: relative MASE of learned models
    # (boosting, recurrent, deep) over classical ones, against shift depth.
    fam = {m: S.family(m) for m in X.columns}
    learned = [m for m, f in fam.items() if f in ("boosting", "recurrent", "deep")]
    classical = [m for m, f in fam.items() if f in ("simple", "statistical", "decomposition")]
    gap = np.log(X[learned].median(axis=1) / X[classical].median(axis=1))
    rho, p = stats.spearmanr(df["depth"], gap, nan_policy="omit")
    pd.DataFrame({"dataset": df["dataset"], "depth": df["depth"], "log_gap": gap}).to_csv(
        out / "depth.csv")
    print(f"Friedman chi2={summary['chi2']:.2f} p={summary['p']:.2g} over {summary['blocks']} "
          f"episodes, CD={summary['cd']:.2f}; depth vs learned/classical gap: "
          f"Spearman {rho:.2f} (p={p:.3g})")
    print(ranks.round(2).to_string())


if __name__ == "__main__":
    main()
