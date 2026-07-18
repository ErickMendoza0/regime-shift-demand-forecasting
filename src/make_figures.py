"""Paper figures built from the stored predictions and tables.

  fig_predictions_stable.pdf   selected models vs. truth, fold 2023
  fig_predictions_crisis.pdf   selected models vs. truth, fold 2024
  fig_mape_by_fold.pdf         MAPE per model across folds
  fig_cd_ranks.pdf             mean-rank plot (Friedman/Nemenyi)

    python -m src.make_figures
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C
from src import common

plt.rcParams.update({"font.family": "serif", "font.size": 11})

SHOW = ["seasonal_naive", "sarima_dis", "lgbm", "lstm", "nbeats", "patchtst",
        "ras_switch"]


def _plot_fold(df, fold, path):
    sub = df[(df.fold == fold) & (df.model.isin(SHOW))]
    if sub.empty:
        return
    truth = sub.drop_duplicates("date").sort_values("date")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(truth.date, truth.y_true, "k-o", lw=2, label="Observed")
    for m, g in sub.groupby("model"):
        g = g.sort_values("date")
        ax.plot(g.date, g.y_pred, marker=".", lw=1.2, label=m)
    ax.set_ylabel("National demand (GWh)")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main() -> None:
    df = common.load_all_predictions()
    _plot_fold(df, 2023, C.FIGURES / "fig_predictions_stable.pdf")
    _plot_fold(df, 2024, C.FIGURES / "fig_predictions_crisis.pdf")

    by_fold = pd.read_csv(C.TABLES / "metrics_by_fold.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for m, g in by_fold.groupby("model"):
        ax.plot(g.fold, g.MAPE, marker="o", label=m)
    ax.set_xlabel("Test year (rolling origin)")
    ax.set_ylabel("MAPE (%)")
    ax.legend(fontsize=7, ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_mape_by_fold.pdf", bbox_inches="tight")
    plt.close(fig)

    ranks_path = C.TABLES / "mean_ranks.csv"
    if ranks_path.exists():
        ranks = pd.read_csv(ranks_path, index_col=0)["mean_rank"].sort_values()
        fig, ax = plt.subplots(figsize=(6, 0.35 * len(ranks) + 1))
        ax.barh(ranks.index, ranks.values, color="#5e8ab4", edgecolor="k", lw=0.5)
        ax.set_xlabel("Mean rank (lower is better)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(C.FIGURES / "fig_cd_ranks.pdf", bbox_inches="tight")
        plt.close(fig)

    print(f"figures written to {C.FIGURES}")


if __name__ == "__main__":
    main()
