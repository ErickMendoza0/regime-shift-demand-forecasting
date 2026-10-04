"""Turn the result tables into LaTeX macros and a short Markdown summary.

    python -m src.summary

work/tables/results.tex defines \\res{key}, so the manuscript can write
\\res{ecuador.shift.swa3.mase} instead of copying a number by hand. Every key
is also listed, with its value, in work/tables/results_keys.csv, and the
headline numbers are gathered in work/tables/summary.md.
"""
import json

import numpy as np
import pandas as pd

import config as C
from src import plot_style as S

DATASETS = ("ecuador", "brazil", "europe")
CLASSICAL = {"simple", "statistical", "decomposition"}
LEARNED = {"boosting", "recurrent", "deep"}


def fmt(v, digits=2):
    """Text ready for LaTeX: underscores escaped, tiny numbers as powers of ten."""
    if isinstance(v, str):
        return v.replace("_", r"\_")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return f"{v:d}"
    if abs(v) < 1e-3 and v != 0:
        mantissa, exponent = f"{v:.1e}".split("e")
        return rf"${mantissa}\times10^{{{int(exponent)}}}$"
    text = f"{v:.{digits}f}"
    return "$-$" + text[1:] if text.startswith("-") else text


def collect() -> dict:
    out = {}

    def put(key, value, digits=2):
        out[key] = fmt(value, digits)

    for d in DATASETS:
        t = C.TABLES / d
        deg = pd.read_csv(t / "degradation.csv")
        for _, r in deg.iterrows():
            m = r["model"]
            put(f"{d}.stable.{m}.mase", r["mase_stable"])
            put(f"{d}.shift.{m}.mase", r["mase_shift"])
            put(f"{d}.{m}.ratio", r["ratio"])
            put(f"{d}.{m}.ratiolow", r["ratio_low"])
            put(f"{d}.{m}.ratiohigh", r["ratio_high"])
            put(f"{d}.stable.{m}.rank", int(r["rank_stable"]))
            put(f"{d}.shift.{m}.rank", int(r["rank_shift"]))
        fam = deg["model"].map(S.family)
        for group, members in (("classical", CLASSICAL), ("learned", LEARNED),
                               ("foundation", {"foundation"}), ("combination", {"combination"})):
            put(f"{d}.ratio.median.{group}", deg.loc[fam.isin(members), "ratio"].median())
        put(f"{d}.models", len(deg))
        put(f"{d}.best.stable", deg.loc[deg["mase_stable"].idxmin(), "model"])
        put(f"{d}.best.shift", deg.loc[deg["mase_shift"].idxmin(), "model"])

        acc = pd.read_csv(t / "accuracy.csv")
        for _, r in acc.iterrows():
            for col in ("MAPE", "RMSE", "MAE", "bias"):
                put(f"{d}.{r['regime']}.{r['model']}.{col.lower()}", r[col])

        line = pd.read_csv(t / "on_the_line.csv").iloc[0]
        put(f"{d}.spearman", line["spearman"])
        put(f"{d}.spearmanlow", line["low"])
        put(f"{d}.spearmanhigh", line["high"])
        put(f"{d}.spearmanp", line["p"], 3)

        gw = pd.read_csv(t / "gw.csv")
        put(f"{d}.gw.significant", int((gw["p_holm"] < 0.05).sum()))
        put(f"{d}.gw.tested", len(gw))

        dm = pd.read_csv(t / "dm_all.csv")
        put(f"{d}.dm.pairs", len(dm))
        put(f"{d}.dm.holm", int((dm["p_holm"] < 0.05).sum()))

        ch = pd.read_csv(t / "combine_choices.csv")
        for i, r in ch.iterrows():
            for col in ("default", "fallback", "ras_factor", "ras_window", "bocpd_threshold",
                        "fs_eta", "fs_share"):
                put(f"{d}.block{i + 1}.{col}", r[col])

        err = pd.read_parquet(t / "errors.parquet", columns=["target", "date", "regime"])
        months = err.drop_duplicates(["target", "date"])["regime"].value_counts()
        for regime in ("stable", "shift", "rebound"):
            put(f"{d}.months.{regime}", int(months.get(regime, 0)))

        # H2: are shift/stable ratios larger for learned than for classical models?
        from scipy import stats
        lr = deg.loc[fam.isin(LEARNED), "ratio"]
        cl = deg.loc[fam.isin(CLASSICAL), "ratio"]
        put(f"{d}.h2.p", stats.mannwhitneyu(lr, cl, alternative="greater").pvalue, 3)

        # H4: each ONI model against the same model without ONI, all months.
        pairs = [("sarimax_x1", "sarima"), ("sarimax_x2", "sarima"), ("lgbm_oni_x1", "lgbm"),
                 ("lgbm_oni_x2", "lgbm"), ("lstm_oni_x1", "lstm"), ("lstm_oni_x2", "lstm")]
        for a, b in pairs:
            row = dm[((dm["model_1"] == a) & (dm["model_2"] == b))
                     | ((dm["model_1"] == b) & (dm["model_2"] == a))]
            if row.empty:
                continue
            r = row.iloc[0]
            sign = 1 if r["model_1"] == a else -1
            put(f"{d}.oni.{a}.diff", sign * r["mean_diff"], 3)
            put(f"{d}.oni.{a}.pholm", r["p_holm"], 3)

        # H5: rules against the default model picked on the selection window
        # of the first block (the one that holds the shift months).
        default = ch.iloc[0]["default"]
        dd = deg.set_index("model")
        for rule in ("ras", "bocpd", "fixed_share", "median_all", "comb3"):
            if rule in dd.index and default in dd.index:
                better_shift = dd.loc[rule, "mase_shift"] < dd.loc[default, "mase_shift"]
                not_worse = dd.loc[rule, "mase_stable"] <= dd.loc[default, "mase_stable"]
                put(f"{d}.h5.{rule}", "yes" if better_shift and not_worse else "no")

    from src import jobs
    for d in DATASETS:
        put(f"{d}.origins", len(jobs.origins(d)))
        put(f"{d}.forecasts", len(list((C.PREDS / d).glob("*/*_s*.parquet"))))
    gaps = pd.read_csv(C.PANELS / "ecuador_reporting_gaps.csv")
    put("ecuador.gaps", int((~gaps["incomplete_month"]).sum()))
    scored = set(pd.read_parquet(C.TABLES / "ecuador" / "errors.parquet", columns=["date"])["date"])
    window = pd.date_range(min(scored), max(scored), freq="MS")
    put("ecuador.months.unscored", len(set(window) - scored))
    ep = pd.read_csv(C.TABLES / "pooled" / "episodes.csv", index_col=0)
    for d in DATASETS:
        put(f"pooled.episodes.{d}", int((ep["dataset"] == d).sum()))

    cps = pd.read_csv(C.TABLES / "changepoints_ecuador.csv")
    seg = cps[cps["method"] == "segmentation"].reset_index(drop=True)
    for i, r in seg.iterrows():
        put(f"ecuador.break{i + 1}", r["break"][:7])
        put(f"ecuador.break{i + 1}.low", r["ci_low"][:7])
        put(f"ecuador.break{i + 1}.high", r["ci_high"][:7])

    sens = pd.read_csv(C.TABLES / "ecuador" / "sensitivity.csv")
    names = {"all 2024": "all2024", "Sep-Dec 2024": "sepdec", "Oct 2023-Dec 2024": "drought",
             "detected": "detected"}
    for _, r in sens.iterrows():
        put(f"ecuador.window.{names[r['window']]}.{r['model']}.rank", int(r["rank"]))
        put(f"ecuador.window.{names[r['window']]}.{r['model']}.mase", r["MASE"])

    pooled = C.TABLES / "pooled"
    fr = pd.read_csv(pooled / "friedman.csv").iloc[0]
    put("pooled.friedman.chi2", fr["chi2"])
    put("pooled.friedman.p", fr["p"], 3)
    put("pooled.friedman.cd", fr["cd"])
    put("pooled.episodes", int(fr["blocks"]))
    ranks = pd.read_csv(pooled / "ranks.csv", index_col=0)["mean_rank"]
    for m, v in ranks.items():
        put(f"pooled.rank.{m}", v)
    bayes = pd.read_csv(pooled / "bayes.csv")
    for _, r in bayes.iterrows():
        put(f"pooled.bayes.{r['reference']}.{r['model']}", r["p_first_better"], 3)
    depth = pd.read_csv(pooled / "depth.csv", index_col=0)
    from scipy import stats
    rho, p = stats.spearmanr(depth["depth"], depth["log_gap"], nan_policy="omit")
    put("pooled.depth.rho", rho)
    put("pooled.depth.p", p, 3)

    trials, gains = 0, {}
    for f in C.TUNING.glob("*/*/*.json"):
        j = json.loads(f.read_text())
        trials += j["trials"]
        gains.setdefault(f.parent.name, []).append(j["value"] / j["default_value"])
    put("tuning.trials", trials)
    for m, g in gains.items():
        put(f"tuning.{m}.gain", 100 * (1 - np.median(g)), 1)
    return out


def write(values: dict) -> None:
    lines = ["% Generated by python -m src.summary; do not edit by hand.",
             "\\makeatletter",
             "\\newcommand{\\res}[1]{\\@ifundefined{res@#1}{\\textbf{??#1??}}{\\@nameuse{res@#1}}}"]
    for k, v in sorted(values.items()):
        lines.append(f"\\@namedef{{res@{k}}}{{{v}}}")
    lines.append("\\makeatother")
    (C.TABLES / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.Series(values, name="value").rename_axis("key").to_csv(C.TABLES / "results_keys.csv")

    v = values
    md = [f"# Headline numbers ({len(values)} keys in results.tex)", ""]
    for d in DATASETS:
        md += [f"## {d}", "",
               f"- best stable: {v[f'{d}.best.stable']}; best shift: {v[f'{d}.best.shift']}",
               f"- Spearman stable vs shift MASE: {v[f'{d}.spearman']} "
               f"[{v[f'{d}.spearmanlow']}, {v[f'{d}.spearmanhigh']}]",
               f"- median shift/stable ratio: classical {v[f'{d}.ratio.median.classical']}, "
               f"learned {v[f'{d}.ratio.median.learned']}, foundation "
               f"{v[f'{d}.ratio.median.foundation']}, combination {v[f'{d}.ratio.median.combination']}",
               f"- H2 one-sided Mann-Whitney (learned > classical ratios): p {v[f'{d}.h2.p']}",
               f"- H5 rules better in shift and not worse in stable than "
               f"{v[f'{d}.block1.default']}: "
               + ", ".join(r for r in ("ras", "bocpd", "fixed_share", "median_all", "comb3")
                           if v.get(f"{d}.h5.{r}") == "yes"),
               f"- Giacomini-White, Holm < 0.05: {v[f'{d}.gw.significant']} of {v[f'{d}.gw.tested']}",
               f"- months: stable {v[f'{d}.months.stable']}, shift {v[f'{d}.months.shift']}, "
               f"rebound {v[f'{d}.months.rebound']}", ""]
    md += ["## pooled episodes", "",
           f"- Friedman chi2 {v['pooled.friedman.chi2']} (p {v['pooled.friedman.p']}), "
           f"{v['pooled.episodes']} episodes, CD {v['pooled.friedman.cd']}",
           f"- depth vs learned/classical gap (exploratory): Spearman {v['pooled.depth.rho']} "
           f"(p {v['pooled.depth.p']})",
           f"- tuning trials: {v['tuning.trials']}"]
    (C.TABLES / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    values = collect()
    write(values)
    print(f"{len(values)} keys -> {C.TABLES / 'results.tex'}")
    print((C.TABLES / "summary.md").read_text())


if __name__ == "__main__":
    main()
