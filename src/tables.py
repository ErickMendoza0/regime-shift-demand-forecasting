"""LaTeX tables for the paper, written from the result tables.

    python -m src.tables          # -> work/tables/latex/*.tex

Each file holds a bare tabular (booktabs), so captions and labels stay in the
manuscript.
"""
import json
import re

import numpy as np
import pandas as pd

import config as C
from src import plot_style as S
from src.spaces import DEFAULTS, SPACES

OUT = C.TABLES / "latex"
DATASETS = ("ecuador", "brazil", "europe")
LABEL = {"ecuador": "Ecuador", "brazil": "Brazil", "europe": "EU-27"}
FAMILY_ORDER = {f: i for i, f in enumerate(S.FAMILIES)}
NAMES = {
    "naive": "Naive", "snaive": "Seasonal naive", "snaive_drift": "Seasonal naive + drift",
    "swa3": "Seasonal mean (3 y)", "drift": "Drift", "ets": "ETS", "theta": "Theta",
    "comb": "M4 Comb", "arima_agg": "Aggregate ARIMA(1,1,1)", "arima": "ARIMA",
    "sarima": "SARIMA", "sarimax_x1": "SARIMAX (ONI x1)", "sarimax_x2": "SARIMAX (ONI x2)",
    "sarimax_x3": "SARIMAX (ONI x3)", "prophet": "Prophet", "lgbm": "LightGBM",
    "lgbm_oni_x1": "LightGBM (ONI x1)", "lgbm_oni_x2": "LightGBM (ONI x2)",
    "lgbm_oni_x3": "LightGBM (ONI x3)", "gru": "GRU", "lstm": "LSTM",
    "lstm_oni_x1": "LSTM (ONI x1)", "lstm_oni_x2": "LSTM (ONI x2)", "lstm_oni_x3": "LSTM (ONI x3)",
    "bilstm": "BiLSTM", "nbeats": "N-BEATS", "nhits": "N-HiTS", "patchtst": "PatchTST",
    "dlinear": "DLinear", "tide": "TiDE", "chronos": "Chronos-2", "timesfm": "TimesFM 3.0",
    "moirai": "Moirai 2.0", "ras": "RAS switch", "bocpd": "BOCPD switch",
    "fixed_share": "Fixed share", "median_all": "Median of all", "comb3": "Mean of three",
}


def f2(v):
    return "--" if pd.isna(v) else f"{v:.2f}"


def fp(p):
    if pd.isna(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


def write(name, header, rows, align):
    # A typographic minus for negative numbers in the body of the table.
    rows = [re.sub(r"(?<=[\s\[])-(?=\d)", "$-$", r) for r in rows]
    lines =[rf"\begin{{tabular}}{{{align}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines += rows
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def ordered(models):
    return sorted(models, key=lambda m: (FAMILY_ORDER[S.family(m)], list(NAMES).index(m)
                                         if m in NAMES else 99))


def table_breaks():
    rows = []
    for d in DATASETS:
        cp = pd.read_csv(C.TABLES / f"changepoints_{d}.csv")
        reg = pd.read_csv(C.TABLES / f"regimes_{d}.csv", parse_dates=["date"])
        seg = cp[cp["method"] == "segmentation"]
        for target, g in seg.groupby("target"):
            r = reg[reg["target"] == target].copy()
            r["run"] = (r["regime"] != r["regime"].shift()).cumsum()
            runs = r[r["regime"].isin(["shift", "rebound"])].groupby("run").agg(
                regime=("regime", "first"), start=("date", "min"), end=("date", "max"),
                mean=("yoy", "mean"))
            if d == "europe" or runs.empty:
                continue
            for _, x in runs.iterrows():
                b = g[g["break"] == f"{x['start']:%Y-%m-%d}"]
                ci = (f"{b['ci_low'].iloc[0][:7]} to {b['ci_high'].iloc[0][:7]}"
                      if len(b) else "--")
                rows.append(f"{LABEL[d]} & {x['regime']} & {x['start']:%Y-%m} & "
                            f"{x['end']:%Y-%m} & {x['mean']:.1f} & {ci} \\\\")
    write("breaks", r"Dataset & Regime & First month & Last month & Mean YoY growth (\%) & "
          r"95\% interval of the break", rows, "llllrl")


def table_europe_episodes():
    reg = pd.read_csv(C.TABLES / "regimes_europe.csv", parse_dates=["date"])
    reg = reg[reg["date"] >= C.DATASETS["europe"]["origins"][0][0]]
    reg["run"] = ((reg["regime"] != reg["regime"].shift())
                  | (reg["target"] != reg["target"].shift())).cumsum()
    runs = reg[reg["regime"] == "shift"].groupby("run").agg(
        target=("target", "first"), start=("date", "min"), months=("date", "size"),
        mean=("yoy", "mean"))
    runs["year"] = runs["start"].dt.year
    rows = []
    for y, g in runs.groupby("year"):
        rows.append(f"{y} & {len(g)} & {', '.join(sorted(g['target']))} & "
                    f"{g['months'].median():.0f} & {g['mean'].median():.1f} \\\\")
    write("europe_episodes", r"Start year & Episodes & Countries & Median months & "
          r"Median YoY growth (\%)", rows, "rrp{6.2cm}rr")


def table_accuracy(d="ecuador"):
    deg = pd.read_csv(C.TABLES / d / "degradation.csv").set_index("model")
    acc = pd.read_csv(C.TABLES / d / "accuracy.csv")
    reb = acc[acc["regime"] == "rebound"].set_index("model")["MASE"]
    bias = acc[acc["regime"] == "shift"].set_index("model")["bias"]
    mae = acc[acc["regime"] == "shift"].set_index("model")["MAE"]
    rows, last = [], None
    for m in ordered(deg.index):
        fam = S.family(m)
        if last is not None and fam != last:
            rows.append(r"\addlinespace")
        last = fam
        r = deg.loc[m]
        rows.append(f"{esc(NAMES.get(m, m))} & {fam} & {f2(r['mase_stable'])} & "
                    f"{int(r['rank_stable'])} & {f2(r['mase_shift'])} & {int(r['rank_shift'])} & "
                    f"{f2(reb.get(m))} & {f2(r['ratio'])} & {mae.get(m, np.nan):.0f} & "
                    f"{bias.get(m, np.nan):.0f} \\\\")
    write(f"accuracy_{d}", r"Model & Family & \multicolumn{2}{c}{Stable} & "
          r"\multicolumn{2}{c}{Shift} & Rebound & Shift/stable & Shift MAE & Shift bias \\"
          "\n" r" & & MASE & rank & MASE & rank & MASE & ratio & (GWh) & (GWh)",
          rows, "llrrrrrrrr")


def table_external():
    rows = []
    for d in DATASETS:
        deg = pd.read_csv(C.TABLES / d / "degradation.csv")
        line = pd.read_csv(C.TABLES / d / "on_the_line.csv").iloc[0]
        fam = deg["model"].map(S.family)
        med = lambda fs: deg.loc[fam.isin(fs), "ratio"].median()
        best_s = deg.loc[deg["mase_stable"].idxmin(), "model"]
        best_k = deg.loc[deg["mase_shift"].idxmin(), "model"]
        rows.append(f"{LABEL[d]} & {esc(NAMES[best_s])} & {esc(NAMES[best_k])} & "
                    f"{line['spearman']:.2f} [{line['low']:.2f}, {line['high']:.2f}] & "
                    f"{med({'simple', 'statistical', 'decomposition'}):.2f} & "
                    f"{med({'boosting', 'recurrent', 'deep'}):.2f} & "
                    f"{med({'foundation'}):.2f} & {med({'combination'}):.2f} \\\\")
    write("datasets_summary", r"Dataset & Best, stable & Best, shift & Spearman (95\% CI) & "
          r"\multicolumn{4}{c}{Median shift/stable MASE ratio} \\" "\n"
          r" & & & & classical & learned & foundation & combination", rows, "lllrrrrr")


def table_sensitivity():
    s = pd.read_csv(C.TABLES / "ecuador" / "sensitivity.csv")
    order = ["detected", "Sep-Dec 2024", "all 2024", "Oct 2023-Dec 2024"]
    months = s.groupby("window")["months"].first()
    piv = s.pivot(index="model", columns="window", values="rank")[order]
    keep = ["swa3", "snaive", "naive", "moirai", "chronos", "timesfm", "sarima", "ets", "theta",
            "lgbm", "lstm", "nhits", "nbeats", "patchtst", "median_all", "comb3"]
    rows = [f"{esc(NAMES[m])} & " + " & ".join(f"{int(piv.loc[m, w])}" for w in order) + r" \\"
            for m in keep if m in piv.index]
    head = "Model & " + " & ".join(
        f"{w.replace('-', '--')} ({int(months[w])})" for w in order)
    write("sensitivity", head, rows, "l" + "r" * len(order))


def table_oni():
    acc = pd.read_csv(C.TABLES / "ecuador" / "degradation.csv").set_index("model")
    dm = pd.read_csv(C.TABLES / "ecuador" / "dm_all.csv")
    rows = []
    for base, ext in (("sarima", "sarimax"), ("lgbm", "lgbm_oni"), ("lstm", "lstm_oni")):
        for tier in ("x1", "x2", "x3"):
            m = f"{ext}_{tier}"
            if m not in acc.index:
                continue
            hit = dm[((dm["model_1"] == m) & (dm["model_2"] == base))
                     | ((dm["model_1"] == base) & (dm["model_2"] == m))]
            p = hit["p_holm"].iloc[0] if len(hit) else np.nan
            rows.append(f"{esc(NAMES[base])} & {tier} & {f2(acc.loc[base, 'mase_stable'])} & "
                        f"{f2(acc.loc[m, 'mase_stable'])} & {f2(acc.loc[base, 'mase_shift'])} & "
                        f"{f2(acc.loc[m, 'mase_shift'])} & {fp(p)} \\\\")
    write("oni", r"Base model & ONI tier & \multicolumn{2}{c}{Stable MASE} & "
          r"\multicolumn{2}{c}{Shift MASE} & DM $p$ (Holm) \\" "\n"
          r" & & without & with & without & with & all months", rows, "llrrrrr")


def table_rules():
    rows = []
    for d in DATASETS:
        deg = pd.read_csv(C.TABLES / d / "degradation.csv").set_index("model")
        ch = pd.read_csv(C.TABLES / d / "combine_choices.csv").iloc[0]
        default = ch["default"]
        rows.append(rf"\multicolumn{{4}}{{l}}{{\emph{{{LABEL[d]}: default {esc(NAMES[default])}, "
                    rf"fallback {esc(NAMES[ch['fallback']])}}}}} \\")
        for m in [default, "ras", "bocpd", "fixed_share", "median_all", "comb3"]:
            if m in deg.index:
                rows.append(f"\\quad {esc(NAMES[m])} & {f2(deg.loc[m, 'mase_stable'])} & "
                            f"{f2(deg.loc[m, 'mase_shift'])} & {f2(deg.loc[m, 'ratio'])} \\\\")
    write("rules", r"Forecaster & Stable MASE & Shift MASE & Shift/stable", rows, "lrrr")


def table_pooled():
    ranks = pd.read_csv(C.TABLES / "pooled" / "ranks.csv", index_col=0)["mean_rank"]
    bayes = pd.read_csv(C.TABLES / "pooled" / "bayes.csv")
    pb = bayes.pivot(index="model", columns="reference", values="p_first_better")
    rows = []
    for m, r in ranks.items():
        rows.append(f"{esc(NAMES[m])} & {S.family(m)} & {r:.2f} & "
                    f"{f2(pb.loc[m, 'snaive']) if m in pb.index else '--'} & "
                    f"{f2(pb.loc[m, 'swa3']) if m in pb.index and not pd.isna(pb.loc[m, 'swa3']) else '--'} \\\\")
    write("pooled", r"Model & Family & Mean rank & $P$(better than & $P$(better than \\" "\n"
          r" & & & seasonal naive) & seasonal mean)", rows, "llrrr")


def table_spaces():
    class Recorder:
        def __init__(self):
            self.rows = []

        def suggest_categorical(self, name, choices):
            self.rows.append((name, ", ".join(map(str, choices))))
            return choices[0]

        def suggest_float(self, name, low, high, log=False):
            self.rows.append((name, f"[{low}, {high}]" + (", log" if log else "")))
            return low

        def suggest_int(self, name, low, high, log=False):
            self.rows.append((name, f"{low} to {high}" + (", log" if log else "")))
            return low

    rows = []
    for model, space in SPACES.items():
        if model.endswith("_oni"):
            continue
        rec = Recorder()
        space(rec)
        first = True
        for name, rng in rec.rows:
            label = esc(NAMES.get(model, model)) if first else ""
            rows.append(f"{label} & \\texttt{{{esc(name)}}} & {esc(rng)} & "
                        f"{esc(DEFAULTS[model].get(name, ''))} \\\\")
            first = False
        rows.append(r"\addlinespace")
    write("spaces", "Model & Parameter & Search range & Default", rows[:-1], "llp{5cm}l")


def table_tuning():
    gains = {}
    for f in C.TUNING.glob("*/*/*.json"):
        j = json.loads(f.read_text())
        gains.setdefault(f.parent.name, []).append(1 - j["value"] / j["default_value"])
    rows = [f"{esc(NAMES.get(m, m))} & {len(g)} & {100 * np.median(g):.1f} & "
            f"{100 * np.min(g):.1f} & {100 * np.max(g):.1f} \\\\"
            for m, g in sorted(gains.items(), key=lambda kv: FAMILY_ORDER[S.family(kv[0])])]
    write("tuning", r"Model & Studies & \multicolumn{3}{c}{Validation MASE reduction over the "
          r"default (\%)} \\" "\n" r" & & median & min & max", rows, "lrrrr")


def table_gaps():
    g = pd.read_csv(C.PANELS / "ecuador_reporting_gaps.csv", parse_dates=["date"])
    g = g[~g["incomplete_month"]].sort_values("date")
    rows = [f"{r['date']:%Y-%m} & {esc(r['company'])} & {r['usual_clients']:,.0f} \\\\"
            for _, r in g.iterrows()]
    write("gaps", r"Month & Distributor & Customers usually billed", rows, "llr")


def main() -> None:
    S.apply()
    for d in DATASETS:
        table_accuracy(d)
    for fn in (table_breaks, table_europe_episodes, table_external, table_sensitivity,
               table_oni, table_rules, table_pooled, table_spaces, table_tuning, table_gaps):
        fn()
    print(f"tables written to {OUT}")


if __name__ == "__main__":
    main()
