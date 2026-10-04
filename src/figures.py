"""Paper figures, drawn from the tables that src.evaluate writes.

    python -m src.figures            # every figure whose inputs exist
    python -m src.figures --only fig_degradation

Each function reads what it needs and returns quietly if a table is missing,
so the figures can be redrawn while the campaign is still running.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

import config as C
from src import data, oni
from src import plot_style as S

DATASETS = ("ecuador", "brazil", "europe")
LABEL = {"ecuador": "Ecuador", "brazil": "Brazil", "europe": "EU countries"}
SHORT = {"ecuador": "EC", "brazil": "BR", "europe": "EU"}

# A short list for the dense figures: the references plus the best-known
# member of every family.
CORE = ["naive", "snaive", "swa3", "ets", "theta", "comb", "sarima", "prophet", "lgbm",
        "lgbm_oni_x1", "gru", "lstm", "bilstm", "nbeats", "nhits", "patchtst", "dlinear",
        "tide", "chronos", "timesfm", "moirai", "median_all", "fixed_share"]


# One model per family, so every line keeps its own colour and marker.
ONE_PER_FAMILY = ["swa3", "sarima", "lgbm", "lstm", "patchtst", "moirai", "median_all"]


def _log_axis(ax, values):
    """Log scale with plain tick labels (0.5, 1, 2, 3...) instead of 2x10^0."""
    ax.set_xscale("log")
    lo, hi = np.nanmin(values) * 0.9, np.nanmax(values) * 1.1
    ticks = [t for t in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8) if lo <= t <= hi]
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _table(dataset, name, **kw):
    path = C.TABLES / dataset / name
    if not path.exists():
        return None
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, **kw)


def _shade(ax, regimes, target):
    """Grey bands over shift months, lighter ones over rebound months."""
    r = regimes[regimes["target"] == target].sort_values("date")
    for kind, colour in (("shift", S.BAND), ("rebound", S.BAND_2)):
        on = r["regime"].eq(kind).to_numpy()
        dates = r["date"].to_numpy()
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                ax.axvspan(dates[i] - np.timedelta64(15, "D"), dates[j] + np.timedelta64(15, "D"),
                           color=colour, lw=0, zorder=0)
                i = j + 1
            else:
                i += 1


def fig_series():
    """F1. Ecuador's consumption, its year-on-year growth with the detected
    segments, and the ONI, on one time axis."""
    path = C.TABLES / "regimes_ecuador.csv"
    if not path.exists():
        return
    reg = pd.read_csv(path, parse_dates=["date"])
    cps = pd.read_csv(C.TABLES / "changepoints_ecuador.csv", parse_dates=["break", "ci_low", "ci_high"])
    cps = cps[cps["method"] == "segmentation"]
    panel, static = data.load("ecuador")
    truth = data.actuals(panel, static)
    # Unscored months (reporting gaps) stay blank instead of being bridged.
    y = truth[truth["target"] == "EC"].set_index("date")["y"].asfreq("MS")
    o = oni.load().loc[y.index.min():y.index.max()]

    fig, axes = plt.subplots(3, 1, figsize=(S.DOUBLE, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.6, 0.9]})
    for ax in axes:
        _shade(ax, reg, "EC")
    axes[0].plot(y.index, y.values, color=S.INK, lw=1.2, marker="o", ms=1.6)
    axes[0].set_ylabel("Billed consumption\n(GWh per month)")

    r = reg[reg["target"] == "EC"]
    axes[1].plot(r["date"], r["yoy"], color=S.INK_2, lw=0.9)
    axes[1].step(r["date"], r["segment_mean"], where="mid", color=S.COLOR["simple"], lw=1.6)
    axes[1].axhline(0, color=S.AXIS, lw=0.6)
    axes[1].set_ylabel("Year-on-year\ngrowth (%)")
    # Interval bars only for the breaks that open or close a shift or rebound.
    edges = r[r["regime"] != r["regime"].shift()]["date"]
    low, top = axes[1].get_ylim()
    axes[1].set_ylim(low, top + 0.25 * (top - low))
    top = top + 0.15 * (top - low)
    shown = 0
    for _, c in cps.iterrows():
        for ax in axes[:2]:
            ax.axvline(c["break"], color=S.AXIS, lw=0.6, ls="--", zorder=1)
        if c["break"] in set(edges):
            # Alternate two heights so neighbouring intervals do not overlap.
            level = top - (shown % 2) * 0.08 * (top - low)
            axes[1].plot([c["ci_low"], c["ci_high"]], [level] * 2, color=S.INK_2,
                         lw=1.2, marker="|", ms=5)
            shown += 1

    colours = np.where(o.values >= 0, S.DIVERGING[2], S.DIVERGING[0])
    axes[2].bar(o.index, o.values, width=25, color=colours, lw=0)
    axes[2].axhline(0, color=S.AXIS, lw=0.6)
    axes[2].set_ylabel("ONI (°C)")
    axes[2].grid(False)
    for ax, letter in zip(axes, "abc"):
        ax.text(0.006, 0.97, f"({letter})", transform=ax.transAxes, va="top", fontsize=8)
    fig.align_ylabels(axes)
    S.save(fig, "fig_series_ecuador")


def fig_protocol():
    """F2. Monthly origins, the expanding window, the horizon and tuning."""
    start, data_end = pd.Timestamp("2014-01-01"), pd.Timestamp("2024-12-01")
    rows = [("origin 2019-01", "2019-01-01", "forecast"),
            ("origin 2019-02", "2019-02-01", "forecast"),
            ("origin 2021-01", "2021-01-01", "forecast"),
            ("origin 2024-10", "2024-10-01", "forecast"),
            ("validation origin 2020-01", "2020-01-01", "validation"),
            ("validation origin 2020-07", "2020-07-01", "validation")]
    fig, ax = plt.subplots(figsize=(S.DOUBLE, 2.1))
    for i, (label, o, kind) in enumerate(rows):
        o = pd.Timestamp(o)
        yv = len(rows) - i
        end = o + pd.DateOffset(months=12)
        colour = S.COLOR["simple"] if kind == "forecast" else S.COLOR["statistical"]
        if kind == "validation":
            end = min(end, pd.Timestamp("2021-01-01"))    # nothing from the tuning year on
        ax.barh(yv, (o - start).days, left=start, height=0.55, color=S.GRID, lw=0)
        scored = min(end, data_end + pd.DateOffset(months=1))
        ax.barh(yv, (scored - o).days, left=o, height=0.55, color=colour, lw=0)
        if end > scored:
            ax.barh(yv, (end - scored).days, left=scored, height=0.55, color="white",
                    edgecolor=colour, hatch="////", lw=0.6)
        ax.text(start - pd.Timedelta(days=40), yv, label, ha="right", va="center", fontsize=7)
    ax.axvline(pd.Timestamp("2021-01-01"), ymax=0.3, color=S.AXIS, lw=0.6, ls="--")
    ax.text(pd.Timestamp("2021-01-20"), 0.55, "tuning origin 2021-01", fontsize=6.5, color=S.INK_2)
    ax.axvline(data_end + pd.DateOffset(months=1), color=S.AXIS, lw=0.6, ls=":")
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(start - pd.Timedelta(days=30), pd.Timestamp("2026-01-01"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=S.GRID),
               plt.Rectangle((0, 0), 1, 1, color=S.COLOR["simple"]),
               plt.Rectangle((0, 0), 1, 1, color=S.COLOR["statistical"]),
               plt.Rectangle((0, 0), 1, 1, fc="white", ec=S.COLOR["simple"], hatch="////")]
    ax.legend(handles, ["training window", "scored forecast (12 months)",
                        "validation forecast", "beyond the last observed month"],
              ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    S.save(fig, "fig_protocol")


def fig_error_heatmap(dataset="ecuador"):
    """F3. Scaled error of every model by target month (all horizons pooled)."""
    err = _table(dataset, "errors.parquet")
    if err is None:
        return
    err = err[~err["model"].str.endswith("_x3")]
    L = err.groupby(["model", "date"])["sae"].mean().unstack("date")
    stable = err[err["regime"] == "stable"].groupby("model")["sae"].mean()
    order = sorted(L.index, key=lambda m: (S.FAMILIES.index(S.family(m)), stable.get(m, np.inf)))
    L = L.loc[order]
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#256abf", "#0d366b"])
    fig, ax = plt.subplots(figsize=(S.DOUBLE, 0.14 * len(L) + 1.0))
    im = ax.imshow(L.to_numpy(), aspect="auto", cmap=cmap, vmin=0,
                   vmax=np.nanpercentile(L.to_numpy(), 98), interpolation="nearest")
    ax.set_yticks(range(len(L)), L.index, fontsize=6)
    ticks = [i for i, d in enumerate(L.columns) if d.month == 1]
    ax.set_xticks(ticks, [L.columns[i].year for i in ticks])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Mean scaled error (MASE units)")
    cb.outline.set_visible(False)
    S.save(fig, f"fig_error_heatmap_{dataset}")


def fig_degradation():
    """F4. Stable versus shift MASE per model, sorted by their ratio."""
    tabs = {d: _table(d, "degradation.csv") for d in DATASETS}
    tabs = {d: t for d, t in tabs.items() if t is not None}
    if not tabs:
        return
    fig, axes = plt.subplots(1, len(tabs), figsize=(S.DOUBLE, 5.4), sharey=False,
                             gridspec_kw={"wspace": 0.55})
    axes = np.atleast_1d(axes)
    for ax, (d, t) in zip(axes, tabs.items()):
        t = t.sort_values("ratio", ascending=False).reset_index(drop=True)
        for i, row in t.iterrows():
            fam = S.family(row["model"])
            ax.plot([row["mase_stable"], row["mase_shift"]], [i, i], color=S.COLOR[fam], lw=1)
            ax.plot(row["mase_stable"], i, marker=S.MARKER[fam], mfc="white", mec=S.COLOR[fam],
                    ms=4, ls="")
            ax.plot(row["mase_shift"], i, marker=S.MARKER[fam], color=S.COLOR[fam], ms=4, ls="")
        ax.set_yticks(range(len(t)), t["model"], fontsize=6)
        _log_axis(ax, np.r_[t["mase_stable"], t["mase_shift"]])
        ax.set_xlabel("MASE (open: stable,\nfilled: shift)")
        ax.set_title(LABEL[d])
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    _family_legend(fig)
    S.save(fig, "fig_degradation")


def _family_legend(fig, families=None):
    families = families or S.FAMILIES
    handles = [plt.Line2D([], [], color=S.COLOR[f], marker=S.MARKER[f], ls="", ms=5)
               for f in families]
    fig.legend(handles, families, ncol=len(families), loc="lower center",
               bbox_to_anchor=(0.5, -0.04))


def fig_on_the_line():
    """F5. Is shift-regime accuracy predictable from stable-regime accuracy?"""
    tabs = {d: _table(d, "degradation.csv") for d in DATASETS}
    tabs = {d: t for d, t in tabs.items() if t is not None}
    if not tabs:
        return
    fig, axes = plt.subplots(1, len(tabs), figsize=(S.DOUBLE, 2.5))
    axes = np.atleast_1d(axes)
    for ax, (d, t) in zip(axes, tabs.items()):
        for _, row in t.iterrows():
            fam = S.family(row["model"])
            g = S.GROUP[fam]
            ax.plot(row["mase_stable"], row["mase_shift"], marker=S.MARKER[fam], ls="",
                    color=S.GROUP_COLOR[g], ms=4.5, mec="white", mew=0.4)
        x0, x1 = t["mase_stable"].min() * 0.9, t["mase_stable"].max() * 1.1
        y0, y1 = t["mase_shift"].min() * 0.9, t["mase_shift"].max() * 1.1
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        lo, hi = min(x0, y0), max(x1, y1)
        ax.plot([lo, hi], [lo, hi], color=S.AXIS, lw=0.6, ls="--")
        line = _table(d, "on_the_line.csv")
        if line is not None:
            r = line.iloc[0]
            ax.text(0.03, 0.97, f"Spearman {r['spearman']:.2f}\n[{r['low']:.2f}, {r['high']:.2f}]",
                    transform=ax.transAxes, va="top", fontsize=7, color=S.INK_2)
        ax.set_xlabel("MASE, stable months")
        ax.set_title(LABEL[d])
        ax.grid(axis="both")
    axes[0].set_ylabel("MASE, shift months")
    handles = [plt.Line2D([], [], color=c, marker="o", ls="", ms=5) for c in S.GROUP_COLOR.values()]
    fig.subplots_adjust(wspace=0.3, bottom=0.25)
    fig.legend(handles, list(S.GROUP_COLOR), ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    S.save(fig, "fig_on_the_line")


def fig_mcs():
    """F6. Models in the 90% model confidence set, for every dataset and every
    regime long enough to test (see src.evaluate.MIN_TEST_MONTHS)."""
    cols = {}
    for d in DATASETS:
        for regime in ("all", "stable", "shift", "rebound"):
            t = _table(d, f"mcs_{regime}.csv", index_col=0)
            if t is not None:
                cols[f"{SHORT[d]}\n{regime}"] = t["mcs_p"]
    if not cols:
        return
    P = pd.DataFrame(cols)
    P = P.loc[[m for m in CORE if m in P.index]]
    fig, ax = plt.subplots(figsize=(S.DOUBLE * 0.7, 0.17 * len(P) + 1.0))
    for j, col in enumerate(P.columns):
        for i, m in enumerate(P.index):
            p = P.loc[m, col]
            if np.isnan(p):
                continue
            c = S.COLOR[S.family(m)]
            ax.plot(j, i, "o", ms=3 + 6 * min(p, 1), color=c if p >= 0.10 else "white",
                    mec=c, mew=0.8)
    ax.set_xticks(range(len(P.columns)), P.columns, fontsize=6.5)
    ax.set_yticks(range(len(P.index)), P.index, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(-0.6, len(P.columns) - 0.4)
    ax.grid(False)
    ax.set_title("Filled: inside the 90% MCS; size: MCS p-value", fontsize=7, color=S.INK_2)
    S.save(fig, "fig_mcs")


def fig_dm_matrix(dataset="ecuador", regime="all"):
    """F7. Signed Diebold-Mariano statistics, Holm-significant pairs outlined."""
    t = _table(dataset, f"dm_{regime}.csv")
    if t is None:
        return
    models = [m for m in CORE if m in set(t["model_1"]) | set(t["model_2"])]
    k = len(models)
    M = np.full((k, k), np.nan)
    sig = np.zeros((k, k), bool)
    idx = {m: i for i, m in enumerate(models)}
    for _, r in t.iterrows():
        if r["model_1"] in idx and r["model_2"] in idx:
            i, j = idx[r["model_1"]], idx[r["model_2"]]
            a, b = max(i, j), min(i, j)
            sign = 1 if i == a else -1
            M[a, b] = sign * r["dm"]
            sig[a, b] = r["p_holm"] < 0.05
    cmap = LinearSegmentedColormap.from_list("div", list(S.DIVERGING))
    lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
    fig, ax = plt.subplots(figsize=(S.DOUBLE * 0.62, S.DOUBLE * 0.56))
    im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
    for a, b in zip(*np.where(sig)):
        ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, fill=False, ec=S.INK, lw=0.9))
    ax.set_xticks(range(k), models, rotation=90, fontsize=6)
    ax.set_yticks(range(k), models, fontsize=6)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("DM statistic (negative: row model has lower loss)")
    cb.outline.set_visible(False)
    S.save(fig, f"fig_dm_{dataset}_{regime}")


def fig_fluctuation(dataset="ecuador"):
    """F8. Giacomini-Rossi paths against seasonal naive, shift months shaded."""
    t = _table(dataset, "fluctuation.csv", parse_dates=["date"])
    reg_path = C.TABLES / f"regimes_{dataset}.csv"
    if t is None or not reg_path.exists():
        return
    reg = pd.read_csv(reg_path, parse_dates=["date"])
    target = t["target"].iloc[0]
    t = t[t["target"] == target]
    picks = [m for m in ("ets", "sarima", "lgbm", "lstm", "patchtst", "chronos") if m in set(t["model"])]
    if not picks:
        return
    fig, axes = plt.subplots(2, (len(picks) + 1) // 2, figsize=(S.DOUBLE, 3.2), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, picks):
        f = t[t["model"] == m]
        _shade(ax, reg, target)
        fam = S.family(m)
        ax.plot(f["date"], f["statistic"], color=S.COLOR[fam], lw=1.2)
        cv = f["critical"].iloc[0]
        for s in (-cv, cv):
            ax.axhline(s, color=S.AXIS, lw=0.6, ls="--")
        ax.axhline(0, color=S.AXIS, lw=0.4)
        ax.set_title(f"{m} vs seasonal naive", fontsize=7)
    for ax in axes.flat[len(picks):]:
        ax.set_visible(False)
    fig.supylabel("Rolling DM statistic\n(above 0: seasonal naive better)", fontsize=8)
    S.save(fig, f"fig_fluctuation_{dataset}")


def fig_trajectories(dataset="ecuador"):
    """F9. Forecasts from the last origin before each shift, against the data."""
    err = _table(dataset, "errors.parquet")
    reg_path = C.TABLES / f"regimes_{dataset}.csv"
    if err is None or not reg_path.exists():
        return
    reg = pd.read_csv(reg_path, parse_dates=["date"])
    target = err["target"].iloc[0]
    shifts = reg[(reg["target"] == target) & (reg["regime"] == "shift")]["date"]
    if shifts.empty:
        return
    starts = sorted(shifts[shifts.diff().dt.days.fillna(999) > 40])
    picks = [m for m in ONE_PER_FAMILY[:-1] if m in set(err["model"])]
    last = err["date"].max()
    fig, axes = plt.subplots(1, len(starts), figsize=(S.DOUBLE, 2.5), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, starts):
        origin = s - pd.DateOffset(months=2)
        sub = err[(err["origin"] == origin) & (err["target"] == target)]
        truth = sub.drop_duplicates("date").sort_values("date")
        hist = err[(err["target"] == target) & (err["horizon"] == 1)
                   & (err["date"] >= origin - pd.DateOffset(months=12)) & (err["date"] < origin)]
        hist = hist.drop_duplicates("date").sort_values("date")
        _shade(ax, reg, target)
        ax.plot(pd.concat([hist["date"], truth["date"]]), pd.concat([hist["y"], truth["y"]]),
                color=S.INK, lw=1.6, label="observed")
        for m in picks:
            g = sub[sub["model"] == m].sort_values("date")
            fam = S.family(m)
            ax.plot(g["date"], g["y_hat"], color=S.COLOR[fam], marker=S.MARKER[fam], ms=3, lw=1,
                    label=m)
        ax.axvline(origin, color=S.AXIS, lw=0.6, ls="--")
        ax.set_title(f"Origin {origin:%Y-%m}", fontsize=7)
        ax.set_xlim(origin - pd.DateOffset(months=12),
                    min(origin + pd.DateOffset(months=12), last + pd.DateOffset(months=1)))
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].set_ylabel("GWh per month")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(labels), loc="lower center", bbox_to_anchor=(0.5, -0.1))
    S.save(fig, f"fig_trajectories_{dataset}")


def fig_episodes():
    """F10. MASE relative to seasonal naive in every shift episode."""
    rows = []
    for d in DATASETS:
        t = _table(d, "episodes.csv", index_col=0)
        if t is None or "snaive" not in t:
            continue
        rel = t.div(t["snaive"], axis=0)
        rel["dataset"] = d
        rows.append(rel)
    if not rows:
        return
    rel = pd.concat(rows)
    picks = [m for m in ONE_PER_FAMILY if m in rel]
    # EU episodes are summarised across countries by median and quartiles.
    eu = rel[rel["dataset"] == "europe"]
    groups = [(i, rel.loc[[i], picks]) for i in rel.index[rel["dataset"] != "europe"]]
    if not eu.empty:
        year = eu.index.str.split(":").str[1].str[:4]
        for y in sorted(set(year)):
            n = int((year == y).sum())
            noun = "country" if n == 1 else "countries"
            groups.append((f"EU, {y} ({n} {noun})", eu.loc[year == y, picks]))
    fig, ax = plt.subplots(figsize=(S.DOUBLE, 0.32 * len(groups) + 1.0))
    off = np.linspace(-0.3, 0.3, len(picks))
    for i, (name, g) in enumerate(groups):
        for k, m in enumerate(picks):
            v = g[m].dropna()
            if v.empty:
                continue
            fam = S.family(m)
            med = v.median()
            ax.plot(med, i + off[k], marker=S.MARKER[fam], color=S.COLOR[fam], ms=4, ls="")
            if len(v) > 1:
                ax.plot([v.quantile(0.25), v.quantile(0.75)], [i + off[k]] * 2, color=S.COLOR[fam], lw=1)
    ax.axvline(1, color=S.AXIS, lw=0.8)
    _log_axis(ax, np.concatenate([g[1].to_numpy().ravel() for g in groups]))
    ax.set_yticks(range(len(groups)), [g[0] for g in groups], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("MASE relative to seasonal naive (below 1: better)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], color=S.COLOR[S.family(m)], marker=S.MARKER[S.family(m)], ls="")
               for m in picks]
    fig.legend(handles, picks, ncol=len(picks), loc="lower center", bbox_to_anchor=(0.5, -0.06))
    S.save(fig, "fig_episodes")


def fig_ras_grid(dataset="ecuador"):
    """F11. RAS sensitivity: MASE after the selection window for every setting."""
    g = _table(dataset, "ras_grid.csv")
    ch = _table(dataset, "combine_choices.csv")
    if g is None:
        return
    blocks = g["block"].unique()
    fig, axes = plt.subplots(1, len(blocks), figsize=(S.SINGLE * len(blocks), 1.9))
    axes = np.atleast_1d(axes)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#86b6ef", "#256abf", "#0d366b"])
    for ax, b in zip(axes, blocks):
        P = g[g["block"] == b].pivot(index="window", columns="factor", values="mase_after")
        im = ax.imshow(P.to_numpy(), cmap=cmap, aspect="auto")
        for (i, j), v in np.ndenumerate(P.to_numpy()):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if v > np.nanmean(P.to_numpy()) else S.INK)
        if ch is not None:
            c = ch[ch["block"] == b].iloc[0]
            j = list(P.columns).index(c["ras_factor"])
            i = list(P.index).index(c["ras_window"])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=S.INK, lw=1.2))
        ax.set_xticks(range(len(P.columns)), P.columns)
        ax.set_yticks(range(len(P.index)), P.index)
        ax.set_xlabel("threshold factor")
        ax.set_ylabel("window (months)")
        ax.grid(False)
    S.save(fig, f"fig_ras_grid_{dataset}")


def fig_oni_tiers(dataset="ecuador"):
    """F12. What the ONI adds under each information tier."""
    acc = _table(dataset, "accuracy.csv")
    if acc is None:
        return
    groups = [("sarima", "sarimax"), ("lgbm", "lgbm_oni"), ("lstm", "lstm_oni")]
    tiers = [("without ONI", "o", False), ("x1, persisted", "^", True),
             ("x2, ARIMA forecast", "s", True), ("x3, realised (hindsight)", "D", False)]
    regimes = [r for r in ("stable", "shift") if r in set(acc["regime"])]
    fig, axes = plt.subplots(1, len(regimes), figsize=(S.DOUBLE, 1.9), sharey=True,
                             gridspec_kw={"wspace": 0.15})
    axes = np.atleast_1d(axes)
    for ax, regime in zip(axes, regimes):
        a = acc[acc["regime"] == regime].set_index("model")["MASE"]
        for i, (base, ext) in enumerate(groups):
            names = [base, f"{ext}_x1", f"{ext}_x2", f"{ext}_x3"]
            c = S.COLOR[S.family(base)]
            for k, (name, (_, marker, filled)) in enumerate(zip(names, tiers)):
                if name in a:
                    ax.plot(a[name], i + (k - 1.5) * 0.16, marker=marker, ls="", ms=4.5,
                            color=c, mfc=c if filled else "white")
        ax.set_yticks(range(len(groups)), [g[0] for g in groups])
        ax.set_title(f"{regime} months", fontsize=7)
        ax.set_xlabel("MASE")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], marker=m, ls="", color=S.INK_2,
                          mfc=S.INK_2 if f else "white", ms=5) for _, m, f in tiers]
    fig.legend(handles, [t[0] for t in tiers], ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, -0.12))
    S.save(fig, f"fig_oni_{dataset}")


def fig_cd():
    """F13. Critical-difference diagram over the independent shift episodes."""
    ranks_path = C.TABLES / "pooled" / "ranks.csv"
    if not ranks_path.exists():
        return
    ranks = pd.read_csv(ranks_path, index_col=0)["mean_rank"].sort_values()
    fr = pd.read_csv(C.TABLES / "pooled" / "friedman.csv").iloc[0]
    nem = pd.read_csv(C.TABLES / "pooled" / "nemenyi.csv", index_col=0)
    k = len(ranks)
    half = (k + 1) // 2
    fig, ax = plt.subplots(figsize=(S.DOUBLE, 0.2 * half + 1.6))
    lo, hi = np.floor(ranks.min()), np.ceil(ranks.max())
    ax.set_xlim(lo - 0.5, hi + 0.5)
    ax.set_ylim(-(1.6 + 0.9 * half), 1.5)
    ax.axhline(0, color=S.INK, lw=0.8)
    for r in np.arange(lo, hi + 1):
        ax.plot([r, r], [0, 0.15], color=S.INK, lw=0.6)
        ax.text(r, 0.3, f"{r:.0f}", ha="center", fontsize=7)
    for i, (m, r) in enumerate(ranks.items()):
        left = i < half
        j = i if left else k - 1 - i
        y = -1.6 - j * 0.9
        xt = lo - 0.4 if left else hi + 0.4
        fam = S.family(m)
        ax.plot([r, r], [0, y], color=S.AXIS, lw=0.5)
        ax.plot([r, xt], [y, y], color=S.AXIS, lw=0.5)
        ax.plot(r, 0, marker=S.MARKER[fam], color=S.COLOR[fam], ms=4)
        ax.text(xt + (-0.05 if left else 0.05), y, m, ha="right" if left else "left",
                va="center", fontsize=6.5)
    # Bars join runs of models whose Nemenyi test does not separate them; runs
    # contained in a longer one are dropped, and each bar takes the first row
    # where it does not overlap another.
    order = list(ranks.index)
    runs = []
    for a in range(k):
        b = a
        while b + 1 < k and nem.loc[order[a], order[b + 1]] > 0.05:
            b += 1
        if b > a and not any(sa <= a and b <= sb for sa, sb in runs):
            runs.append((a, b))
    rows = []
    for a, b in runs:
        x0, x1 = ranks.iloc[a] - 0.05, ranks.iloc[b] + 0.05
        row = next((i for i, used in enumerate(rows) if all(x0 > e + 0.2 or x1 < s - 0.2
                                                           for s, e in used)), len(rows))
        if row == len(rows):
            rows.append([])
        rows[row].append((x0, x1))
        ax.plot([x0, x1], [-0.3 - 0.22 * row] * 2, color=S.INK, lw=1.6, solid_capstyle="butt")
    ax.plot([lo, lo + fr["cd"]], [1.0, 1.0], color=S.INK, lw=1.2)
    ax.text(lo + fr["cd"] / 2, 1.15, f"CD = {fr['cd']:.2f}", ha="center", fontsize=7)
    ax.axis("off")
    S.save(fig, "fig_cd_episodes")


FIGURES = [fig_series, fig_protocol, fig_error_heatmap, fig_degradation, fig_on_the_line,
           fig_mcs, fig_dm_matrix, fig_fluctuation, fig_trajectories, fig_episodes,
           fig_ras_grid, fig_oni_tiers, fig_cd]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    S.apply()
    for f in FIGURES:
        if a.only and f.__name__ not in a.only:
            continue
        f()


if __name__ == "__main__":
    main()
