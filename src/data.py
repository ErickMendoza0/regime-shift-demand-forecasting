"""Load each dataset as a long panel and cut it at a forecast origin.

A panel has one row per (series_id, date) with the observed value `y` (GWh) and
gaps left as NaN. Its static table says which aggregate ("target") each series
adds up to and, for Ecuador, the consumer group and province centroid.

Nothing here looks past an origin: `history` is the only way models get data.
"""
import numpy as np
import pandas as pd

import config as C

MONTHS = {'Ene': 1, 'Feb': 2, 'Mar': 3, 'Abr': 4, 'May': 5, 'Jun': 6,
          'Jul': 7, 'Ago': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dic': 12}

# Cantons whose province is mislabelled in the ARCONEL extract.
PROVINCE_FIXES = {
    'EL PIEDRERO': 'GUAYAS',
    'JUVAL': 'CHIMBORAZO',
    'MATILDE ESTHER': 'GUAYAS',
    'SANTA ROSA DE AGUA CLARA': 'CAÑAR',
    'ZONA NO LIMITADA': 'CHIMBORAZO',
}

COORDS = {
    'AZUAY': (-2.9006, -79.0045), 'BOLÍVAR': (-1.6167, -79.0000),
    'CAÑAR': (-2.5489, -78.9382), 'CARCHI': (0.5022, -77.8857),
    'CHIMBORAZO': (-1.6708, -78.6569), 'COTOPAXI': (-0.9381, -78.6140),
    'EL ORO': (-3.2596, -79.9585), 'ESMERALDAS': (0.9682, -79.6517),
    'GALÁPAGOS': (-0.9538, -90.9656), 'GUAYAS': (-2.1709, -79.9224),
    'IMBABURA': (0.3517, -78.1223), 'LOJA': (-3.9931, -79.2042),
    'LOS RÍOS': (-1.7946, -79.5342), 'MANABÍ': (-0.9676, -80.7089),
    'MORONA SANTIAGO': (-2.3930, -78.1105), 'NAPO': (-1.0515, -77.7320),
    'ORELLANA': (-0.6743, -76.9864), 'PASTAZA': (-1.4910, -77.9916),
    'PICHINCHA': (-0.2295, -78.5243), 'SANTA ELENA': (-2.2301, -80.8599),
    'SANTO DOMINGO DE LOS TSÁCHILAS': (-0.2511, -79.1717),
    'SUCUMBÍOS': (0.0870, -76.8881), 'TUNGURAHUA': (-1.2543, -78.6220),
    'ZAMORA CHINCHIPE': (-4.0655, -78.9542),
}


def _monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Re-index every series to a full monthly grid so gaps show up as NaN."""
    out = []
    for sid, g in df.groupby("series_id"):
        s = g.set_index("date")["y"].sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="MS"))
        out.append(pd.DataFrame({"series_id": sid, "date": s.index, "y": s.values}))
    return pd.concat(out, ignore_index=True)


def _ecuador() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(C.ECUADOR_CSV, encoding="latin-1", sep=";", decimal=",")
    if len(raw) < 100_000:
        # data/raw once held a small mock file with this same name; refuse it.
        raise ValueError(f"{C.ECUADOR_CSV} has only {len(raw)} rows; "
                         "this is not the ARCONEL extract")
    raw["date"] = pd.to_datetime(
        raw["Anio"].astype(int).astype(str) + "-"
        + raw["Mes"].map(MONTHS).astype(int).astype(str), format="%Y-%m")
    raw["y"] = pd.to_numeric(raw["Energia Facturada (kWh)"], errors="coerce") / 1e6
    canton = raw["Canton"].astype(str).str.upper()
    for c, p in PROVINCE_FIXES.items():
        raw.loc[canton == c, "Provincia"] = p
    raw["province"] = raw["Provincia"].astype(str).str.upper().str.strip()
    raw["group"] = raw["Grupo Consumo"].astype(str)
    raw = raw[raw["province"].isin(COORDS)].dropna(subset=["y"])

    agg = raw.groupby(["group", "province", "date"], as_index=False)["y"].sum()
    keys = agg[["group", "province"]].drop_duplicates().sort_values(["group", "province"])
    keys["series_id"] = [f"ec{i:03d}" for i in range(len(keys))]
    agg = agg.merge(keys, on=["group", "province"])
    panel = _monthly(agg[["series_id", "date", "y"]])

    codes = {g: i for i, g in enumerate(sorted(keys["group"].unique()))}
    static = keys.assign(
        target="EC",
        group_code=keys["group"].map(codes),
        latitude=keys["province"].map(lambda p: COORDS[p][0]),
        longitude=keys["province"].map(lambda p: COORDS[p][1]),
    )
    return panel, static.reset_index(drop=True)


def _brazil() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(C.BRAZIL_CSV, parse_dates=["date"])
    raw = raw[raw["date"] >= C.DATASETS["brazil"]["train_start"]]
    panel = _monthly(raw.rename(columns={"region": "series_id", "gwh": "y"}))
    static = pd.DataFrame({"series_id": sorted(panel["series_id"].unique()), "target": "BR"})
    return panel, static


def _europe() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(C.EUROPE_CSV, parse_dates=["date"])
    panel = _monthly(raw.rename(columns={"country": "series_id", "gwh": "y"}))
    ids = sorted(panel["series_id"].unique())
    # Each country is its own aggregate; the panel is shared by the global models.
    static = pd.DataFrame({"series_id": ids, "target": ids})
    return panel, static


BUILDERS = {"ecuador": _ecuador, "brazil": _brazil, "europe": _europe}


def load(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel and static table, cached as parquet under WORK/panels."""
    p_path = C.PANELS / f"{dataset}_panel.parquet"
    s_path = C.PANELS / f"{dataset}_static.parquet"
    if p_path.exists() and s_path.exists():
        return pd.read_parquet(p_path), pd.read_parquet(s_path)
    panel, static = BUILDERS[dataset]()
    C.PANELS.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(p_path, index=False)
    static.to_parquet(s_path, index=False)
    return panel, static


def actuals(panel: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """Observed value of every target per month, from the raw (unfilled) data."""
    df = panel.merge(static[["series_id", "target"]], on="series_id")
    out = df.groupby(["target", "date"])["y"].sum(min_count=1).rename("y").reset_index()
    return out.dropna()


def admissible(s: pd.Series) -> bool:
    """Admissibility judged only on the history `s` (already cut at the origin)."""
    s = s.loc[s.first_valid_index():] if s.notna().any() else s.iloc[:0]
    if len(s) < C.MIN_MONTHS or s.iloc[-12:].isna().all():
        return False
    bad_years = (s.isna().groupby(s.index.year).sum() > C.MAX_MISSING_PER_YEAR).sum()
    return int(bad_years) <= C.MAX_BAD_YEARS


def history(panel: pd.DataFrame, origin, dataset: str):
    """Training panel as it looked at `origin`.

    Only rows strictly before the origin are kept. Admissibility (Ecuador only)
    and gap filling are computed on that window, so a gap or an outage after the
    origin cannot change which series a model sees or how it is filled.
    Returns the filled panel and the ids of series left out at this origin.
    """
    origin = pd.Timestamp(origin)
    start = pd.Timestamp(C.DATASETS[dataset]["train_start"])
    end = origin - pd.DateOffset(months=1)
    past = panel[(panel["date"] < origin) & (panel["date"] >= start)]
    kept, dropped = [], []
    for sid, g in past.groupby("series_id"):
        # Run every series up to the month before the origin, so one that has
        # stopped reporting shows trailing gaps instead of simply ending early.
        s = g.set_index("date")["y"].asfreq("MS")
        s = s.reindex(pd.date_range(s.index.min(), end, freq="MS"))
        if s.iloc[-12:].isna().all() or (dataset == "ecuador" and not admissible(s)):
            dropped.append(sid)
            continue
        s = s.loc[s.first_valid_index():]
        s = s.interpolate(limit_direction="both")
        kept.append(pd.DataFrame({"series_id": sid, "date": s.index, "y": s.values}))
    hist = pd.concat(kept, ignore_index=True) if kept else past.iloc[:0]
    return hist, dropped


def fallback(panel: pd.DataFrame, ids, origin, horizon: int = C.HORIZON) -> pd.DataFrame:
    """Seasonal-naive forecast for the series a model does not cover at this
    origin, so every model is scored against the same complete aggregate."""
    origin = pd.Timestamp(origin)
    dates = pd.date_range(origin, periods=horizon, freq="MS")
    rows = []
    for sid in ids:
        s = panel[(panel["series_id"] == sid) & (panel["date"] < origin)]
        s = s.set_index("date")["y"].asfreq("MS")
        if s.loc[origin - pd.DateOffset(months=12):].isna().all():
            # Nothing reported for a year: treat the series as discontinued.
            rows += [{"series_id": sid, "date": d, "y_hat": 0.0} for d in dates]
            continue
        for d in dates:
            prev = s.get(d - pd.DateOffset(years=1), np.nan)
            if pd.isna(prev):
                prev = s.dropna().iloc[-1] if s.notna().any() else 0.0
            rows.append({"series_id": sid, "date": d, "y_hat": float(prev)})
    return pd.DataFrame(rows, columns=["series_id", "date", "y_hat"])
