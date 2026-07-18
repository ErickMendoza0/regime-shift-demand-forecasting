"""Turn the raw SISDAT-ARCONEL billing CSV into a clean sub-series panel plus
the aggregated national series.

Steps:
  * read the latin-1, ';'-separated, ','-decimal source file
  * map month abbreviations to datetimes
  * convert kWh to GWh
  * apply a few known canton -> province corrections
  * attach province centroid coordinates
  * build one series per (consumer group x coordinates), re-index monthly,
    filter by admissibility and interpolate small gaps

    python -m src.prepare_data
"""
import numpy as np
import pandas as pd

import config as C

# Spanish month abbreviations as they appear in the source file.
MONTHS = {'Ene': 1, 'Feb': 2, 'Mar': 3, 'Abr': 4, 'May': 5, 'Jun': 6,
          'Jul': 7, 'Ago': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dic': 12}

# Cantons whose province is mislabelled in the raw data.
PROVINCE_FIXES = {
    'EL PIEDRERO': 'GUAYAS',
    'JUVAL': 'CHIMBORAZO',
    'MATILDE ESTHER': 'GUAYAS',
    'SANTA ROSA DE AGUA CLARA': 'CAÑAR',
    'ZONA NO LIMITADA': 'CHIMBORAZO',
}

# Province -> (latitude, longitude) centroid.
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


def load_raw() -> pd.DataFrame:
    # The source headers are Spanish; rename them to English on ingestion.
    df = pd.read_csv(C.RAW_CSV, encoding='latin-1', sep=';', decimal=',')
    df = df.rename(columns={
        'Anio': 'year', 'Mes': 'month', 'Empresa': 'company',
        'Grupo Consumo': 'consumer_group', 'Provincia': 'province',
        'Canton': 'canton', 'Parroquia': 'parish',
        'Numero Clientes': 'n_clients',
        'Energia Facturada (kWh)': 'energy_kwh',
        'Facturacion Servicio Electrico (USD)': 'billing_usd',
        'PIB %': 'gdp_pct', 'Inflacion %': 'inflation_pct',
    })
    df['date'] = pd.to_datetime(
        df['year'].astype(int).astype(str) + '-'
        + df['month'].map(MONTHS).astype(int).astype(str), format='%Y-%m')
    df['energy_gwh'] = pd.to_numeric(df['energy_kwh'], errors='coerce') / 1_000_000
    for canton, prov in PROVINCE_FIXES.items():
        df.loc[df['canton'].astype(str).str.upper() == canton, 'province'] = prov
    prov = df['province'].astype(str).str.upper().str.strip()
    df['latitude'] = prov.map(lambda x: COORDS.get(x, (np.nan, np.nan))[0])
    df['longitude'] = prov.map(lambda x: COORDS.get(x, (np.nan, np.nan))[1])
    df = df.dropna(subset=['energy_gwh', 'latitude', 'longitude', 'consumer_group'])
    return df[['date', 'consumer_group', 'latitude', 'longitude', 'energy_gwh']]


def admissible(series: pd.Series) -> bool:
    """Keep a series only if it is long enough and has few missing months:
    at most MAX_BAD_YEARS years with more than MAX_MISSING_PER_YEAR gaps."""
    if len(series) < C.MIN_MONTHS:
        return False
    missing_per_year = series.isna().groupby(series.index.year).sum()
    return int((missing_per_year > C.MAX_MISSING_PER_YEAR).sum()) <= C.MAX_BAD_YEARS


def build_subseries(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (df.groupby(['date', 'consumer_group', 'latitude', 'longitude'])
                 ['energy_gwh'].sum().reset_index())
    panels, dropped = [], 0
    keys = grouped[['consumer_group', 'latitude', 'longitude']].drop_duplicates()
    for sid, (_, k) in enumerate(keys.iterrows()):
        sel = grouped[(grouped['consumer_group'] == k['consumer_group'])
                      & (grouped['latitude'] == k['latitude'])
                      & (grouped['longitude'] == k['longitude'])]
        s = sel.set_index('date')['energy_gwh'].sort_index().asfreq('MS')
        if not admissible(s):
            dropped += 1
            continue
        s = s.interpolate(limit_direction='both').fillna(0.0)
        p = s.reset_index()
        p['series_id'] = f"s{sid:03d}"
        p['consumer_group'] = k['consumer_group']
        p['latitude'] = k['latitude']
        p['longitude'] = k['longitude']
        panels.append(p)
    panel = pd.concat(panels, ignore_index=True)
    codes = {g: i for i, g in enumerate(sorted(panel['consumer_group'].unique()))}
    panel['group_code'] = panel['consumer_group'].map(codes)
    print(f"kept {panel['series_id'].nunique()} sub-series, dropped {dropped}")
    return panel


def main() -> None:
    df = load_raw()
    panel = build_subseries(df)
    national = panel.groupby('date')['energy_gwh'].sum().reset_index()

    panel.to_parquet(C.DATA_PROCESSED / 'subseries.parquet', index=False)
    national.to_parquet(C.DATA_PROCESSED / 'national.parquet', index=False)
    print(f"national series: {len(national)} months "
          f"({national['date'].min():%Y-%m} .. {national['date'].max():%Y-%m})")


if __name__ == '__main__':
    main()
