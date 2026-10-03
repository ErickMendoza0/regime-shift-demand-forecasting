# Data

Nothing in `data/raw/` is versioned.

## Ecuador (manual download)

`data/raw/Datos_Energeticos_Ecuador_2014_2024.csv`: monthly billed energy by
company, consumer group, province, canton and parish, January 2014 to December
2024, from the *Facturación Clientes Regulados* module of SISDAT-ARCONEL
(<https://reportes.arconel.gob.ec/>). The portal is interactive, so the yearly
extracts were exported by hand and merged. The file is `latin-1`, `;`-separated,
with `,` as decimal mark, and has about 637,000 rows.

| Column | Example |
|---|---|
| Anio | 2014 |
| Mes | Feb (Spanish abbreviations) |
| Empresa | CNEL-Guayas |
| Grupo Consumo | Residencial, Comercial, Industrial, Alumbrado Público, Otros |
| Provincia, Canton, Parroquia | COTOPAXI, LA MANÁ, EL CARMEN |
| Numero Clientes | 1 |
| Energia Facturada (kWh) | 3974 |
| Facturacion Servicio Electrico (USD) | 321,64 |
| PIB %, Inflacion % | 4,22 |

`src/data.py` sums it to 116 series (province by consumer group) and refuses to
run on a file with fewer than 100,000 rows. During late 2024 these values are
served consumption under scheduled load shedding, not unconstrained demand.

Later months come as yearly Excel extracts from the same module, saved as
`data/raw/ARCONEL_Ecuador_<year>.xlsx` (same columns plus some billing fields,
header on the second row). The 2025 and 2026 files were exported on
3 October 2026; the 2026 one ends in an unfinished August, which the loader
drops.

The extracts have distributor-months with no billing record (zero customers and
zero energy), for example CNEL-Guayas Los Ríos in February 2024. `src/data.py`
treats a distributor billing fewer than half of its customers of the previous
six months as not having reported, blanks the series it bills into for that
month, and leaves the month out of scoring when the missing part exceeds 0.5 %
of the national total. The list is written to `work/panels/ecuador_reporting_gaps.csv`.

## Downloaded by `python -m src.fetch`

| File | Source |
|---|---|
| `oni.ascii.txt` | NOAA CPC Oceanic Niño Index, <https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt> |
| `brazil_ipeadata.csv` | Monthly electricity consumption by region (Eletrobras/EPE) through the Ipeadata API, series `ELETRO12_CEE{NO,NE,SE,SU,CO}12`, GWh |
| `europe_nrg_cb_em.csv` | Eurostat `nrg_cb_em`, electricity available to the internal market (`AIM`, `E7000`), GWh, EU-27 |

ONI values are revised from time to time when NOAA updates its sea-surface
temperature reconstruction; the file used for the paper was downloaded on
3 October 2026.
