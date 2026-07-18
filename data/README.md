# Data

## 1. `data/raw/Datos_Energeticos_Ecuador_2014_2024.csv` (required, not versioned)

Consolidated billing dataset built from SISDAT-ARCONEL (the *Regulated Customer
Billing* module, <https://reportes.arconel.gob.ec/>) together with
macroeconomic series from the Ministry of Energy and Mines. The file is
`latin-1` encoded, `;`-separated, with `,` as the decimal mark.

The raw headers are Spanish and are mapped to English internal names during
ingestion (`src/prepare_data.py`):

| Raw column | Type | Example | Internal name |
|---|---|---|---|
| Anio | int | 2014 | year |
| Mes | str (Ene..Dic) | Feb | month |
| Empresa | str | CNEL-Guayas | company |
| Grupo Consumo | str | Residencial / Comercial / Industrial / Alumbrado Público / Otros | consumer_group |
| Provincia / Canton / Parroquia | str | COTOPAXI / LA MANÁ / EL CARMEN | province / canton / parish |
| Numero Clientes | int | 1 | n_clients |
| Energia Facturada (kWh) | num | 3974 | energy_kwh |
| Facturacion Servicio Electrico (USD) | num | 321,64 | billing_usd |
| PIB % / Inflacion % | num | 4,22 | gdp_pct / inflation_pct |

Place the file at `data/raw/Datos_Energeticos_Ecuador_2014_2024.csv` before
running the preprocessing step.

## 2. `data/raw/oni.csv` (fetched automatically)

The NOAA/CPC Oceanic Niño Index (ONI), the standard ENSO indicator. Download it
once from a machine with internet access:

```bash
conda activate energyq1 && python -m src.fetch_oni
```
