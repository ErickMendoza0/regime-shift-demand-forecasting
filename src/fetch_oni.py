"""Download the NOAA Oceanic Nino Index (ONI) and store it as data/raw/oni.csv.

ONI is the three-month running-mean sea-surface-temperature anomaly in the
Nino 3.4 region, the standard operational ENSO indicator. The source is public
and needs no authentication:
https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

Run once from a machine with internet access:  python -m src.fetch_oni
"""
import io
import urllib.request

import pandas as pd

import config as C

URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# Three-month season code -> central month of the running mean.
SEASON_MONTH = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}


def main() -> None:
    with urllib.request.urlopen(URL, timeout=60) as r:
        raw = r.read().decode()
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+")
    df.columns = [c.strip() for c in df.columns]
    df["month"] = df["SEAS"].map(SEASON_MONTH)
    df["date"] = pd.to_datetime(df["YR"].astype(str) + "-" + df["month"].astype(str))
    out = df[["date", "ANOM"]].rename(columns={"ANOM": "oni"}).sort_values("date")
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    out.to_csv(C.ONI_CSV, index=False)
    print(f"wrote {len(out)} months to {C.ONI_CSV} "
          f"({out['date'].min():%Y-%m} .. {out['date'].max():%Y-%m})")


if __name__ == "__main__":
    main()
