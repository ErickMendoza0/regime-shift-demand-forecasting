"""Download the public inputs that are not versioned.

Compute nodes on most clusters have no outbound network, so run this once on a
login node:

    python -m src.fetch            # ONI, Brazil (Ipeadata), Europe (Eurostat)
    python -m src.fetch --weights  # also cache the foundation-model weights

The Ecuador billing file cannot be scripted (the ARCONEL portal is interactive);
see data/README.md.
"""
import argparse
import io

import pandas as pd
import requests

import config as C

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
IPEA_URL = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{}')"
EUROSTAT_URL = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
                "nrg_cb_em?format=JSON&lang=EN&nrg_bal=AIM&siec=E7000&unit=GWH")

# Monthly consumption by region (Eletrobras/EPE), GWh.
BRAZIL_REGIONS = {
    "N": "ELETRO12_CEENO12",
    "NE": "ELETRO12_CEENE12",
    "SE": "ELETRO12_CEESE12",
    "S": "ELETRO12_CEESU12",
    "CO": "ELETRO12_CEECO12",
}

EU27 = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR",
        "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO",
        "SE", "SI", "SK"]

WEIGHTS = ["amazon/chronos-2", "google/timesfm-3.0-pytorch", "Salesforce/moirai-2.0-R-small"]


def fetch_oni() -> None:
    r = requests.get(ONI_URL, timeout=60)
    r.raise_for_status()
    C.ONI_TXT.write_text(r.text)
    print(f"ONI -> {C.ONI_TXT}")


def fetch_brazil() -> None:
    frames = []
    for region, code in BRAZIL_REGIONS.items():
        r = requests.get(IPEA_URL.format(code), timeout=120)
        r.raise_for_status()
        v = pd.DataFrame(r.json()["value"])
        frames.append(pd.DataFrame({
            "region": region,
            "date": pd.to_datetime(v["VALDATA"].str[:10]),
            "gwh": pd.to_numeric(v["VALVALOR"], errors="coerce"),
        }))
    out = pd.concat(frames).dropna().sort_values(["region", "date"])
    out.to_csv(C.BRAZIL_CSV, index=False)
    print(f"Brazil: {len(out)} rows -> {C.BRAZIL_CSV}")


def fetch_europe() -> None:
    url = EUROSTAT_URL + "".join(f"&geo={g}" for g in EU27)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    d = r.json()
    # JSON-stat: values are keyed by the flat position in the dimension grid.
    dims, sizes = d["id"], d["size"]
    cats = {}
    for k in dims:
        idx = d["dimension"][k]["category"]["index"]
        cats[k] = sorted(idx, key=idx.get)
    rows = []
    for pos, value in d["value"].items():
        pos = int(pos)
        coord = {}
        for k, n in zip(reversed(dims), reversed(sizes)):
            coord[k] = cats[k][pos % n]
            pos //= n
        rows.append({"country": coord["geo"], "date": coord["time"], "gwh": value})
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["country", "date"])
    out.to_csv(C.EUROPE_CSV, index=False)
    print(f"Europe: {len(out)} rows, {out.country.nunique()} countries -> {C.EUROPE_CSV}")


def fetch_weights() -> None:
    from huggingface_hub import snapshot_download
    for repo in WEIGHTS:
        path = snapshot_download(repo)
        print(f"{repo} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", action="store_true")
    args = ap.parse_args()
    C.RAW.mkdir(parents=True, exist_ok=True)
    fetch_oni()
    fetch_brazil()
    fetch_europe()
    if args.weights:
        fetch_weights()


if __name__ == "__main__":
    main()
