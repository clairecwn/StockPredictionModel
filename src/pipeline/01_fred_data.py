"""
Step 1: market-regime and market-mood series from FRED (needs FRED_API_KEY in .env). Run once:
  python src/pipeline/01_fred_data.py
Saves data/raw/public/fred_daily.csv with, per calendar day (forward-filled):
  vix      CBOE VIX (fear gauge)                         VIXCLS
  epu      economic policy uncertainty, daily news index  USEPUINDXD
  hy_oas   high-yield credit spread                       BAMLH0A0HYM2
  umcsent  Michigan consumer sentiment (monthly)          UMCSENT
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import os, requests, pandas as pd
from config import PUBLIC, load_env
load_env()
KEY = os.environ.get("FRED_API_KEY")
assert KEY, "FRED_API_KEY missing in .env"
SERIES = {"vix": "VIXCLS", "epu": "USEPUINDXD", "hy_oas": "BAMLH0A0HYM2", "umcsent": "UMCSENT"}
out = []
for name, sid in SERIES.items():
    r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                     params=dict(series_id=sid, api_key=KEY, file_type="json", observation_start="1998-01-01"), timeout=60)
    if r.status_code != 200:
        raise SystemExit(f"FRED returned HTTP {r.status_code} for {sid}; check FRED_API_KEY")   # never print the URL (has the key)
    d = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    d["date"] = pd.to_datetime(d.date); d[name] = pd.to_numeric(d.value, errors="coerce")
    out.append(d.set_index("date")[[name]]); print(f"  {name:8s} {sid:14s} {d[name].notna().sum():,} observations")
f = pd.concat(out, axis=1).sort_index()
f = f.reindex(pd.date_range(f.index.min(), f.index.max())).ffill()
f.index.name = "date"
f.to_csv(PUBLIC / "fred_daily.csv")
print(f"saved {len(f):,} days -> data/raw/public/fred_daily.csv")
