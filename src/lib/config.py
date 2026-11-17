"""Paths, settings and .env loading shared by every script."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
WRDS = DATA / "raw" / "wrds"          # files you download from the WRDS website go here
PUBLIC = DATA / "raw" / "public"
PROC = DATA / "processed"
LIVE = os.environ.get("LIVE_FEATS") == "1"      # live mode: models trained only on features computable from free live data
RESULTS = ROOT / "results" / ("live_models" if LIVE else "")
MODELS = ROOT / "models" / ("live" if LIVE else "")
OOS_FILE = "oos_predictions_live.parquet" if LIVE else "oos_predictions.parquet"
for p in (WRDS, PUBLIC, PROC, RESULTS, MODELS):
    p.mkdir(parents=True, exist_ok=True)

def load_env():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
load_env()

def find_file(stem):
    """Return data/raw/wrds/<stem>.csv, .csv.gz or .zip, whichever exists."""
    for ext in (".csv", ".csv.gz", ".zip", ".txt"):
        f = WRDS / f"{stem}{ext}"
        if f.exists():
            return f
    return None

EST_WINDOW = (-250, -30)    # trading days used to estimate each stock's normal relation to the market
HORIZONS = (5, 20, 60)      # drift horizons (trading days after the event)
FIRST_EVENT_YEAR = 2001
