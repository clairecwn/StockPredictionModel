"""Download FINRA daily short-sale volume (Reg SHO) and build one compact table.

Free, public data. Per stock per day: short volume / total volume (the daily short ratio).
  - 2018-08-01 onward: one consolidated file per day (CNMS).
  - Before that: sum of the trade-reporting-facility files (FNSQ, FNYX, FORF, FNQC).
Runs on your own computer (it needs normal internet). Safe to stop and re-run: finished days are cached.

Output: data/raw/public/finra_shortvol.parquet  (date, symbol, short_vol, total_vol)
Usage:  python src/s03d_finra_shortvol.py [start_year]      (default 2009)
"""
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests

from config import PUBLIC

BASE = "https://cdn.finra.org/equity/regsho/daily/{pfx}shvol{d}.txt"
CNMS_START = date(2018, 8, 1)
CACHE = PUBLIC / "finra_cache"
CACHE.mkdir(parents=True, exist_ok=True)
SES = requests.Session()
SES.headers["User-Agent"] = "research-project (student)"


def fetch(pfx, d):
    url = BASE.format(pfx=pfx, d=d.strftime("%Y%m%d"))
    for attempt in range(3):
        try:
            r = SES.get(url, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text), sep="|")
            df = df[pd.to_numeric(df["ShortVolume"], errors="coerce").notna()]
            return df[["Symbol", "ShortVolume", "TotalVolume"]].astype(
                {"ShortVolume": float, "TotalVolume": float})
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None


def one_day(d):
    f = CACHE / f"{d:%Y%m%d}.csv.gz"
    if f.exists():
        return d, "cached"
    pfxs = ["CNMS"] if d >= CNMS_START else ["FNSQ", "FNYX", "FORF", "FNQC"]
    parts = [x for x in (fetch(p, d) for p in pfxs) if x is not None]
    if not parts:
        f.write_bytes(b"")          # marks weekend/holiday/no file so we do not retry it
        return d, "none"
    g = pd.concat(parts).groupby("Symbol", as_index=False).sum()
    g.insert(0, "date", pd.Timestamp(d))
    g.to_csv(f, index=False)
    return d, "ok"


def main():
    start_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2009
    d, days = date(start_year, 1, 1), []
    while d <= date.today():
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    print(f"{len(days):,} weekdays to check (cached days are skipped)")
    done = 0
    with ThreadPoolExecutor(max_workers=20) as ex:
        for _, status in ex.map(one_day, days):
            done += 1
            if done % 250 == 0:
                print(f"  {done:,}/{len(days):,} days processed")
    frames = []
    for f in sorted(CACHE.glob("*.csv.gz")):
        if f.stat().st_size > 20:
            frames.append(pd.read_csv(f, parse_dates=["date"]))
    out = pd.concat(frames, ignore_index=True)
    out.columns = ["date", "symbol", "short_vol", "total_vol"]
    out.to_parquet(PUBLIC / "finra_shortvol.parquet", index=False)
    print(f"saved {len(out):,} rows, {out['date'].min():%Y-%m-%d} to {out['date'].max():%Y-%m-%d}, "
          f"{out['symbol'].nunique():,} symbols")


if __name__ == "__main__":
    main()
