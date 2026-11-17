"""Live quotes for the page: latest price, today's open/high/low/volume and the 5-minute path since the open.
Runs every 15 minutes during US trading hours (.github/workflows/quotes.yml) and writes docs/data/quotes.json only.
It needs no models and no private data, so it is cheap and safe to run often.
Sources: Yahoo Finance 5-minute bars (delayed about 15 minutes); Finnhub's free quote endpoint fills any ticker Yahoo drops.
Run: python src/q01_quotes.py [output path]
"""
import json, os, sys, warnings; warnings.filterwarnings("ignore")
from datetime import datetime, timezone
import pandas as pd, requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WL = json.load(open(os.path.join(ROOT, "watchlist.json")))
TICKERS = [w["ticker"] for w in WL["stocks"]]
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "data", "quotes.json")
now = datetime.now(timezone.utc)
ny = pd.Timestamp(now).tz_convert("America/New_York")
is_open = ny.weekday() < 5 and (9 * 60 + 30) <= ny.hour * 60 + ny.minute < 16 * 60

quotes = {}
try:
    import yfinance as yf
    d = yf.download(TICKERS, period="5d", interval="5m", progress=False, group_by="ticker", auto_adjust=False, threads=True)
    for t in TICKERS:
        try:
            x = d[t].dropna(subset=["Close"])
        except KeyError:
            continue
        if not len(x): continue
        idx = x.index.tz_convert("America/New_York")
        last_day = idx[-1].date()
        today, before = x[idx.date == last_day], x[idx.date < last_day]
        prev = float(before.Close.iloc[-1]) if len(before) else None
        quotes[t] = {"price": float(today.Close.iloc[-1]), "prev_close": prev, "open": float(today.Open.iloc[0]),
                     "high": float(today.High.max()), "low": float(today.Low.min()), "volume": float(today.Volume.sum()),
                     "time": idx[-1].isoformat(), "session_date": str(last_day),
                     "path": [[t_.strftime("%H:%M"), round(float(c), 2)] for t_, c in zip(idx[idx.date == last_day], today.Close)]}
except Exception as e:
    print("Yahoo intraday failed:", e)

key = os.environ.get("FINNHUB_API_KEY")
for t in TICKERS:
    if t in quotes or not key: continue
    try:
        q = requests.get("https://finnhub.io/api/v1/quote", params={"symbol": t, "token": key}, timeout=15).json()
        if q.get("c"):
            quotes[t] = {"price": q["c"], "prev_close": q.get("pc"), "open": q.get("o"), "high": q.get("h"), "low": q.get("l"),
                         "volume": None, "time": datetime.fromtimestamp(q["t"], timezone.utc).isoformat(), "path": []}
    except Exception as e:
        print(f"Finnhub quote failed for {t}: {e}")

for q in quotes.values():
    if q.get("prev_close"):
        q["change_pct"] = 100 * (q["price"] / q["prev_close"] - 1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"generated_utc": now.isoformat(timespec="minutes"), "market_open": is_open, "quotes": quotes}, open(OUT, "w"))
print(f"quotes for {len(quotes)} of {len(TICKERS)} tickers; market {'open' if is_open else 'closed'} -> {OUT}")
