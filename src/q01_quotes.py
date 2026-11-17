"""Live quotes for the page: latest price, today's open/high/low/volume and the 5-minute path since the open.
Runs every 15 minutes during US trading hours (.github/workflows/quotes.yml) and writes docs/data/quotes.json only.
It needs no models and no private data, so it is cheap and safe to run often.
Sources: Yahoo Finance 5-minute bars (delayed about 15 minutes); Finnhub's free quote endpoint fills any ticker Yahoo drops.
Outside trading hours it also records the latest pre-market or after-hours trade, and the time of the next open.
Writes quotes.json (latest prices for every stock) and paths.json next to it (today's price path, every 15 minutes,
loaded only when a stock's "Today" chart is opened).
Run: python src/q01_quotes.py [output path]
"""
import json, os, sys, warnings; warnings.filterwarnings("ignore")
from datetime import datetime, timezone
import pandas as pd, requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WL = json.load(open(os.path.join(ROOT, "watchlist.json")))
CORE = [w["ticker"] for w in WL["stocks"]]
_U = os.path.join(ROOT, "universe.json")
UNIV = [u["ticker"] for u in json.load(open(_U))["stocks"]] if os.path.exists(_U) else []
TICKERS = list(dict.fromkeys(CORE + UNIV + ["SPY"]))            # SPY: the market move shown next to each stock
YMAP = {t: t.replace(".", "-") for t in TICKERS}                  # Yahoo writes BRK.B as BRK-B
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "data", "quotes.json")
now = datetime.now(timezone.utc)
ny = pd.Timestamp(now).tz_convert("America/New_York")
# NYSE full-day holidays (update yearly)
HOLIDAYS = {"2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
            "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"}
trading_day = lambda d: d.weekday() < 5 and str(d.date()) not in HOLIDAYS
mins = ny.hour * 60 + ny.minute
is_open = trading_day(ny) and 9 * 60 + 30 <= mins < 16 * 60
session = ("open" if is_open else "pre" if trading_day(ny) and 4 * 60 <= mins < 9 * 60 + 30
           else "post" if trading_day(ny) and 16 * 60 <= mins < 20 * 60 else "closed")
nxt = ny.normalize() + pd.Timedelta(hours=9, minutes=30)
if mins >= 9 * 60 + 30 or not trading_day(ny): nxt += pd.Timedelta(days=1)
while not trading_day(nxt): nxt += pd.Timedelta(days=1)
regular = lambda ix: (ix.hour * 60 + ix.minute >= 9 * 60 + 30) & (ix.hour * 60 + ix.minute < 16 * 60)

quotes = {}
try:
    import yfinance as yf
    d = yf.download(list(YMAP.values()), period="5d", interval="5m", prepost=True, progress=False, group_by="ticker", auto_adjust=False, threads=True)
    for t in TICKERS:
        try:
            x = d[YMAP[t]].dropna(subset=["Close"])
        except KeyError:
            continue
        if not len(x): continue
        ix_all = x.index.tz_convert("America/New_York")
        ext = x[~regular(ix_all)]; ext_ix = ix_all[~regular(ix_all)]
        x = x[regular(ix_all)]
        if not len(x): continue
        idx = x.index.tz_convert("America/New_York")
        last_day = idx[-1].date()
        today, before = x[idx.date == last_day], x[idx.date < last_day]
        prev = float(before.Close.iloc[-1]) if len(before) else None
        quotes[t] = {"price": float(today.Close.iloc[-1]), "prev_close": prev, "open": float(today.Open.iloc[0]),
                     "high": float(today.High.max()), "low": float(today.Low.min()), "volume": float(today.Volume.sum()),
                     "time": idx[-1].isoformat(), "session_date": str(last_day),
                     "path": [[t_.strftime("%H:%M"), round(float(c), 2)] for t_, c in zip(idx[idx.date == last_day], today.Close)]}
        late = ext[ext_ix > idx[-1]]                                 # trades after the last regular bar (after hours, then pre-market)
        if len(late):
            lt = ext_ix[ext_ix > idx[-1]][-1]
            quotes[t]["ext"] = {"price": float(late.Close.iloc[-1]), "change_pct": 100 * (float(late.Close.iloc[-1]) / quotes[t]["price"] - 1),
                                "session": "pre-market" if lt.hour < 9 or lt.date() > last_day else "after hours", "time": lt.isoformat()}
except Exception as e:
    print("Yahoo intraday failed:", e)

key = os.environ.get("FINNHUB_API_KEY")
for t in CORE + ["SPY"]:                                         # the Finnhub fallback is kept to the hand-picked stocks (rate limit)
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
paths = {t: q.pop("path")[::3] for t, q in quotes.items() if q.get("path")}       # every 15 minutes is enough for the small chart
json.dump(paths, open(os.path.join(os.path.dirname(OUT), "paths.json"), "w"), separators=(",", ":"))
json.dump({"generated_utc": now.isoformat(timespec="minutes"), "market_open": is_open, "session": session,
           "next_open_utc": nxt.tz_convert("UTC").isoformat(),
           "close_utc": (ny.normalize() + pd.Timedelta(hours=16)).tz_convert("UTC").isoformat() if is_open else None, "quotes": quotes}, open(OUT, "w"))
print(f"quotes for {len(quotes)} of {len(TICKERS)} tickers ({sum('ext' in q for q in quotes.values())} with extended hours); session {session}; next open {nxt} -> {OUT}")
